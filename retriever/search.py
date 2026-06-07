from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from sentence_transformers import CrossEncoder
import config

class AdvancedRetriever:
    """Handles expanding a user's question and sorting the results using a re-ranker."""

    def __init__(self):
        # Connect to our cloud database and embedding model
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.hf_client = InferenceClient(token=config.HF_TOKEN)

        # Load the smart re-ranker model onto our local machine
        print("🧠 Loading local Cross-Encoder Re-ranker (BAAI/bge-reranker-base)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")


    def optimize_and_expand_query(self, original_query: str) -> list[str]:
        """
        Fixes vague questions and creates 3 alternative ways to ask the 
        same thing using Hugging Face's conversational API format.
        """
        # Define our system instructions and user message as a structured chat history
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Retrieval-Augmented Generation (RAG) Query Optimizer.\n"
                    "Your task is to analyze the user's input question and generate 3 distinct search queries.\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "- If the input is vague, short, or uses ambiguous pronouns ('he', 'that company', 'this system'), "
                    "expand it to be self-contained and clear.\n"
                    "- Provide exactly 3 variations, one per line.\n"
                    "- Do not include numbers, bullet points, markdown, or introductory text."
                )
            },
            {
                "role": "user",
                "content": f"Original Question: {original_query}"
            }
        ]
        
        try:
            # We shift from text_generation to chat_completion -> as it is a conversational task specific model
            response = self.hf_client.chat_completion(
                messages=messages,
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                max_tokens=150,
                temperature=0.4
            )
            
            # Extract the raw text out of the generated chat message object
            raw_text = response.choices[0].message.content
            
            # Split the model's response into a clean list of strings
            queries = [q.strip() for q in raw_text.strip().split("\n") if q.strip()]
            
            # Always add the original question to ensure core intent is preserved
            queries.append(original_query)
            return list(set(queries))
            
        except Exception as e:
            print(f"⚠️ Query optimization failed, using raw baseline input. Error: {e}")
            return [original_query]
        

    def retrieve_context(self, query: str, video_id: str, top_k: int = 4) -> list[dict]:

        # Step 1: Fix up and expand the original question
        optimized_queries = self.optimize_and_expand_query(query)
        unique_results = {}

        print(f"\n🧠 Optimized Query Variations: {optimized_queries}")
        print(f"🔍 Executing domain-isolated searches inside video context: '{video_id}'...")
        
        # Step 2: Grab matching candidates from Pinecone for all query variations
        for q in optimized_queries:
            query_vector = self.embeddings.embed_query(q)

            # We fetch 3x more results than requested so the re-ranker has a good pool to pick from
            response = self.index.query(
                vector=query_vector,
                top_k=top_k * 3, 
                include_metadata=True,
                filter={"video_id": {"$eq": video_id}} # Lock search strictly to this specific video ID
            )

            # Deduplicate the results across our different query runs using their unique IDs
            for match in response.get("matches", []):
                doc_id = match["id"]
                if doc_id not in unique_results:
                    unique_results[doc_id] = {
                        "text": match["metadata"]["text"],
                        "start_time": match["metadata"]["start_time"],
                        "source_url": match["metadata"]["source_url"]
                    }

        candidate_chunks = list(unique_results.values())
        if not candidate_chunks:
            return []

        # Step 3: Deep analysis using the Cross-Encoder re-ranker
        print(f"🔥 Re-ranking {len(candidate_chunks)} candidates using Cross-Encoder...")

        # Pair the original user query side-by-side with every single text chunk
        query_text_pairs = [[query, chunk["text"]] for chunk in candidate_chunks]

        # The model reads the query and text chunk TOGETHER to calculate an ultra-accurate score
        rerank_scores = self.reranker.predict(query_text_pairs)

        # Replace the simple vector database scores with our new high-precision scores
        for idx, score in enumerate(rerank_scores):
            candidate_chunks[idx]["score"] = float(score)

        # Step 4: Sort everything from highest to lowest score based on the re-ranker's judgment
        sorted_hits = sorted(candidate_chunks, key=lambda x: x["score"], reverse=True)
        
        # Cut the list off and return just the top matches the user asked for
        return sorted_hits[:top_k]