import config
from huggingface_hub import InferenceClient
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from sentence_transformers import CrossEncoder


class AdvancedRetriever:
    """Handles expanding a user's question, filtering out redundancy with MMR, and sorting with a re-ranker."""

    def __init__(self):
        # Establish connections to our database, baseline embedding model, and cloud LLM
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.hf_client = InferenceClient(token=config.HF_TOKEN)

        # Load a high-precision sorting model right onto our local machine
        print("🧠 Loading local Cross-Encoder Re-ranker (BAAI/bge-reranker-base)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

    def optimize_and_expand_query(self, original_query: str) -> list[str]:
        """
        Fixes vague questions and generates 3 alternative phrasings 
        to ensure our search doesn't miss key information.
        """
        # Set up the exact structure that Llama-3 expects for chat completions
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
            # Send the request to our conversational cloud model
            response = self.hf_client.chat_completion(
                messages=messages,
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                max_tokens=150,
                temperature=0.4
            )
            raw_text = response.choices[0].message.content
            
            # Parse out the response text into individual query strings
            queries = [q.strip() for q in raw_text.strip().split("\n") if q.strip()]
            
            # Keep the original query in the mix so we don't drift from the core intent
            queries.append(original_query)
            return list(set(queries)) # Use a set to drop any duplicate phrasings automatically
            
        except Exception as e:
            print(f"⚠️ Query optimization failed, using raw baseline input. Error: {e}")
            return [original_query]

    def _calculate_mmr(self, query_vector: list, candidate_chunks: list, fetch_k: int, lambda_mult: float = 0.5) -> list:
        """Balances query relevance with information diversity using LangChain's built-in utility."""
        if not candidate_chunks:
            return []

        # Convert our raw text chunks into embeddings so the library can calculate distance metrics
        chunk_texts = [c["text"] for c in candidate_chunks]
        chunk_vectors = self.embeddings.embed_documents(chunk_texts)

        # Let the library find chunks that are relevant to the query but different from each other
        selected_indices = maximal_marginal_relevance(
            query_embedding=query_vector,
            embedding_list=chunk_vectors,
            lambda_mult=lambda_mult,
            k=min(fetch_k, len(candidate_chunks))
        )

        # Extract and return the filtered, diverse chunk objects
        return [candidate_chunks[i] for i in selected_indices]

    def retrieve_context(self, query: str, video_id: str, top_k: int = 4) -> list[dict]:
        # Step 1: Pre-Retrieval (Generate alternative query angles)
        optimized_queries = self.optimize_and_expand_query(query)
        unique_results = {}

        print(f"\n🧠 Optimized Query Variations: {optimized_queries}")
        print(f"🔍 Fetching broad initial candidate pool from Pinecone...")
        
        # Step 2: In-Retrieval (Cast a wide net in Pinecone using all query variations)
        for q in optimized_queries:
            query_vector = self.embeddings.embed_query(q)

            # Over-fetch entries (top_k * 4) so we have plenty of data to filter down later
            response = self.index.query(
                vector=query_vector,
                top_k=top_k * 4, 
                include_metadata=True,
                filter={"video_id": {"$eq": video_id}} # Keep our search isolated strictly to this video
            )

            # Deduplicate incoming records by their unique vector IDs
            for match in response.get("matches", []):
                doc_id = match["id"]
                if doc_id not in unique_results:
                    unique_results[doc_id] = {
                        "text": match["metadata"]["text"],
                        "start_time": match["metadata"]["start_time"],
                        "source_url": match["metadata"]["source_url"]
                    }

        raw_candidates = list(unique_results.values())
        if not raw_candidates:
            return []

        # Step 3: During-Retrieval Filtering (Drop repetitive context using the original query)
        print(f"🛡️ Applying MMR library filter to drop redundant information from {len(raw_candidates)} tracks...")
        original_query_vector = self.embeddings.embed_query(query)
        
        # Filter the pool down to (top_k * 2) unique, information-rich entries
        diverse_candidates = self._calculate_mmr(
            query_vector=original_query_vector, 
            candidate_chunks=raw_candidates, 
            fetch_k=top_k * 2,
            lambda_mult=0.5 # 0.5 splits focus evenly between similarity (i.e, semantic search wise) and diversity
        )

        # Step 4: Post-Retrieval Scoring (Judge chunk fit using the original query)
        print(f"🔥 Re-ranking {len(diverse_candidates)} unique candidates using Cross-Encoder...")
        
        # Pair the core question directly side-by-side with each unique text block
        query_text_pairs = [[query, chunk["text"]] for chunk in diverse_candidates]
        
        # Calculate precise relevance scores by processing the question and text together
        rerank_scores = self.reranker.predict(query_text_pairs)

        # Assign the new smart scores back to our candidates
        for idx, score in enumerate(rerank_scores):
            diverse_candidates[idx]["score"] = float(score)

        # Step 5: Final Sort (Order everything from highest to lowest score based on re-ranker judgment)
        sorted_hits = sorted(diverse_candidates, key=lambda x: x["score"], reverse=True)
        
        # Cut the list off and return just the top matches the user wants
        return sorted_hits[:top_k]