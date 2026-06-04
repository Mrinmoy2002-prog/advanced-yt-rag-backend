from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import config

class PineconeIndexer:
    """Handles connecting to Pinecone and uploading text embeddings."""
    
    def __init__(self):
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        # MiniLM model outputs 384 dimensions, so the index must match
        self._ensure_index_exists(dimension=384)
        
        # Connect directly to the target index
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)

    def _ensure_index_exists(self, dimension: int):
        """Creates the index if it doesn't already exist in the Pinecone account."""
        existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]
        
        if config.PINECONE_INDEX_NAME not in existing_indexes:
            print(f"🌲 Index '{config.PINECONE_INDEX_NAME}' not found. Initializing serverless instance...")
            
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=dimension,
                metric="cosine",  
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # it is completely free to create serverless indexes on Pinecone and can handle parallel upserts without needing to manage capacity
            )
            print("✅ Index built successfully.")

    def index_documents(self, documents: list[Document]):
        """Embeds text documents and uploads them to Pinecone in a single batch."""
        vectors_to_upsert = []
        
        print(f"✨ Generating vector embeddings for {len(documents)} semantic chunks...")
        
        for i, doc in enumerate(documents):
            vector_values = self.embeddings.embed_query(doc.page_content)
            
            vectors_to_upsert.append({
                # Create a unique ID combining the video ID and loop index
                "id": f"{doc.metadata['video_id']}_{i}",
                "values": vector_values,
                "metadata": {
                    "text": doc.page_content,  # Save the raw text so the LLM can read it later
                    **doc.metadata             # Unpack other metadata like timestamps or source URLs from the chunk documents
                }
            })
            
        print("🚀 Streaming batch updates into Pinecone...")
        self.index.upsert(vectors=vectors_to_upsert)  # upsert - Read + write in one operation, so it will create new vectors or update existing ones based on the ID
        print("🎉 Vector indexing completed successfully!")