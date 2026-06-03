import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Open Source Model Configs (Hugging Face Hub)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# RAG Hyperparameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Database Configs
PINECONE_INDEX_NAME = "youtube-rag-index"