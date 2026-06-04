import sys
from langchain_core.documents import Document
from ingestion.indexer import PineconeIndexer

def run_database_test():
    try:
        print("\n🌲 Initializing the Pinecone Indexer Testing Suite...")
        # 1. Initialize your indexer class (This runs connection check and index creation logic)
        indexer = PineconeIndexer()
        print("🚀 Connection established successfully.")

        # 2. Construct Mock Semantic Documents to emulate your chunker's output payload
        print("\n📦 Generating mock semantic document chunks for testing...")
        mock_docs = [
            Document(
                page_content="This is a test chunk explaining machine learning and vector data storage solutions.",
                metadata={
                    "video_id": "test_video_123",
                    "start_time": 0.0,
                    "source_url": "https://www.youtube.com/watch?v=test_video_123&t=0s"
                }
            ),
            Document(
                page_content="In this second block, we evaluate advanced RAG retrieval patterns and pipeline automation engineering.",
                metadata={
                    "video_id": "test_video_123",
                    "start_time": 15.5,
                    "source_url": "https://www.youtube.com/watch?v=test_video_123&t=15s"
                }
            )
        ]

        # 3. Trigger the vector indexing method
        print("\n📤 Initiating vector conversion and cloud upsert protocol...")
        indexer.index_documents(mock_docs)
        
        print("\n✅ TEST PASSED: Check your Pinecone Web Console dashboard to see the records!")
        print("="*65 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    run_database_test()