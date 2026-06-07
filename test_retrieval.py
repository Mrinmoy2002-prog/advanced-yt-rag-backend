import sys
from retriever.search import AdvancedRetriever

def run_retrieval_test():
    try:
        print("\n🔍 Initializing the Advanced Retriever Testing Suite...")
        # 1. Initialize the retriever (This will also download/load the local Cross-Encoder model)
        retriever = AdvancedRetriever()
        print("✅ Retriever and Re-ranker models loaded successfully.")

        # 2. Define a test question that uses a vague pronoun ("it") to test our optimizer
        user_question = "What did he say about it? Tell me about vector automation solutions."
        target_video = "test_video_123"
        
        print("\n" + "="*70)
        print(f"📥 Incoming Vague User Question: '{user_question}'")
        print(f"🎥 Targeting Video ID Filter:   '{target_video}'")
        print("="*70)
        
        # 3. Trigger the full retrieval and re-ranking pipeline
        # We ask for top_k = 2 results
        hits = retriever.retrieve_context(query=user_question, video_id=target_video, top_k=2)
        
        # 4. Display the final, re-ranked results
        print(f"\n📋 Final Re-ranked Results (Top {len(hits)} Matches):")
        for idx, hit in enumerate(hits):
            print(f"\n[Rank {idx+1}] Cross-Encoder Score: {hit['score']:.4f}")
            print(f"⏰ Start Location: {hit['start_time']} seconds")
            print(f"🔗 Live Source URL: {hit['source_url']}")
            print(f"📝 Text Context:    {hit['text']}")
        print("\n" + "="*70)
        print("✅ TEST PASSED: Retrieval pipeline executed completely!")

    except Exception as e:
        print(f"\n❌ RETRIEVAL TEST FAILED: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    run_retrieval_test()