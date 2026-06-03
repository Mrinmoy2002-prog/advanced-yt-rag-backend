from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import config

class TranscriptChunker:
    """Splits video transcripts dynamically based on semantic meaning and meaning containing topic transitions."""
    
    @staticmethod
    def split_transcript(transcript_data: list[dict], video_id: str) -> list[Document]:
        documents = []
        
        for entry in transcript_data:
            doc = Document(
                page_content=entry['text'],
                metadata={
                    "video_id": video_id,
                    "start_time": entry['start'],
                    "source_url": f"https://www.youtube.com/watch?v={video_id}&t={int(entry['start'])}s"
                }
            )
            documents.append(doc)
            
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        semantic_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile"
        )
        
        print("🧠 Analyzing text semantic distances and calculating conversational splits...")
        return semantic_splitter.split_documents(documents)
    

if __name__ == "__main__":
    sample_transcript = [
        {"text": "Welcome to the video on advanced RAG techniques.", "start": 0},
        {"text": "In this section, we will cover semantic chunking.", "start": 10},
        {"text": "Semantic chunking allows us to split text based on meaning rather than just length.", "start": 20},
        {"text": "This is particularly useful for videos with multiple topics.", "start": 30},
        {"text": "Let's dive into how to implement this in Python.", "start": 40}
    ]
    
    video_id = "h3Tv7t-zn2o"
    chunks = TranscriptChunker.split_transcript(sample_transcript, video_id)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"Content: {chunk.page_content}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)