# import os
# import torch
# import yt_dlp
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# import os
# import sys

# # Inject your exact FFmpeg binary path into the system environment at runtime
# FFMPEG_DIR = r"C:\Users\mrinm\Downloads\ffmpeg-2026-06-01-git-bf608f16fd-essentials_build\ffmpeg-2026-06-01-git-bf608f16fd-essentials_build\bin"
# if FFMPEG_DIR not in os.environ["PATH"]:
#     os.environ["PATH"] += os.pathsep + FFMPEG_DIR

# class YouTubeTranscriptFetcher:
#     """
#     Downloads YouTube video audio natively and utilizes Whisper to translate 
#     and transcribe foreign or local speech into clean English text.
#     """

#     @staticmethod
#     def _download_audio(video_id: str) -> str:
#         """Downloads the best quality audio stream from YouTube and saves it as a local file."""
#         video_url = f"https://www.youtube.com/watch?v={video_id}"
#         output_filename = f"{video_id}.mp3"

#         ydl_opts = {
#             'format': 'bestaudio/best',
#             'outtmpl': output_filename,
#             'postprocessors': [{
#                 'key': 'FFmpegExtractAudio',
#                 'preferredcodec': 'mp3',
#                 'preferredquality': '192',  
#             }],
#             'quiet': True, 
#         }


#         print(f"📥 Downloading audio stream for video: {video_id}...")
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([video_url])
            
#         return output_filename
    

#     @classmethod
#     def get_transcript(cls, video_id: str) -> list[dict]:
#         """Runs the offline speech-to-text pipeline to generate an English transcript with timestamps."""
#         audio_path = cls._download_audio(video_id)

#         # 1. Determine running hardware environment (GPU if available, else standard CPU)
#         device = "cuda:0" if torch.cuda.is_available() else "cpu"
#         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
#         model_id = "distil-whisper/distil-large-v3"

#         print(f"🧠 Initializing open-source {model_id} on engine mapping: [{device}]...")

#         # 2. Allocate memory weights efficiently using Auto Classes
#         model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
#         ).to(device)

#         processor = AutoProcessor.from_pretrained(model_id)

#         # 3. Formulate the Hugging Face Pipeline
#         pipe = pipeline(
#             "automatic-speech-recognition",
#             model=model,
#             tokenizer=processor.tokenizer,
#             feature_extractor=processor.feature_extractor,
#             chunk_length_s=30,
#             batch_size=16,
#             torch_dtype=torch_dtype,
#             device=device
#         )

#         print("🎙️ Translating and processing audio timelines (This may take a moment)...")

#         # 4. Request explicit sentence level timestamps and target English translation natively
#         result = pipe(
#             audio_path, 
#             return_timestamps=True, 
#             generate_kwargs={"task": "translate"}
#         )

#         # Clean up the downloaded local audio stream file from space
#         if os.path.exists(audio_path):
#             os.remove(audio_path)

#         # 5. Format the payload arrays so they perfectly map to your downstream Chunker module
#         formatted_transcript = []
#         for chunk in result["chunks"]:
#             formatted_transcript.append({
#                 "text": chunk["text"].strip(),
#                 "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0
#             })

#         print("✅ Translation and transcription tasks finalized successfully.")
#         return formatted_transcript
    


# if __name__ == "__main__":
#     # Feel free to change this to any YouTube Video ID you want to test
#     SAMPLE_VIDEO_ID = "h3Tv7t-zn2o" 
    
#     # Run the fetcher
#     transcript_results = YouTubeTranscriptFetcher.get_transcript(SAMPLE_VIDEO_ID)
    
#     # Print out the structured results to verify everything works
#     print(f"\n📋 Extracted {len(transcript_results)} timeline entries:\n")
#     for item in transcript_results[:10]:  # Showcases the first 10 entries in terminal
#         print(f"[{item['start']:.2f}s] {item['text']}")