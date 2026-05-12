import json
import os
from youtube_transcript_api import YouTubeTranscriptApi

def main():
    # The ID from your URL: https://www.youtube.com/watch?v=bUXgtEkvHTQ
    video_id = "bUXgtEkvHTQ" 
    output_file = "youtube_transcript.json"

    print(f"--- Fetching Arabic transcript for video ID: {video_id}...")

    try:
        # Fetch the Arabic transcript
        # We try 'ar' first. If it fails, we catch the exception.
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ar'])
        
        dataset_chunks = []
        
        for segment in transcript:
            start_time = segment['start']
            end_time = start_time + segment['duration']
            text = segment['text']
            
            # Create a clean chunk
            chunk_data = {
                "text": text,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "video_id": video_id
            }
            dataset_chunks.append(chunk_data)

        # Save to JSON for our Vector Search later
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_chunks, f, ensure_ascii=False, indent=4)

        print(f"DONE: Extracted {len(dataset_chunks)} chunks.")
        print(f"FILE: Saved to: {os.path.abspath(output_file)}")
        
        # Show a preview
        print("\n--- Preview (First 3 chunks):")
        for chunk in dataset_chunks[:3]:
            print(f"  [{chunk['start_time']} - {chunk['end_time']}] {chunk['text']}")

    except Exception as e:
        print(f"ERROR: {e}")
        print("Note: Some videos do not have Arabic subtitles enabled.")

if __name__ == "__main__":
    main()
