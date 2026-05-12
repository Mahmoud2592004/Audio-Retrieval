import json
import re
import os

def time_to_seconds(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0

def process_transcript(input_file, output_file, video_id):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    chunks = []
    current_time = None
    current_text = []

    for line in lines:
        # Check if line is a timestamp (e.g., 0:00, 12:34, 1:23:45)
        if re.match(r'^\d+:\d+(:\d+)?$', line):
            if current_time is not None:
                # Save previous chunk
                chunks.append({
                    "text": " ".join(current_text),
                    "start_time": current_time,
                    "video_id": video_id
                })
            current_time = time_to_seconds(line)
            current_text = []
        else:
            current_text.append(line)

    # Add the last chunk
    if current_time is not None:
        chunks.append({
            "text": " ".join(current_text),
            "start_time": current_time,
            "video_id": video_id
        })

    # Calculate end_times
    for i in range(len(chunks) - 1):
        chunks[i]["end_time"] = chunks[i+1]["start_time"]
    
    # Estimate end_time for the last chunk (+5 seconds or similar)
    if chunks:
        chunks[-1]["end_time"] = chunks[-1]["start_time"] + 5

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    return len(chunks)

if __name__ == "__main__":
    v_id = "bUXgtEkvHTQ"
    count = process_transcript("youtube_transcript.txt", "youtube_transcript.json", v_id)
    print(f"Processed {count} chunks into youtube_transcript.json")
