import os
import csv
from faster_whisper import WhisperModel
from tqdm import tqdm

# Configuration
AUDIO_DIR = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9"
OUTPUT_DIR = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9_transcriptions"
METADATA_FILE = os.path.join(OUTPUT_DIR, "transcriptions_metadata.csv")
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16" # Optimized for 4GB VRAM
LANGUAGE = "ar"
INITIAL_PROMPT = "Egyptian Arabic slang transcription. عامية مصرية"

def transcribe_batch():
    # Load Model
    print(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}...")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.wav', '.mp3', '.m4a'))]
    print(f"Found {len(audio_files)} audio files in {AUDIO_DIR}")
    
    # Check already transcribed files (from TXT exports)
    existing_txts = {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith('.txt')}
    
    # Check already transcribed files (from CSV metadata)
    existing_csv_targets = set()
    if os.path.isfile(METADATA_FILE):
        try:
            with open(METADATA_FILE, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                existing_csv_targets = {row['filename'] for row in reader if row.get('filename')}
        except Exception:
            pass

    # Filter audio files to process
    to_process = []
    skipped_count = 0
    for filename in audio_files:
        name_no_ext = os.path.splitext(filename)[0]
        if name_no_ext in existing_txts or filename in existing_csv_targets:
            skipped_count += 1
        else:
            to_process.append(filename)
            
    print(f"⏭️  Skipped {skipped_count} already transcribed files.")
    print(f"🚀 Processing {len(to_process)} remaining files...")
    
    if not to_process:
        print("✅ All files already transcribed. Nothing to do.")
        return

    # Initialize metadata CSV if it doesn't exist
    file_exists = os.path.isfile(METADATA_FILE)
    
    with open(METADATA_FILE, mode='a', encoding='utf-8-sig', newline='') as csvfile:
        fieldnames = ['filename', 'transcription']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            
        # Process files
        for filename in tqdm(to_process, desc="Transcribing"):
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            audio_path = os.path.join(AUDIO_DIR, filename)
            
            try:
                # Transcribe
                segments, info = model.transcribe(
                    audio_path, 
                    language=LANGUAGE, 
                    initial_prompt=INITIAL_PROMPT,
                    beam_size=5
                )
                
                transcription = " ".join([segment.text for segment in segments]).strip()
                
                # Save individual txt file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                
                # Save to metadata CSV
                writer.writerow({'filename': filename, 'transcription': transcription})
                csvfile.flush() # Ensure data is written
                
            except Exception as e:
                print(f"\nError transcribing {filename}: {str(e)}")

if __name__ == "__main__":
    transcribe_batch()
