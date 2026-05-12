import os
import glob
import pandas as pd
from datasets import Dataset, Audio
from typing import Optional

class EgyptDatasetBuilder:
    """
    Scans audio and transcription folders to build a HuggingFace Dataset.
    """
    def __init__(self, audio_dir: str, trans_dir: str):
        self.audio_dir = audio_dir
        self.trans_dir = trans_dir
        
    def build(self) -> Dataset:
        data = []
        audio_files = glob.glob(os.path.join(self.audio_dir, "*.wav"))
        
        print(f"📦 Found {len(audio_files)} audio files. Matching with transcriptions...")
        
        for audio_path in audio_files:
            filename = os.path.basename(audio_path)
            name_no_ext = os.path.splitext(filename)[0]
            trans_path = os.path.join(self.trans_dir, f"{name_no_ext}.txt")
            
            if os.path.exists(trans_path):
                with open(trans_path, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
                
                if transcription:
                    data.append({
                        "audio": audio_path,
                        "sentence": transcription,
                        "id": name_no_ext
                    })
        
        print(f"✅ Successfully matched {len(data)} pairs.")
        
        # Create HF Dataset
        ds = Dataset.from_list(data)
        
        # ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        
        return ds

if __name__ == "__main__":
    # Test stub
    AUDIO_PATH = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9"
    TRANS_PATH = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9_transcriptions"
    
    builder = EgyptDatasetBuilder(AUDIO_PATH, TRANS_PATH)
    dataset = builder.build()
    print(dataset)
    
    if len(dataset) > 0:
        import librosa
        sample = dataset[0]
        audio_path = sample["audio"]
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"✅ Sample match: {sample['id']}")
        print(f"📝 Text: {sample['sentence']}")
        print(f"🎵 Audio shape: {audio.shape} at {sr}Hz")
    else:
        print("❌ No matches found. Check your directory paths.")
