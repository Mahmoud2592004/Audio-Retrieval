import os
import glob
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Any

class DataExplorer:
    """
    Analyzes the distribution of audio durations and transcription text.
    Handles the cut_clips9 and cut_clips9_transcriptions folders.
    """
    
    def __init__(self, audio_dir: str, transcript_dir: str):
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        
        audio_pattern = os.path.join(audio_dir, "*.wav")
        transcript_pattern = os.path.join(transcript_dir, "*.txt")
        
        self.audio_files = glob.glob(audio_pattern)
        self.transcript_files = glob.glob(transcript_pattern)
        
        print(f"DEBUG: Searching for audio in: {audio_pattern}")
        print(f"DEBUG: Found {len(self.audio_files)} audio files.")
        print(f"DEBUG: Searching for transcripts in: {transcript_pattern}")
        print(f"DEBUG: Found {len(self.transcript_files)} transcript files.")
        
    def analyze_audio_durations(self) -> pd.DataFrame:
        """
        Calculates durations for all audio files.
        """
        durations = []
        filenames = []
        
        print(f"Analyzing {len(self.audio_files)} audio files...")
        for file in tqdm(self.audio_files):
            try:
                # get_duration is fast as it only reads the header
                duration = librosa.get_duration(path=file)
                durations.append(duration)
                filenames.append(os.path.basename(file))
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        return pd.DataFrame({
            "filename": filenames,
            "duration_sec": durations
        })
        
    def analyze_transcriptions(self) -> pd.DataFrame:
        """
        Calculates word and character counts for transcriptions.
        """
        word_counts = []
        char_counts = []
        filenames = []
        texts = []
        
        print(f"Analyzing {len(self.transcript_files)} transcriptions...")
        for file in tqdm(self.transcript_files):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                texts.append(text)
                word_counts.append(len(text.split()))
                char_counts.append(len(text))
                filenames.append(os.path.basename(file).replace(".txt", ".wav"))
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        return pd.DataFrame({
            "filename": filenames,
            "text": texts,
            "word_count": word_counts,
            "char_count": char_counts
        })
        
    def check_integrity(self, audio_df: pd.DataFrame, transcript_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Checks matching between audio and transcripts.
        """
        audio_filenames = set(audio_df['filename'])
        transcript_filenames = set(transcript_df['filename'])
        
        matched = audio_filenames.intersection(transcript_filenames)
        missing_audio = transcript_filenames - audio_filenames
        missing_transcript = audio_filenames - transcript_filenames
        
        return {
            "total_audio": len(audio_filenames),
            "total_transcripts": len(transcript_filenames),
            "matched_count": len(matched),
            "missing_audio_files": list(missing_audio),
            "missing_transcript_files": list(missing_transcript)
        }

if __name__ == "__main__":
    # Test paths
    AUDIO_DIR = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9"
    TRANS_DIR = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9_transcriptions"
    
    explorer = DataExplorer(AUDIO_DIR, TRANS_DIR)
    # This might take a while if many files, so we'd typically run this from the notebook
    print(f"Found {len(explorer.audio_files)} audio files and {len(explorer.transcript_files)} transcripts.")
