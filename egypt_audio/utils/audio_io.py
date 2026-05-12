import librosa
import soundfile as sf
import numpy as np
import os
from typing import Tuple, Optional, Any

class AudioIO:
    """
    Handles reliable loading, saving, and resampling of audio files.
    Optimized for production pipelines.
    """
    
    @staticmethod
    def load(file_path: str, target_sr: Optional[int] = 16000) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file and optionally resamples it.
        Returns: (audio_data, sample_rate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # load with librosa for automatic format handling
        # sr=None preserves original SR
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        return audio, sr

    @staticmethod
    def save(audio: np.ndarray, file_path: str, sr: int):
        """
        Saves audio data to a WAV file using soundfile.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        sf.write(file_path, audio, sr)

    @staticmethod
    def get_info(file_path: str) -> dict:
        """
        Gets metadata without loading the entire file into memory.
        """
        info = sf.info(file_path)
        return {
            "samplerate": info.samplerate,
            "channels": info.channels,
            "duration": info.duration,
            "format": info.format
        }

if __name__ == "__main__":
    # Quick test
    TEST_PATH = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9\ARA NORM 00080.wav"
    if os.path.exists(TEST_PATH):
        info = AudioIO.get_info(TEST_PATH)
        print(f"Info for {os.path.basename(TEST_PATH)}: {info}")
        
        audio, sr = AudioIO.load(TEST_PATH, target_sr=16000)
        print(f"Loaded audio with shape {audio.shape} at {sr}Hz")
