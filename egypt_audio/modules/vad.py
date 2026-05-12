import torch
import numpy as np
import librosa
from typing import List, Dict

class NeuralVAD:
    """
    Wraps Silero VAD for robust speech activity detection on the CPU.
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self.use_fallback = False
        
        try:
            # Load model from torch hub
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            (self.get_speech_timestamps, _, _, _, _) = self.utils
            print("✅ Silero VAD loaded successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load Neural VAD due to environment error: {e}")
            print("⚙️ Falling back to Energy-Based VAD.")
            self.use_fallback = True
        
    def get_speech_segments(self, audio: np.ndarray, sr: int) -> List[Dict[str, float]]:
        """
        Detects speech segments. Uses Silero if available, else Energy-based fallback.
        """
        if self.use_fallback:
            return self._energy_vad(audio, sr)
            
        # 1. Silero VAD requires 16k sample rate
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            current_sr = 16000
        else:
            audio_16k = audio
            current_sr = sr
            
        # 2. Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_16k.astype(np.float32))
        
        # 3. Predict timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor, 
            self.model, 
            sampling_rate=current_sr,
            threshold=self.threshold
        )
        
        # 4. Convert to seconds
        segments = []
        for ts in speech_timestamps:
            segments.append({
                "start": round(ts['start'] / current_sr, 3),
                "end": round(ts['end'] / current_sr, 3)
            })
            
        return segments

    def _energy_vad(self, audio: np.ndarray, sr: int) -> List[Dict[str, float]]:
        """
        Simple Energy-based VAD as a fallback when Neural VAD fails.
        """
        # Split on silence based on peak energy
        # top_db: the threshold (in decibels) below the maximum to consider as silence
        intervals = librosa.effects.split(audio, top_db=30)
        
        segments = []
        for start_sample, end_sample in intervals:
            segments.append({
                "start": round(start_sample / sr, 3),
                "end": round(end_sample / sr, 3)
            })
        return segments

if __name__ == "__main__":
    # Test
    vad = NeuralVAD()
    print("Silero VAD loaded.")
