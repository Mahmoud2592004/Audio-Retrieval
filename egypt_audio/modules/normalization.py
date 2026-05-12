import numpy as np
from typing import Optional

class AudioNormalizer:
    """
    Standardizes audio loudness and removes artifacts like DC offset.
    Supports Peak and RMS normalization.
    """
    
    def __init__(self, target_peak: float = 0.9, target_rms: float = 0.15):
        self.target_peak = target_peak
        self.target_rms = target_rms
        
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Centers the waveform around zero.
        """
        return audio - np.mean(audio)
        
    def normalize_peak(self, audio: np.ndarray, target: Optional[float] = None) -> np.ndarray:
        """
        Scales audio so the absolute maximum matches the target.
        """
        t = target if target is not None else self.target_peak
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        return (audio / peak) * t

    def normalize_rms(self, audio: np.ndarray, target: Optional[float] = None) -> np.ndarray:
        """
        Scales audio to a target Root Mean Square (RMS) energy.
        """
        t = target if target is not None else self.target_rms
        rms = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return audio
        scaled = audio * (t / (rms + 1e-9))
        
        # Soft-clip to ensure no digital distortion
        return np.clip(scaled, -1.0, 1.0)

    def process(self, audio: np.ndarray, method: str = "peak") -> np.ndarray:
        """
        Runs the standard normalization chain.
        """
        audio = self.remove_dc_offset(audio)
        if method == "peak":
            return self.normalize_peak(audio)
        elif method == "rms":
            return self.normalize_rms(audio)
        return audio

if __name__ == "__main__":
    # Test
    norm = AudioNormalizer()
    dummy = np.random.normal(0, 0.01, 16000)
    print(f"Max before: {np.max(np.abs(dummy))}")
    processed = norm.process(dummy, method="peak")
    print(f"Max after peak norm: {np.max(np.abs(processed))}")
