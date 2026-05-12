import numpy as np
import librosa
from typing import Dict, Any
from enum import Enum

class QualityTier(Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"

class AudioScorer:
    """
    Measures audio quality tiers without relying on VAD.
    Uses Energy-Based Approximation and Spectral Noise Floor.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        
    def calculate_snr_energy(self, audio: np.ndarray) -> float:
        """
        Estimates SNR using Energy-Ratio Approximation.
        Separates 'signal' and 'noise' energy based on amplitude percentiles.
        """
        if len(audio) == 0:
            return 0.0
            
        # 1. Calculate energy (squared amplitude)
        energy = audio**2
        
        # 2. Estimate noise floor (bottom 10% energy samples)
        # This assumes that even in a short clip, there is some silence or background noise.
        noise_floor = np.percentile(energy, 10)
        
        # 3. Estimate signal level (top 90% energy samples)
        signal_level = np.percentile(energy, 90)
        
        if noise_floor <= 0:
            # High quality or digital silence
            return 60.0
            
        snr = 10 * np.log10(signal_level / noise_floor)
        return float(snr)

    def calculate_snr_spectral(self, audio: np.ndarray) -> float:
        """
        Estimates SNR using Spectral Noise Floor method.
        Estimates noise from the STFT magnitude spectrogram.
        """
        # compute STFT
        S = np.abs(librosa.stft(audio))
        
        # Estimate noise level for each frequency bin
        # We use the minimum energy across time frames as the noise floor for that bin
        noise_floor_per_bin = np.percentile(S, 10, axis=1)
        
        # Estimate signal level per bin
        signal_level_per_bin = np.percentile(S, 90, axis=1)
        
        avg_noise = np.mean(noise_floor_per_bin)
        avg_signal = np.mean(signal_level_per_bin)
        
        if avg_noise <= 0:
            return 60.0
            
        snr = 20 * np.log10(avg_signal / avg_noise)
        return float(snr)

    def get_quality_report(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Generates a comprehensive quality report and assigns a tier.
        """
        snr_energy = self.calculate_snr_energy(audio)
        snr_spectral = self.calculate_snr_spectral(audio)
        
        # We blend or choose the more conservative one
        final_snr = (snr_energy + snr_spectral) / 2
        
        tier = self._assign_tier(final_snr)
        
        return {
            "snr_db": round(final_snr, 2),
            "snr_energy_db": round(snr_energy, 2),
            "snr_spectral_db": round(snr_spectral, 2),
            "quality_tier": tier.value,
            "peak_norm": float(np.max(np.abs(audio))),
            "dynamic_range": float(20 * np.log10(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio**2)) + 1e-9)))
        }
        
    def _assign_tier(self, snr: float) -> QualityTier:
        if snr >= 25: return QualityTier.EXCELLENT
        if snr >= 15: return QualityTier.GOOD
        if snr >= 8:  return QualityTier.FAIR
        return QualityTier.POOR

if __name__ == "__main__":
    # Test
    scorer = AudioScorer()
    dummy_clean = np.sin(np.linspace(0, 1000, 16000)) * 0.5
    dummy_noise = np.random.normal(0, 0.05, 16000)
    
    print("Clean sine wave:")
    print(scorer.get_quality_report(dummy_clean))
    
    print("\nNoisy signal:")
    print(scorer.get_quality_report(dummy_noise))
