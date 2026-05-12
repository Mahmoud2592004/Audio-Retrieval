from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class QualityTierConfig:
    denoise_factor: float
    norm_method: str
    target_peak: float
    description: str

@dataclass
class PipelineConfig:
    # General
    target_sr: int = 16000
    
    # Tier Routing Definitions
    tiers: Dict[str, QualityTierConfig] = field(default_factory=lambda: {
        "EXCELLENT": QualityTierConfig(0.0, "peak", 0.9, "Pristine audio, no denoising needed."),
        "GOOD": QualityTierConfig(0.4, "peak", 0.9, "Clear audio, very light neural blending."),
        "FAIR": QualityTierConfig(0.75, "rms", 0.15, "Noisy, standard neural enhancement."),
        "POOR": QualityTierConfig(0.95, "rms", 0.12, "Very poor, extreme neural enhancement.")
    })
    
    # Segmentation
    min_segment_len: float = 0.5
    max_segment_len: float = 30.0
    
    # VAD
    vad_threshold: float = 0.5
