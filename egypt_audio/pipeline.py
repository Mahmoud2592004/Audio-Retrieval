import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from utils.audio_io import AudioIO
from modules.validation import AudioValidator
from modules.quality_scoring import AudioScorer
from modules.normalization import AudioNormalizer
from modules.denoising import SpectralDenoizer
from modules.vad import NeuralVAD
from modules.segmentation import IntelligentSegmenter
from config import PipelineConfig

class AdaptivePipeline:
    """
    Main orchestrator for the Egyptian Audio Preprocessing Pipeline.
    Implements adaptive routing: scores audio first, then picks parameters.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize modules
        self.validator = AudioValidator()
        self.scorer = AudioScorer()
        self.normalizer = AudioNormalizer()
        self.denoizer = SpectralDenoizer()
        self.vad = NeuralVAD()
        self.segmenter = IntelligentSegmenter(
            min_duration=self.config.min_segment_len,
            max_duration=self.config.max_segment_len
        )
        
    def process_file(self, input_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a single audio file end-to-end.
        """
        # 1. Validation
        is_valid, val_report = self.validator.validate(input_path)
        if not is_valid:
            return {"status": "failed", "report": val_report}
            
        # 2. Loading
        audio, sr = AudioIO.load(input_path, target_sr=self.config.target_sr)
        
        # 3. Scoring (Pre-processing)
        quality_report = self.scorer.get_quality_report(audio)
        tier_name = quality_report["quality_tier"]
        tier_cfg = self.config.tiers[tier_name]
        
        # 4. Adaptive Denoising
        processed_audio = self.denoizer.denoise(
            audio, sr, reduction_factor=tier_cfg.denoise_factor
        )
        
        # 5. Adaptive Normalization
        processed_audio = self.normalizer.process(
            processed_audio, method=tier_cfg.norm_method
        )
        
        # 6. VAD & Segmentation
        speech_segments = self.vad.get_speech_segments(processed_audio, sr)
        chunks = self.segmenter.chunk_audio(processed_audio, sr, speech_segments)
        
        # 7. Final Scoring (Post-processing)
        post_quality = self.scorer.get_quality_report(processed_audio)
        
        # 8. Saving (Optional)
        output_path = None
        if output_dir:
            file_name = os.path.basename(input_path)
            output_path = os.path.join(output_dir, f"prepped_{file_name}")
            AudioIO.save(processed_audio, output_path, sr)
            
        return {
            "status": "success",
            "file_name": os.path.basename(input_path),
            "tier": tier_name,
            "pre_snr": quality_report["snr_db"],
            "post_snr": post_quality["snr_db"],
            "segments_count": len(chunks),
            "output_path": output_path,
            "processed_audio": processed_audio, # Added for interactive use
            "full_report": {
                "validation": val_report,
                "pre_quality": quality_report,
                "post_quality": post_quality,
                "applied_config": tier_cfg.__dict__
            }
        }

    def process_folder(self, input_dir: str, output_dir: str) -> pd.DataFrame:
        """
        Batch processes an entire folder and returns a summary DataFrame.
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
        print(f"Batch processing {len(files)} files...")
        
        for f in tqdm(files):
            input_path = os.path.join(input_dir, f)
            res = self.process_file(input_path, output_dir)
            
            # Flatten result for DataFrame
            flat_res = {
                "file": f,
                "status": res["status"],
                "tier": res.get("tier"),
                "pre_snr": res.get("pre_snr"),
                "post_snr": res.get("post_snr"),
                "seg_count": res.get("segments_count")
            }
            results.append(flat_res)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test stub
    # pipeline = AdaptivePipeline()
    # print("Pipeline initialized.")
    pass
