import numpy as np
import os
from typing import Dict, Any, Tuple
from utils.audio_io import AudioIO

class AudioValidator:
    """
    Validates audio files for production pipelines.
    Checks for decoding errors, clipping, zero duration, and DC offset.
    """
    
    def __init__(self, max_clipping_ratio: float = 0.001):
        self.max_clipping_ratio = max_clipping_ratio
        
    def validate(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Runs comprehensive validation checks.
        Returns: (is_valid, report)
        """
        report = {
            "file_name": os.path.basename(file_path),
            "status": "pending",
            "checks": {},
            "errors": []
        }
        
        # 1. Existence check
        if not os.path.exists(file_path):
            report["status"] = "failed"
            report["errors"].append("File not found.")
            return False, report
            
        try:
            # 2. Metadata check
            info = AudioIO.get_info(file_path)
            report["checks"]["metadata"] = info
            
            if info["duration"] <= 0:
                report["status"] = "failed"
                report["errors"].append("Zero or negative duration.")
                return False, report
                
            # 3. Load audio for internal checks
            audio, sr = AudioIO.load(file_path, target_sr=None)
            
            # 4. Clipping Detection
            # Audio is usually normalized to [-1, 1]
            peak = np.max(np.abs(audio))
            clipped_samples = np.sum(np.abs(audio) >= 0.999)
            clipping_ratio = clipped_samples / len(audio)
            
            report["checks"]["peak_amplitude"] = round(float(peak), 4)
            report["checks"]["clipping_ratio"] = round(float(clipping_ratio), 6)
            
            if clipping_ratio > self.max_clipping_ratio:
                report["errors"].append(f"Excessive clipping detected: {clipping_ratio:.2%}")
                
            # 5. DC Offset Check
            dc_offset = np.mean(audio)
            report["checks"]["dc_offset"] = round(float(dc_offset), 6)
            if abs(dc_offset) > 0.05:
                report["errors"].append(f"High DC offset detected: {dc_offset:.4f}")
                
            # 6. Silent Audio Check
            rms = np.sqrt(np.mean(audio**2))
            report["checks"]["rms"] = round(float(rms), 6)
            if rms < 1e-4:
                report["status"] = "failed"
                report["errors"].append("Audio appears to be silent.")
                return False, report
                
            report["status"] = "passed" if not report["errors"] else "passed_with_warnings"
            return True, report
            
        except Exception as e:
            report["status"] = "failed"
            report["errors"].append(f"Decoding/Process error: {str(e)}")
            return False, report

if __name__ == "__main__":
    # Test
    validator = AudioValidator()
    TEST_PATH = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9\ARA NORM 00080.wav"
    if os.path.exists(TEST_PATH):
        valid, report = validator.validate(TEST_PATH)
        print(f"Validation Result: {valid}")
        print(f"Report: {report}")
