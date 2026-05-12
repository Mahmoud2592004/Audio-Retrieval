import os
import numpy as np
from faster_whisper import WhisperModel
from typing import Dict, Any, List

class WhisperASR:
    """
    Wrapper for faster-whisper with confidence scoring and Egyptian Slang optimization.
    """
    def __init__(self, model_size: str = "large-v3", device: str = "cuda", compute_type: str = "int8_float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            print(f"Loading Whisper {self.model_size}...")
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
            )
            
    def transcribe(self, audio: np.ndarray, initial_prompt: str = "Egyptian Arabic slang. عامية مصرية") -> Dict[str, Any]:
        """
        Transcribes audio and returns text + confidence metrics.
        """
        self._load_model()
        
        # Whisper expects float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        segments, info = self.model.transcribe(
            audio, 
            language="ar", 
            initial_prompt=initial_prompt,
            beam_size=5
        )
        
        full_text = []
        confidences = []
        
        for s in segments:
            full_text.append(s.text)
            # avg_logprob is a log probability. Higher (closer to 0) is better.
            # We convert it to a rough 0-1 confidence for easier understanding
            conf = np.exp(s.avg_logprob)
            confidences.append(conf)
            
        text = " ".join(full_text).strip()
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "text": text,
            "confidence": round(float(avg_confidence), 3),
            "language": info.language,
            "language_probability": info.language_probability
        }
