import os
import sys
import torch
import numpy as np
import librosa
from types import ModuleType

# --- SUPER SANDBOX FIX ---
# We block the specific modules that are causing the infinite crash loop
# This prevents SpeechBrain from even TRYING to look at k2 or flair.
broken_modules = [
    "k2", "_k2", "flair", 
    "speechbrain.integrations.k2_fsa", 
    "speechbrain.integrations.nlp",
    "speechbrain.pretrained"
]
for mod in broken_modules:
    if mod not in sys.modules:
        sys.modules[mod] = ModuleType(mod)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# -------------------------

# Import the inference class directly
from speechbrain.inference.enhancement import SpectralMaskEnhancement

class SpectralDenoizer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing SpeechBrain Enhancer on {self.device}...")
        
        # Bypass the standard loading if it's still crashing
        try:
            self.enhancer = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir="pretrained_models/metricgan-plus",
                run_opts={"device": self.device}
            )
            self.use_neural = True
            print("✅ SpeechBrain MetricGAN+ Sandboxed & Ready.")
        except Exception as e:
            print(f"⚠️ SpeechBrain Load Error: {e}")
            self.use_neural = False

    def denoise(self, audio: np.ndarray, sr: int, reduction_factor: float = 1.0) -> np.ndarray:
        """
        Denoises audio using SpeechBrain MetricGAN+.
        reduction_factor: 0.0 (no change) to 1.0 (full enhancement).
        """
        if not self.use_neural or reduction_factor <= 0:
            return audio
            
        try:
            # Resample to 16k if needed (model requirement)
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
            audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                enhanced = self.enhancer.enhance_batch(audio_tensor, lengths=torch.tensor([1.0]).to(self.device))
            
            enhanced_np = enhanced.cpu().squeeze(0).numpy()
            
            # Match lengths in case of resampling artifacts
            if len(enhanced_np) != len(audio_16k):
                min_len = min(len(enhanced_np), len(audio_16k))
                enhanced_np = enhanced_np[:min_len]
                audio_16k = audio_16k[:min_len]
            
            # --- BLENDING LOGIC ---
            # If reduction_factor < 1.0, we blend original and enhanced
            # This preserves naturalness for high-quality audio
            final_16k = (reduction_factor * enhanced_np) + ((1.0 - reduction_factor) * audio_16k)
            
            # Resample back to original sr
            return librosa.resample(final_16k, orig_sr=16000, target_sr=sr) if sr != 16000 else final_16k
            
        except Exception as e:
            print(f"⚠️ Denoising Error: {e}")
            return audio
