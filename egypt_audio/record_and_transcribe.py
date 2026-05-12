import os
import sys
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import warnings

# Suppress warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

# Add current directory to path for local imports
sys.path.append(os.path.abspath('.'))

try:
    from pipeline import AdaptivePipeline
    from config import PipelineConfig
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Ensure you are running this from the 'egypt_audio' directory and all dependencies are installed.")
    sys.exit(1)

def format_rtl(text):
    """
    Handles RTL text direction for terminal printing.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except ImportError:
        # Fallback if libraries are missing
        return text

def record_audio(duration=5, fs=16000):
    """
    Records audio from the default microphone.
    """
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        # Simple progress bar
        for i in range(duration, 0, -1):
            print(f"  [{i}s remaining]", end="\r")
            time.sleep(1)
        sd.wait()
        print("✅ Recording complete.       ")
        return recording.flatten(), fs
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None, fs

def main():
    # 1. Paths & Config
    base_model_name = "openai/whisper-small"
    
    # Resolve path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # adapter_path = os.path.join(script_dir, "finetuning", "whisper-small-egyptian-lora-final")
    adapter_path = os.path.join(script_dir, "finetuning", "whisper-small-egyptian-lora-pruned-30")

    output_file = "transcription_output.txt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Initializing Environment (Device: {device})...")

    # 2. Load Model & Processor
    try:
        print(f"📦 Loading Fine-tuned Model from {adapter_path}...")
        # Load processor from adapter path to ensure tokenizer consistency
        processor = WhisperProcessor.from_pretrained(adapter_path)
        
        # Load base model
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Tip: Ensure you have 'peft' and 'transformers' installed.")
        return

    # 3. Initialize Preprocessing Pipeline
    pipeline = AdaptivePipeline()

    # 4. Record
    audio, sr = record_audio(duration=10)
    if audio is None:
        return

    # Save temporary file for the pipeline (which expects a path)
    temp_wav = "temp_live_audio.wav"
    sf.write(temp_wav, audio, sr)

    # 5. Preprocess
    print("🛠 Applying Adaptive Preprocessing...")
    result = pipeline.process_file(temp_wav)
    
    if result["status"] == "success":
        processed_audio = result["processed_audio"]
        tier = result["tier"]
        print(f"✨ Preprocessing Done (Quality Tier: {tier})")
    else:
        print("⚠️ Preprocessing failed, using raw audio.")
        processed_audio = audio

    # 6. Transcribe
    print("📝 Transcribing...")
    input_features = processor(processed_audio, sampling_rate=sr, return_tensors="pt").input_features.to(device).to(model.dtype)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language="arabic", task="transcribe")
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # 7. Output Handling
    rtl_text = format_rtl(transcription)
    
    print("\n" + "="*50)
    print("📜 TRANSCRIPTION (Terminal):")
    print("-" * 50)
    print(rtl_text)
    print("="*50)

    # Save to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            # We save the logical text to the file, as most modern editors handle RTL correctly.
            # However, we can add an RTL mark for better compatibility.
            f.write("\ufeff") # BOM for UTF-8
            f.write(f"Audio Source: {temp_wav}\n")
            f.write(f"Quality Tier: {result.get('tier', 'N/A')}\n")
            f.write("-" * 20 + "\n")
            f.write(transcription)
        print(f"\n💾 Transcription saved to: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

if __name__ == "__main__":
    main()
