import os
import sys
import torch
import gc
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import re
import warnings
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, pipeline
from peft import PeftModel

# Suppress warnings
warnings.filterwarnings("ignore")

# Monkey-patch to fix the accelerate compatibility bug for Qwen3
import bitsandbytes as bnb
from bitsandbytes.nn import Int8Params, Params4bit

def patch_bnb():
    orig_int8_new = Int8Params.__new__
    def new_int8_new(cls, data=None, *args, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return orig_int8_new(cls, data, *args, **kwargs)
    Int8Params.__new__ = new_int8_new
    
    orig_4bit_new = Params4bit.__new__
    def new_4bit_new(cls, data=None, *args, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return orig_4bit_new(cls, data, *args, **kwargs)
    Params4bit.__new__ = new_4bit_new

patch_bnb()

def get_gpu_memory():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated(0) / (1024**3):.2f}GB"
    return "N/A"

def record_audio(duration=30, fs=16000):
    print(f"\n🎤 [1/3] Recording for {duration} seconds... Speak now!")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        for i in range(duration, 0, -1):
            print(f"  [{i}s remaining]", end="\r")
            time.sleep(1)
        sd.wait()
        print("✅ Recording complete.       ")
        return recording.flatten(), fs
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        return None, fs

def transcribe_audio(audio, sr):
    print(f"\n📝 [2/3] Loading Whisper... (Current VRAM: {get_gpu_memory()})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "openai/whisper-small"
    adapter_path = os.path.join("egypt_audio", "finetuning", "whisper-small-egyptian-lora-pruned-30")
    
    try:
        processor = WhisperProcessor.from_pretrained(adapter_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(device)
        model.eval()

        print("📝 Transcribing with pipeline timestamps...")
        # Using the pipeline API is significantly more robust for timestamps
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if device == "cuda" else -1,
            dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        result = asr_pipe(audio, return_timestamps=True, generate_kwargs={"language": "arabic", "task": "transcribe"})
        
        # Format chunks with timestamps
        formatted_transcription = ""
        if isinstance(result["chunks"], list):
            for chunk in result["chunks"]:
                start, end = chunk["timestamp"]
                text = chunk["text"].strip()
                if text:
                    formatted_transcription += f"[{start:05.2f} -> {end:05.2f}] {text}\n"
        
        transcription = formatted_transcription if formatted_transcription else result["text"]
        
        # CLEANUP technical residues
        transcription = re.sub(r"<\|.*?\|>", "", transcription).strip()
        
        # AGGRESSIVE DISPATCH
        print("🧹 Dispatching Whisper...")
        del model
        del base_model
        del processor
        del asr_pipe
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"✅ Whisper Dispatched. (Current VRAM: {get_gpu_memory()})")
            
        return transcription
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None

def summarize_text(text):
    if not text:
        return None
    
    print(f"\n✨ [3/3] Loading Qwen3... (Current VRAM: {get_gpu_memory()})")
    base_model_name = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_path = os.path.join("summarization", "summarization", "finetuned_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        # Small delay to allow OS to stabilize after Whisper dispatch
        time.sleep(2) 
        
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        # Forced GPU loading to avoid disk offload errors on 4GB cards
        # Use the exact same settings that worked in run_summarization.py
        offload_dir = os.path.join("summarization", "offload")
        os.makedirs(offload_dir, exist_ok=True)
        
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0}, 
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        print("✨ Generating Summary...")
        
        # CLEANUP: Remove timestamps before summarizing so the model doesn't get confused
        clean_text = re.sub(r"\[\d+\.\d+ -> \d+\.\d+\]", "", text).strip()
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"Summarize this Egyptian Arabic text: {clean_text}"}]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], return_tensors="pt").to(base_model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        summary = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # AGGRESSIVE DISPATCH
        print("🧹 Dispatching Qwen3...")
        del model
        del base_model
        del processor
        del inputs
        del output_ids
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"✅ Qwen3 Dispatched. (Current VRAM: {get_gpu_memory()})")

        return summary
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
        return None

def main():
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"🎬 Starting Full Pipeline: {start_time}")
    print(f"📊 Initial VRAM: {get_gpu_memory()}")
    
    # 1. Record
    audio, sr = record_audio(duration=20)
    if audio is None: return

    # 2. Transcribe
    transcription = transcribe_audio(audio, sr)
    if not transcription:
        print("❌ Pipeline stopped: Transcription failed.")
        return

    print(f"\n📜 Transcript: {transcription}")

    # 3. Summarize
    summary = summarize_text(transcription)
    if not summary:
        print("❌ Pipeline stopped: Summarization failed.")
        return

    print(f"\n✨ Summary: {summary}")

    # 4. Save Final Log
    output_log = "pipeline_log.txt"
    with open(output_log, "w", encoding="utf-8") as f:
        f.write("\ufeff") # BOM for UTF-8
        f.write(f"=== AUDIO TO SUMMARY PIPELINE LOG ===\n")
        f.write(f"Timestamp: {start_time}\n")
        f.write("-" * 40 + "\n")
        f.write(f"1. ORIGINAL TRANSCRIPTION:\n")
        f.write(f"{transcription}\n\n")
        f.write(f"2. EGYPTIAN ARABIC SUMMARY:\n")
        f.write(f"{summary}\n")
        f.write("-" * 40 + "\n")
        f.write(f"End of Pipeline.\n")

    print(f"\n✅ FULL PIPELINE COMPLETE!")
    print(f"📂 Results saved to: {os.path.abspath(output_log)}")

if __name__ == "__main__":
    main()
