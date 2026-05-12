import os
import re
import gc
import torch
import psutil
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, logging as tf_logging
from peft import PeftModel
import logging
import bitsandbytes as bnb
from bitsandbytes.nn import Int8Params, Params4bit
import time

# Monkey-patch to fix the accelerate compatibility bug
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

def main():
    # Clear GPU cache and GC
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Paths (Using absolute paths to be safe)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_name = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_path = os.path.join(current_dir, "summarization", "finetuned_model")
    input_file = os.path.join(current_dir, "statements_to_summarize.txt")
    output_file = os.path.join(current_dir, "summarized_outputs_lora.txt")

    # Status Report
    ram = psutil.virtual_memory()
    print(f"🖥 System RAM Status: {ram.available / (1024**3):.2f}GB free")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"📊 GPU Status: {allocated:.2f}GB allocated")

    # Force sequential loading to save RAM spikes
    print(f"🚀 Initializing Summarization Test (Device: cuda)...")
    
    try:
        print("🔍 [DEBUG] Starting loading sequence...")
        print(f"🖥 RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB free")
        
        # 1. Config for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # 2. Load Processor
        print(f"📦 Loading Processor from {base_model_name}...")
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        print("✅ Processor Loaded.")

        # 3. Load Base Model
        print(f"🛠 Loading Base Model: {base_model_name}...")
        # Create an offload folder to act as "Virtual RAM"
        offload_dir = os.path.join(os.path.dirname(adapter_path), "offload")
        os.makedirs(offload_dir, exist_ok=True)
        
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0}, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir  # Use disk if RAM runs out
        )
        print("✅ Base Model Loaded into VRAM.")
        
        # 4. Load Adapter
        print(f"🧠 Loading LoRA adapter from: {adapter_path}")
        if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
             print(f"⚠️ WARNING: adapter_config.json not found at {adapter_path}")
             
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        print("✅ Full Model + Adapter Ready!")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Load LoRA Adapter
    try:
        print(f"📦 Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        print("✅ LoRA adapter merged!")
    except Exception as e:
        print(f"❌ Failed to load adapter: {e}")
        return

    # 4. Read Input File
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    statements = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+-\s*(.*)$", line)
        if match:
            statements.append(match.group(1))
        else:
            statements.append(line)

    print(f"📝 Found {len(statements)} statements. Summarizing...")

    results = []
    for i, text in enumerate(statements):
        print(f"   [{i+1}/{len(statements)}] Processing...")
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"Summarize this Egyptian Arabic text: {text}"}]}
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
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        summary = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        results.append({
            "original": text,
            "summary": summary
        })

    # 5. Save
    print(f"💾 Saving results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for i, res in enumerate(results):
            f.write(f"{i+1}-\n")
            f.write(f"Original: {res['original']}\n")
            f.write(f"Summary: {res['summary']}\n")
            f.write("-" * 30 + "\n")

    print("✅ Done!")

if __name__ == "__main__":
    main()
