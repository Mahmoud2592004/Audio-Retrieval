import os
import re
import gc
import sys
import torch
import time
import psutil
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# --- SECURITY BYPASS (same as pipeline) ---
import transformers.utils.import_utils
import transformers.modeling_utils
def skip_check(): return None
transformers.utils.import_utils.check_torch_load_is_safe = skip_check
transformers.modeling_utils.check_torch_load_is_safe = skip_check

# --- BNB PATCH (from run_summarization.py) ---
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


def run_worker():
    print("STATUS: Worker started")

    if not os.path.exists("temp_transcript.txt"):
        print("ERROR: temp_transcript.txt not found")
        return

    with open("temp_transcript.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("ERROR: temp_transcript.txt is empty")
        return

    # Use ABSOLUTE paths (same as run_summarization.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    summarization_dir = os.path.join(current_dir, "summarization")
    base_model_name = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_path = os.path.join(summarization_dir, "summarization", "finetuned_model")
    offload_dir = os.path.join(summarization_dir, "offload")
    os.makedirs(offload_dir, exist_ok=True)

    # Memory status
    ram = psutil.virtual_memory()
    print(f"STATUS: RAM available: {ram.available / (1024**3):.2f}GB")
    if torch.cuda.is_available():
        print(f"STATUS: GPU allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f}GB")

    # --- EXACT BNB CONFIG from run_summarization.py ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        # STEP 1: Processor
        print("STATUS: Loading processor...")
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        print("STATUS: Processor loaded")

        # STEP 2: Base model — EXACT same as run_summarization.py
        print("STATUS: Loading base model...")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0},          # Force all to GPU 0 - NO CPU splitting
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir
        )
        print("STATUS: Base model loaded")

        # STEP 3: LoRA Adapter
        print("STATUS: Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        print("STATUS: Model ready")

        # STEP 4: Inference — EXACT same prompt format as run_summarization.py
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
            output_ids[0][len(inputs.input_ids[0]):]
        ]
        summary = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        print(f"---RESULT_START---{summary}---RESULT_END---")
        print("STATUS: Done")

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    run_worker()
