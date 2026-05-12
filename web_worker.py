import os
import gc
import torch
import time
import sys
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# --- SECURITY & BNB PATCH ---
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
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

def run_isolated_summary(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    base_model_name = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_path = os.path.join("summarization", "summarization", "finetuned_model")
    offload_dir = os.path.join("summarization", "offload_worker")
    os.makedirs(offload_dir, exist_ok=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    # EXTREME ISOLATION: 2GB VRAM cap
    max_mem = {0: "2.1GiB", "cpu": "3.5GiB"} 
    
    # FORCE DIRECT GPU LOADING to avoid "Meta Tensor" errors
    # Since we are in an isolated worker, we can afford the RAM usage here.
    try:
        print("---LOADING_START---")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0}, # Force to GPU 0
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, adapter_path).eval()
    except Exception as e:
        print(f"WORKER_FATAL_ERROR: {e}")
        sys.exit(1)
    
    messages = [{"role": "user", "content": [{"type": "text", "text": f"Summarize this Egyptian Arabic text into 2-3 sentences: {text}"}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    print(f"---PROMPT_DEBUG---\n{prompt}\n---")
    
    # Use direct string input to the processor
    inputs = processor(text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=200, 
            do_sample=True,      # FORCE START
            temperature=0.5,     # BE CREATIVE
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    generated_ids = [output_ids[0][len(inputs.input_ids[0]):]]
    summary = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Fallback if empty
    if not summary:
        summary = "AI generated an empty response. Please try recording a longer clip."
    
    print("---RESULT_START---")
    print(summary)
    print("---RESULT_END---")

if __name__ == "__main__":
    run_isolated_summary(sys.argv[1])
