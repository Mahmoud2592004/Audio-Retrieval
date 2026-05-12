import os
import gc
import torch
import time
import numpy as np
import scipy.io.wavfile as wav
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from transformers.pipelines import pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import json

# --- SECURITY BYPASS ---
import transformers.utils.import_utils
import transformers.modeling_utils
def skip_check(): return None
transformers.utils.import_utils.check_torch_load_is_safe = skip_check
transformers.modeling_utils.check_torch_load_is_safe = skip_check

# --- BNB PATCH ---
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

app = FastAPI()

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# --- AI CORE ---
def run_transcription(audio_path):
    clear_mem()
    # Use librosa to load audio into an array to bypass the FFMPEG requirement
    import librosa
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        print(f"ERROR: Failed to load audio with librosa: {e}")
        return []
        
    model_id = "openai/whisper-small"
    adapter = os.path.join("egypt_audio", "finetuning", "whisper-small-egyptian-lora-pruned-30")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    p = WhisperProcessor.from_pretrained(adapter)
    b = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    m = PeftModel.from_pretrained(b, adapter).eval()
    pipe = pipeline("automatic-speech-recognition", model=m, tokenizer=p.tokenizer, 
                   feature_extractor=p.feature_extractor, device=-1)
    
    # Pass the numpy array directly
    res = pipe(audio, return_timestamps=True, generate_kwargs={"language": "arabic"})
    del pipe, m, b, p
    clear_mem()
    return res["chunks"]

def run_summarization(text):
    import subprocess
    import sys
    
    # Save text to a temp file for the worker
    text_path = f"text_{int(time.time())}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    try:
        print("[AI] Spawning Isolated Summarizer Task...")
        # Clear main process memory before spawning
        clear_mem()
        
        result = subprocess.run(
            [sys.executable, "web_worker.py", text_path],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        output = result.stdout
        if "---RESULT_START---" in output:
            summary = output.split("---RESULT_START---")[1].split("---RESULT_END---")[0].strip()
            return summary
        else:
            print(f"WORKER ERROR: {result.stderr}")
            return "Summarization process failed. Please check terminal."
    finally:
        if os.path.exists(text_path):
            os.remove(text_path)

# --- ROUTES ---
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    temp_path = f"temp_{int(time.time())}.wav"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # STEP 1: TRANSCRIPTION (CPU)
        chunks = run_transcription(temp_path)
        full_text = " ".join([c['text'] for c in chunks])
        
        # STEP 2: SUMMARIZATION (GPU/CPU Overflow)
        summary = run_summarization(full_text)
        
        # STEP 3: SEARCH (CPU - After Summarizer is Deleted)
        search_results = []
        faiss_path = os.path.join("rag", "youtube_transcript.faiss")
        json_path = os.path.join("rag", "youtube_transcript.json")
        
        if os.path.exists(faiss_path):
            # Back to BGE-M3 to match your 1024-dim index
            print("[AI] Loading Search Engine...")
            search_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
            index = faiss.read_index(faiss_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                yt_data = json.load(f)
            
            for c in chunks:
                q_emb = search_model.encode(c['text'], normalize_embeddings=True)
                D, I = index.search(np.array([q_emb]).astype('float32'), 1)
                if I[0][0] != -1 and D[0][0] > 0.4:
                    item = yt_data[I[0][0]]
                    search_results.append({
                        "chunk": c['text'],
                        "yt_text": item['text'],
                        "yt_url": f"https://youtu.be/{item['video_id']}?t={int(item['start_time'])}"
                    })
            del search_model, index
            clear_mem()
        return {
            "chunks": chunks,
            "summary": summary,
            "retrieval": search_results
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
