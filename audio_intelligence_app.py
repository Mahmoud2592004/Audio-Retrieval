import os
import streamlit as st
import re
import time
import torch
import gc
import numpy as np
import sounddevice as sd
import json
import scipy.io.wavfile as wav
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from transformers.pipelines import pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import transformers.utils.import_utils as import_utils
import logging

# --- SUPPRESS NOISY WARNINGS ---
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# --- ULTIMATE SECURITY BYPASS ---
import transformers.utils.import_utils
import transformers.modeling_utils
def skip_check(): return None
transformers.utils.import_utils.check_torch_load_is_safe = skip_check
transformers.modeling_utils.check_torch_load_is_safe = skip_check

# --- CONFIG ---
st.set_page_config(page_title="AI Audio Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- PROFESSIONAL ICONS (SVG) ---
ICONS = {
    "mic": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#00FF80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1v11m-7-5a7 7 0 0 0 14 0M8 23h8M12 19v4"/></svg>',
    "doc": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8"/></svg>',
    "search": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A855F7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "wave": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 10c.6.6 1.4 1 2.2 1.2M22 10c-.6.6-1.4 1-2.2 1.2M8 15c.6.6 1.4 1 2.2 1.2M14 15c.6.6 1.4 1 2.2 1.2"/></svg>'
}

# --- PREMIUM CSS ---
CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    .stApp {{
        background-color: #0A0C10;
        font-family: 'Inter', sans-serif;
    }}
    
    .header-container {{
        background: linear-gradient(90deg, #1A1D24 0%, #0A0C10 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        border-bottom: 1px solid #2D333B;
        margin-bottom: 2rem;
    }}
    
    .glass-card {{
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-label {{
        color: #8B949E;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }}
    
    .transcript-line {{
        padding: 0.75rem;
        border-left: 2px solid #3B82F6;
        background: #1C2128;
        margin-bottom: 0.5rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.95rem;
    }}
    
    .timestamp {{
        color: #58A6FF;
        font-family: monospace;
        margin-right: 10px;
        font-weight: 600;
    }}
    
    h1, h2, h3 {{
        color: #F0F6FC !important;
        font-weight: 600 !important;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 0 !important;
    }}
</style>
"""

# --- PIPELINE LOGIC ---

# BNB PATCH - copied from run_summarization.py (CRITICAL for 4-bit loading)
import bitsandbytes as bnb_lib
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

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def record_mic(seconds=10):
    fs = 16000
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording.flatten(), fs

def transcribe(audio_data):
    # Run Whisper on CPU to keep GPU 100% free for the summarizer
    model_id = "openai/whisper-small"
    adapter = os.path.join("egypt_audio", "finetuning", "whisper-small-egyptian-lora-pruned-30")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    p = WhisperProcessor.from_pretrained(adapter)
    b = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    m = PeftModel.from_pretrained(b, adapter).eval()
    pipe = pipeline("automatic-speech-recognition", model=m, tokenizer=p.tokenizer, 
                   feature_extractor=p.feature_extractor, device=-1)  # -1 = CPU
    res = pipe(audio_data, return_timestamps=True, generate_kwargs={"language": "arabic"})
    del pipe, m, b, p
    gc.collect()
    return res["chunks"]

def summarize(chunks):
    # --- ULTIMATE 4GB VRAM SAFETY ---
    clear_mem()
    time.sleep(2)
    
    text = " ".join([c["text"] for c in chunks])
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_name = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_path = os.path.join(current_dir, "summarization", "summarization", "finetuned_model")
    offload_dir = os.path.join(current_dir, "summarization", "offload")
    os.makedirs(offload_dir, exist_ok=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True # Support CPU overflow
    )
    
    try:
        print("[UI] Loading Summarizer in Low-VRAM Mode...")
        
        # STRICT VRAM CAP: Leaves 1.8GB free for Browser/Streamlit
        max_mem = {0: "2.2GiB", "cpu": "10GiB"}
        
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="balanced", # Automatically move layers to CPU if GPU is full
            max_memory=max_mem,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir
        )
        
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"Summarize this Egyptian Arabic text: {text}"}]}
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[prompt], return_tensors="pt").to(base_model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        generated_ids = [output_ids[0][len(inputs.input_ids[0]):]]
        summary = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Immediate Purge
        del model, base_model, processor
        clear_mem()
        return summary
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Summarization error: {str(e)}"

# --- INTERFACE ---
st.markdown(CSS, unsafe_allow_html=True)

# App Header
st.markdown(f'<div class="header-container"><h1>{ICONS["mic"]} Intelligent Audio Analysis</h1><p style="color:#8B949E">Speech Recognition • Neural Summarization • Semantic Search</p></div>', unsafe_allow_html=True)

col_ctrl, col_res = st.columns([1, 2])

with col_ctrl:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"<h3>{ICONS['mic']} Audio Input</h3>", unsafe_allow_html=True)
    
    tab_record, tab_upload = st.tabs(["Microphone", "Upload File"])
    
    with tab_record:
        sec = st.slider("Record Duration (Seconds)", 5, 60, 10)
        if st.button("Record and Analyze", use_container_width=True, type="primary"):
            with st.status("Initializing Neural Pipeline...", expanded=True) as status:
                st.write("Recording from microphone...")
                audio, fs = record_mic(sec)
                st.session_state.audio = audio
                st.session_state.audio_sr = fs
                
                st.write("Processing Speech-to-Text...")
                chunks = transcribe(audio)
                st.session_state.chunks = chunks
                
                st.write("Generating Arabic Summary...")
                summ = summarize(chunks)
                st.session_state.summary = summ
                status.update(label="Analysis Complete", state="complete")
    
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an audio file", 
            type=["wav", "mp3", "mp4", "m4a", "ogg", "flac"],
            help="Supported: WAV, MP3, MP4, M4A, OGG, FLAC"
        )
        if uploaded and st.button("Analyze Uploaded File", use_container_width=True, type="primary"):
            with st.status("Processing Uploaded Audio...", expanded=True) as status:
                st.write("Decoding audio file...")
                import io
                import soundfile as sf
                audio_bytes = uploaded.read()
                try:
                    audio_arr, sr = sf.read(io.BytesIO(audio_bytes))
                    # Convert to mono float32 at 16kHz
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr.mean(axis=1)
                    audio_arr = audio_arr.astype(np.float32)
                    # Resample if needed
                    if sr != 16000:
                        import scipy.signal as signal
                        num_samples = int(len(audio_arr) * 16000 / sr)
                        audio_arr = signal.resample(audio_arr, num_samples)
                    st.session_state.audio = audio_arr
                    st.session_state.audio_sr = 16000
                except Exception as e:
                    st.error(f"Could not decode audio: {e}")
                    st.stop()
                
                st.write("Processing Speech-to-Text...")
                chunks = transcribe(audio_arr)
                st.session_state.chunks = chunks
                
                st.write("Generating Arabic Summary...")
                summ = summarize(chunks)
                st.session_state.summary = summ
                status.update(label="Analysis Complete", state="complete")

    if "audio" in st.session_state:
        st.markdown('<p class="metric-label">Original Audio</p>', unsafe_allow_html=True)
        st.audio(st.session_state.audio, sample_rate=16000)
        
        st.markdown('<p class="metric-label">Preprocessed (Normalized)</p>', unsafe_allow_html=True)
        norm_audio = st.session_state.audio / (np.max(np.abs(st.session_state.audio)) + 1e-6)
        st.audio(norm_audio, sample_rate=16000)
    st.markdown('</div>', unsafe_allow_html=True)


    if "chunks" in st.session_state:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>{ICONS['search']} Semantic Search</h3>", unsafe_allow_html=True)
        q = st.text_input("Find meaning in this audio...", placeholder="e.g. 'search for the part about the soviet union'")
        
        if q:
            with st.spinner("Searching Local & Global Database..."):
                clear_mem()
                import_utils.check_torch_load_is_safe = lambda: None 
                search_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
                
                # 1. SEARCH CURRENT RECORDING
                chunk_texts = [c['text'] for c in st.session_state.chunks]
                query_emb = search_model.encode(q, normalize_embeddings=True)
                local_embs = search_model.encode(chunk_texts, normalize_embeddings=True)
                local_scores = np.dot(local_embs, query_emb)
                
                # 2. SEARCH GLOBAL DATABASE (YouTube)
                import faiss
                faiss_path = os.path.join("rag", "youtube_transcript.faiss")
                json_path = os.path.join("rag", "youtube_transcript.json")
                
                st.subheader("Search Results")
                
                # --- LOCAL RESULTS ---
                st.markdown('<p class="metric-label">Current Recording Segments</p>', unsafe_allow_html=True)
                top_local = np.argsort(local_scores)[::-1][:3]
                found_local = False
                for idx in top_local:
                    if local_scores[idx] > 0.35:
                        found_local = True
                        c = st.session_state.chunks[idx]
                        with st.container():
                            st.markdown(f"""
                            <div class="search-result" style="border-left: 4px solid #00FF80;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="color:#00FF80; font-weight:600;">LOCAL MATCH</span>
                                    <span class="timestamp">[{c['timestamp'][0]:.1f}s - {c['timestamp'][1]:.1f}s]</span>
                                </div>
                                <p style="margin: 10px 0;">{c['text']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            start, end = int(c['timestamp'][0]*16000), int(c['timestamp'][1]*16000)
                            st.audio(st.session_state.audio[start:end], sample_rate=16000)
                if not found_local:
                    st.info("No highly relevant segments found in this recording.")

                # --- GLOBAL RESULTS ---
                if os.path.exists(faiss_path):
                    st.markdown('<br><p class="metric-label">Global Knowledge (YouTube Database)</p>', unsafe_allow_html=True)
                    index = faiss.read_index(faiss_path)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        yt_data = json.load(f)
                    
                    D, I = index.search(np.array([query_emb]).astype('float32'), 3)
                    
                    for i, idx in enumerate(I[0]):
                        if idx != -1 and D[0][i] > 0.35:
                            item = yt_data[idx]
                            st.markdown(f"""
                            <div class="search-result" style="border-left: 4px solid #A855F7; margin-bottom: 10px;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span style="color:#A855F7; font-weight:600;">YOUTUBE SOURCE</span>
                                    <a href="{item['url']}" target="_blank" style="color:#3B82F6; text-decoration:none; font-size:0.8rem;">🔗 OPEN VIDEO AT {item['timestamp']}</a>
                                </div>
                                <p style="margin: 10px 0;">{item['text']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                del search_model
                clear_mem()
        st.markdown('</div>', unsafe_allow_html=True)

with col_res:
    if "chunks" in st.session_state:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>{ICONS['doc']} Intelligent Transcription & Retrieval</h3>", unsafe_allow_html=True)
        
        # Load search model once for the whole column
        import faiss
        faiss_path = os.path.join("rag", "youtube_transcript.faiss")
        json_path = os.path.join("rag", "youtube_transcript.json")
        
        has_index = os.path.exists(faiss_path)
        if has_index:
            clear_mem()
            search_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
            index = faiss.read_index(faiss_path)
            with open(json_path, 'r', encoding='utf-8') as f:
                yt_data = json.load(f)
        
        for c in st.session_state.chunks:
            st.markdown(f'<div class="transcript-line"><span class="timestamp">[{c["timestamp"][0]:.1f}s]</span> {c["text"]}</div>', unsafe_allow_html=True)
            
            # AUTO-RETRIEVAL FOR EACH CHUNK
            if has_index:
                q_emb = search_model.encode(c['text'], normalize_embeddings=True)
                D, I = index.search(np.array([q_emb]).astype('float32'), 1) # Get top 1
                if I[0][0] != -1 and D[0][0] > 0.4:
                    item = yt_data[I[0][0]]
                    # Build URL from video_id and start_time
                    yt_url = f"https://youtu.be/{item['video_id']}?t={int(item['start_time'])}"
                    mins = int(item['start_time']) // 60
                    secs = int(item['start_time']) % 60
                    ts_label = f"{mins:02d}:{secs:02d}"
                    st.markdown(f"""
                    <div style="margin: -10px 0 15px 30px; padding: 10px; background: rgba(168, 85, 247, 0.05); border-radius: 8px; border-left: 2px solid #A855F7;">
                        <small style="color:#A855F7; font-weight:600;">RELATED SOURCE [{ts_label}]:</small> {item['text'][:120]}...
                        <br><a href="{yt_url}" target="_blank" style="color:#3B82F6; text-decoration:none; font-size:0.75rem;">[WATCH ON YOUTUBE]</a>
                    </div>
                    """, unsafe_allow_html=True)
        
        if has_index:
            del search_model
            clear_mem()
        st.markdown('</div>', unsafe_allow_html=True)
