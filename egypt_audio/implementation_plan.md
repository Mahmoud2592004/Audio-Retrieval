# Adaptive Egyptian Arabic Audio Preprocessing Pipeline

Build a production-grade, **adaptive** audio preprocessing pipeline in `egypt_audio/` that dynamically routes each audio through different processing strategies based on its measured quality characteristics.

## User Review Required

> [!IMPORTANT]
> **Adaptive Routing**: The pipeline does NOT apply a fixed sequence to every file. Each audio gets scored first, then routed to the appropriate denoising strength, normalization level, and segmentation strategy based on its SNR, clipping, and speech presence.

> [!WARNING]  
> **4GB VRAM Constraint**: Silero VAD is CPU-friendly (~50MB). No GPU-heavy models in the preprocessing pipeline itself. DeepFilterNet is optional and will be gated behind a config flag.

## Proposed Changes

### Project Structure

```
egypt_audio/
├── __init__.py
├── config.py                  # All configurable parameters
├── pipeline.py                # Main AdaptivePipeline orchestrator
├── analysis/                  # Data exploration scripts
│   └── data_explorer.py       # Audio & Text distribution analyst
├── modules/
│   ├── __init__.py
│   ├── validation.py          # Input validation + header checks
│   ├── quality_scoring.py     # Energy-based SNR, clipping %, dynamic range
│   ├── normalization.py       # DC offset, adaptive loudness/peak
│   ├── denoising.py           # Spectral gating (adaptive strength)
│   ├── vad.py                 # Silero VAD wrapper
│   └── segmentation.py        # Pause-based semantic chunking
├── utils/
│   ├── __init__.py
│   └── audio_io.py            # Reliable loading/saving
└── preprocessing_notebook.ipynb  # Interactive validation & analysis notebook
```

---

### Core Design: Adaptive Routing & Energy-Based Scoring

1. **Energy-Based SNR (No VAD)**: Since clips are <5s, we will use a percentile-based energy approximation or spectral noise floor estimation to avoid VAD dependencies in the scoring pass.
2. **Quality Tiering**: Routing remains based on Measured Tier (EXCELLENT, GOOD, FAIR, POOR).

### Phase 1: Analysis & Data Exploration
[NEW] Initial step in the notebook to analyze:
- **Audio Duration**: Mean, Max, Min, Distribution (histograms).
- **Text Analysis**: Word count distribution, character counts, most frequent slang terms.
- **Corpus Integrity**: Matching WAVs to transcriptions.

### Phase 2: Core Infrastructure (Validation, Scoring, I/O)
- Implement `audio_io` for reliable loading.
- `validation` for production-grade header/corruption checks.
- `quality_scoring` with **Energy-Based SNR**.

### Phase 3: Processing Heavy-Lifters (Denoising, VAD, Segmentation)
- Adaptive `SpectralDenoizer`.
- Lightweight `NeuralVAD`.
- `IntelligentSegmenter` using pause boundaries.

### Phase 4: Integration & Batch Execution
- `AdaptivePipeline` class to tie everything together.
- Batch processing implementation with result logging.

### Phase 6: Live Interaction & Confidence Loop
- [NEW] **ASR Integration**: Wrapper for `faster-whisper` with confidence extraction.
- [NEW] **Mic Recording**: Notebook-based recording utility to test with live Egyptian slang.
- [NEW] **The Confidence Loop**: Logic to re-process audio if Whisper's `avg_logprob` is below threshold.
- [NEW] **Interactive Demo Notebook**: Separate notebook for live testing (Record → Process → Transcribe).

### Phase 7: Semantic Search & Retrieval (Future)
- Building the search engine from the generated transcriptions.

### Phase 8: Knowledge Distillation & Fine-Tuning
- **Directory**: `egypt_audio/finetuning/`
- **Goal**: Fine-tune Whisper-Small on Egyptian pseudo-labels (from Whisper-Large).
- **Optimization**: 4-bit QLoRA, Gradient Checkpointing, Frozen Encoder.
- **Evaluation**: WER comparison between Large Teacher and Small Student.

---

### Module Details

#### [config.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/config.py)
- Dataclass `PipelineConfig` with all tunable parameters
- Defaults for target sample rate (16000), normalization target (-23 LUFS), VAD threshold (0.5), denoise factors per tier, min/max segment duration
- User can override any parameter from the notebook

#### [pipeline.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/pipeline.py)
- `AdaptivePipeline` class
- `process_single(file_path)` → runs full pipeline, returns detailed result dict
- `process_batch(directory)` → runs on all WAVs, produces summary CSV
- Each step callable individually: `pipeline.validate(path)`, `pipeline.score(audio)`, etc.
- Returns per-file report: quality tier, processing decisions taken, before/after metrics

#### [modules/validation.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/modules/validation.py)
- File existence, decodability, zero-length, format detection
- Clipping detection (threshold-based)
- DC offset check
- Duration bounds check
- Returns `ValidationReport` dataclass

#### [modules/quality_scoring.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/modules/quality_scoring.py)
- **SNR Estimation**: Spectral noise floor method (estimating noise from low-energy frequency bins) or energy-ratio approximation.
- Clipping percentage
- RMS energy / peak amplitude
- Dynamic range (crest factor)
- Returns `QualityReport` with computed `QualityTier` enum

#### [modules/normalization.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/modules/normalization.py)
- DC offset removal (`audio -= mean`)
- Peak normalization (scale to target peak)
- LUFS-aware loudness normalization (using `pyloudnorm` if available, fallback to RMS-based)
- Adaptive: EXCELLENT tier gets light touch, POOR tier gets heavier normalization

#### [modules/denoising.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/modules/denoising.py)
- Spectral gating via STFT domain
- Noise reduction factor dynamically set by quality tier
- Spectral floor estimation from quietest frames
- Optional: `noisereduce` library integration as alternative backend

#### [modules/vad.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/modules/vad.py)
- Silero VAD wrapper (CPU-only, ~50MB)
- Auto-resamples to 16kHz for VAD
- Returns speech segments with timestamps
- Configurable threshold

#### [modules/segmentation.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/modules/segmentation.py)
- Groups VAD segments into semantic chunks
- Respects min/max duration constraints
- Merges short adjacent segments
- Splits on pause boundaries (not fixed windows)

#### [utils/audio_io.py](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/utils/audio_io.py)
- `load_audio(path, target_sr)` — load + resample
- `save_audio(audio, path, sr)` — save processed output
- `get_audio_info(path)` — metadata without full load

#### [preprocessing_notebook.ipynb](file:///d:/study/4th%20year%202nd%20term/NLP/Project%202%20-%20Audio/egypt_audio/preprocessing_notebook.ipynb)
- Interactive notebook with sections for each pipeline step
- Each step runs independently with visualization (waveform, spectrogram, before/after)
- Config cell at top to change all parameters
- Sample a few files from `cut_clips9` to demonstrate
- Shows quality routing decisions per file
- Summary statistics across batch

## Dependencies

Already installed: `numpy`, `librosa`, `scipy`, `soundfile`, `torch`, `torchaudio`, `tqdm`

May need: `pyloudnorm` (for LUFS normalization — optional, will fallback to RMS), `noisereduce` (optional backend)

## Verification Plan

### Automated Tests
- Process 5 sample audio files from `cut_clips9`
- Verify each module runs without error
- Check that quality tiers are assigned correctly
- Validate before/after SNR improvement

### Manual Verification
- Use the notebook to visually inspect waveforms and spectrograms before/after each step
- Compare transcription quality on a few files (original vs preprocessed)
