import json
import os

nb_path = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\egypt_audio\finetuning\finetune_whisper.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Fine-Tuning Whisper-Small on Egyptian Arabic (4GB VRAM Optimized)\n",
            "\n",
            "This notebook implements Knowledge Distillation:\n",
            "1. Teacher: Whisper-Large-v3\n",
            "2. Student: Whisper-Small\n",
            "\n",
            "Optimization: 4-bit QLoRA with JIT Audio Loading."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import torch\n",
            "import librosa\n",
            "import numpy as np\n",
            "from datasets import DatasetDict\n",
            "from transformers import (\n",
            "    WhisperProcessor, \n",
            "    WhisperForConditionalGeneration, \n",
            "    Seq2SeqTrainingArguments, \n",
            "    Seq2SeqTrainer,\n",
            "    BitsAndBytesConfig\n",
            ")\n",
            "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
            "\n",
            "# Add parent directory to path to import our custom modules\n",
            "sys.path.append(os.path.abspath('..'))\n",
            "from data_loader import EgyptDatasetBuilder"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Dataset Loading (Path Only)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "MODEL_NAME = \"openai/whisper-small\"\n",
            "AUDIO_DIR = r\"d:\\study\\4th year 2nd term\\NLP\\Project 2 - Audio\\cut_clips9\"\n",
            "TRANS_DIR = r\"d:\\study\\4th year 2nd term\\NLP\\Project 2 - Audio\\cut_clips9_transcriptions\"\n",
            "\n",
            "processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=\"arabic\", task=\"transcribe\")\n",
            "\n",
            "builder = EgyptDatasetBuilder(AUDIO_DIR, TRANS_DIR)\n",
            "dataset = builder.build()\n",
            "dataset = dataset.train_test_split(test_size=0.1, seed=42)\n",
            "\n",
            "print(f\"Dataset ready. Train samples: {len(dataset['train'])}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Model Initialization (4-bit QLoRA)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(f\"Loading {MODEL_NAME} in 4-bit...\")\n",
            "\n",
            "bnb_config = BitsAndBytesConfig(\n",
            "    load_in_4bit=True,\n",
            "    bnb_4bit_quant_type='nf4',\n",
            "    bnb_4bit_compute_dtype=torch.float16,\n",
            "    bnb_4bit_use_double_quant=True,\n",
            ")\n",
            "\n",
            "model = WhisperForConditionalGeneration.from_pretrained(\n",
            "    MODEL_NAME, \n",
            "    quantization_config=bnb_config, \n",
            "    device_map='auto'\n",
            ")\n",
            "\n",
            "model = prepare_model_for_kbit_training(model)\n",
            "\n",
            "config = LoraConfig(\n",
            "    r=32, \n",
            "    lora_alpha=64, \n",
            "    target_modules=['q_proj', 'v_proj'], \n",
            "    lora_dropout=0.05, \n",
            "    bias='none'\n",
            ")\n",
            "\n",
            "model = get_peft_model(model, config)\n",
            "model.print_trainable_parameters()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Training Configuration (JIT Data Collator)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from dataclasses import dataclass\n",
            "from typing import Any, Dict, List, Union\n",
            "\n",
            "@dataclass\n",
            "class DataCollatorSpeechSeq2SeqWithPadding:\n",
            "    processor: Any\n",
            "    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:\n",
            "        # Load audio and extract features JIT to save memory\n",
            "        input_features = []\n",
            "        label_features = []\n",
            "        \n",
            "        for feature in features:\n",
            "            audio_path = feature['audio']\n",
            "            # Load at 16kHz\n",
            "            audio_array, _ = librosa.load(audio_path, sr=16000)\n",
            "            \n",
            "            # Extract Whisper features\n",
            "            f = self.processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]\n",
            "            input_features.append({'input_features': f})\n",
            "            \n",
            "            # Tokenize labels\n",
            "            l = self.processor.tokenizer(feature['sentence']).input_ids\n",
            "            label_features.append({'input_ids': l})\n",
            "\n",
            "        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')\n",
            "        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')\n",
            "\n",
            "        # Mask padding in labels\n",
            "        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)\n",
            "        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():\n",
            "            labels = labels[:, 1:]\n",
            "\n",
            "        batch['labels'] = labels\n",
            "        return batch\n",
            "\n",
            "data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)\n",
            "\n",
            "training_args = Seq2SeqTrainingArguments(\n",
            "    output_dir='./whisper-small-egyptian-lora',\n",
            "    per_device_train_batch_size=1,\n",
            "    gradient_accumulation_steps=16,\n",
            "    learning_rate=1e-4,\n",
            "    warmup_steps=50,\n",
            "    max_steps=500,\n",
            "    gradient_checkpointing=True,\n",
            "    fp16=True,\n",
            "    eval_strategy='steps',\n",
            "    per_device_eval_batch_size=1,\n",
            "    predict_with_generate=True,\n",
            "    generation_max_length=225,\n",
            "    save_steps=100,\n",
            "    eval_steps=100,\n",
            "    logging_steps=10,\n",
            "    report_to=['tensorboard'],\n",
            "    load_best_model_at_end=True,\n",
            "    metric_for_best_model='wer',\n",
            "    greater_is_better=False,\n",
            "    push_to_hub=False,\n",
            "    remove_unused_columns=False,\n",
            ")\n",
            "\n",
            "trainer = Seq2SeqTrainer(\n",
            "    args=training_args,\n",
            "    model=model,\n",
            "    train_dataset=dataset['train'],\n",
            "    eval_dataset=dataset['test'],\n",
            "    data_collator=data_collator,\n",
            "    processing_class=processor.feature_extractor,\n",
            ")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Training and Visualization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%load_ext tensorboard\n",
            "%tensorboard --logdir ./whisper-small-egyptian-lora/runs\n",
            "\n",
            "print(\"Starting Training...\")\n",
            "trainer.train()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Comparison Suite (Base vs Fine-tuned vs Large)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sounddevice as sd\n",
            "from scipy.io.wavfile import write\n",
            "from IPython.display import Audio, display, HTML\n",
            "from transformers import pipeline\n",
            "\n",
            "def record_audio(filename='test_input.wav', duration=5):\n",
            "    print(f\"Recording for {duration} seconds...\")\n",
            "    fs = 16000\n",
            "    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')\n",
            "    sd.wait()\n",
            "    write(filename, fs, recording)\n",
            "    print(f\"Saved to {filename}\")\n",
            "    display(Audio(filename))\n",
            "\n",
            "def run_comparison(audio_path='test_input.wav'):\n",
            "    audio, _ = librosa.load(audio_path, sr=16000)\n",
            "    results = {}\n",
            "    \n",
            "    # 1. Fine-tuned Student (in memory)\n",
            "    input_features = processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to('cuda')\n",
            "    with torch.no_grad():\n",
            "        generated_ids = model.generate(input_features)\n",
            "    results['Fine-tuned Student'] = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]\n",
            "\n",
            "    # 2. Original Base Small\n",
            "    print(\"Loading Original Small...\")\n",
            "    base_model = WhisperForConditionalGeneration.from_pretrained(\n",
            "        'openai/whisper-small', \n",
            "        quantization_config=bnb_config, \n",
            "        device_map='auto'\n",
            "    )\n",
            "    with torch.no_grad():\n",
            "        base_ids = base_model.generate(input_features)\n",
            "    results['Original Small'] = processor.batch_decode(base_ids, skip_special_tokens=True)[0]\n",
            "    del base_model\n",
            "    torch.cuda.empty_cache()\n",
            "\n",
            "    # 3. Teacher Large-v3\n",
            "    print(\"Loading Teacher Large-v3...\")\n",
            "    teacher_pipe = pipeline(\n",
            "        'automatic-speech-recognition', \n",
            "        model='openai/whisper-large-v3', \n",
            "        model_kwargs={'quantization_config': bnb_config}, \n",
            "        device_map='auto'\n",
            "    )\n",
            "    teacher_out = teacher_pipe(audio)\n",
            "    results['Teacher Large-v3'] = teacher_out['text']\n",
            "    del teacher_pipe\n",
            "    torch.cuda.empty_cache()\n",
            "\n",
            "    # Display Results\n",
            "    html = f'''\n",
            "    <div style=\"padding:15px; border:1px solid #ddd; border-radius:10px;\">\n",
            "        <h3>Comparison Results</h3>\n",
            "        <p><b>Original Small:</b> {results['Original Small']}</p>\n",
            "        <p style=\"color:green;\"><b>Fine-tuned Student:</b> {results['Fine-tuned Student']}</p>\n",
            "        <p style=\"color:blue;\"><b>Teacher Large-v3:</b> {results['Teacher Large-v3']}</p>\n",
            "    </div>\n",
            "    '''\n",
            "    display(HTML(html))\n",
            "\n",
            "# Usage:\n",
            "# record_audio(duration=5)\n",
            "# run_comparison('test_input.wav')"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Egypt Audio (ASR)",
            "language": "python",
            "name": "egypt_audio_env"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
