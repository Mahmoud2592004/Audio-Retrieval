import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data_loader import EgyptDatasetBuilder

# --- CONFIGURATION ---
MODEL_NAME = "openai/whisper-small" # 244M params
AUDIO_DIR = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9"
TRANS_DIR = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\cut_clips9_transcribed"
OUTPUT_DIR = "./whisper-small-egyptian-lora"

def train():
    # 1. Load Processor
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="arabic", task="transcribe")
    
    # 2. Prepare Dataset
    builder = EgyptDatasetBuilder(AUDIO_DIR, TRANS_DIR)
    dataset = builder.build()
    
    # Split into train/test
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    def prepare_dataset(batch):
        # Load and resample audio
        audio = batch["audio"]
        # Compute log-Mel input features
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # Encode target text to label ids
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("🛠 Preprocessing dataset...")
    column_names = dataset["train"].column_names
    dataset = dataset.map(prepare_dataset, remove_columns=column_names, num_proc=1)

    # 3. Data Collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 4. Load Model with 4-bit Quantization (FOR 4GB VRAM)
    print(f"📥 Loading {MODEL_NAME} in 4-bit...")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        load_in_4bit=True, 
        device_map="auto",
        # bnb_4bit_compute_dtype=torch.float16
    )
    
    # 5. Prepare for PEFT (LoRA)
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 6. Training Arguments (Optimized for 4GB VRAM)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,      # Must be 1 for 4GB
        gradient_accumulation_steps=16,    # Total batch size = 16
        learning_rate=1e-4,
        warmup_steps=50,
        max_steps=500,                     # Adjust based on dataset size
        gradient_checkpointing=True,       # Essential for memory
        fp16=True,                         # Faster training
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,        # Important for Whisper
    )

    # 7. Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    print("🚀 Starting Fine-Tuning...")
    trainer.train()
    
    # Save final model
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Training complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
