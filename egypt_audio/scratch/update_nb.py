import json
import os

nb_path = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\egypt_audio\finetuning\finetune_whisper.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Full corrected cell content for dataset preparation
    dataset_prep_cell = [
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
        "def prepare_dataset(batch):\n",
        "    import librosa\n",
        "    audio_path = batch[\"audio\"]\n",
        "    # Load manually since the automatic Audio feature is disabled\n",
        "    audio_array, _ = librosa.load(audio_path, sr=16000)\n",
        "    batch[\"input_features\"] = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]\n",
        "    batch[\"labels\"] = processor.tokenizer(batch[\"sentence\"]).input_ids\n",
        "    return batch\n",
        "\n",
        "print(\"🛠 Preprocessing dataset (this may take a minute)...\")\n",
        "dataset = dataset.map(prepare_dataset, remove_columns=dataset[\"train\"].column_names, num_proc=1)"
    ]

    # Find the cell that currently has prepare_dataset and replace its content
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "def prepare_dataset(batch):" in "".join(cell["source"]):
            cell["source"] = dataset_prep_cell
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated correctly.")
else:
    print("Notebook not found.")
