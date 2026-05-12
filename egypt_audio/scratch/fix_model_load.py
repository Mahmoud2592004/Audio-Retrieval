import json
import os

nb_path = r"d:\study\4th year 2nd term\NLP\Project 2 - Audio\egypt_audio\finetuning\finetune_whisper.ipynb"
if os.path.exists(nb_path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Corrected cell content for model loading
    model_load_cell = [
        "from transformers import BitsAndBytesConfig\n",
        "\n",
        "print(f\"📥 Loading {MODEL_NAME} in 4-bit...\")\n",
        "\n",
        "bnb_config = BitsAndBytesConfig(\n",
        "    load_in_4bit=True,\n",
        "    bnb_4bit_quant_type=\"nf4\",\n",
        "    bnb_4bit_compute_dtype=torch.float16,\n",
        "    bnb_4bit_use_double_quant=True,\n",
        ")\n",
        "\n",
        "model = WhisperForConditionalGeneration.from_pretrained(\n",
        "    MODEL_NAME, \n",
        "    quantization_config=bnb_config, \n",
        "    device_map=\"auto\"\n",
        ")\n",
        "\n",
        "model = prepare_model_for_kbit_training(model)\n",
        "\n",
        "config = LoraConfig(\n",
        "    r=32, \n",
        "    lora_alpha=64, \n",
        "    target_modules=[\"q_proj\", \"v_proj\"], \n",
        "    lora_dropout=0.05, \n",
        "    bias=\"none\"\n",
        ")\n",
        "\n",
        "model = get_peft_model(model, config)\n",
        "model.print_trainable_parameters()"
    ]

    # Find the cell that currently has the model loading code and replace it
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and "load_in_4bit=True" in "".join(cell["source"]):
            cell["source"] = model_load_cell
            break

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook model loading cell updated.")
else:
    print("Notebook not found.")
