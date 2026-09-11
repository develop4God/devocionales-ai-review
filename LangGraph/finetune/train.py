"""
LoRA fine-tune runner for the ES devotional-content critic model.

Model-agnostic by design: which base model to train is selected via --model-id,
resolved against LangGraph/config/finetune_models.yml. Nothing about the base
model (checkpoint path, chat template, LoRA target modules) is hardcoded here --
switching from Gemma 3 1B to Gemma 4 E4B, or to a future DeepSeek entry, is a
config change in that YAML file, not a change to this script.

Usage:
    source .venv/bin/activate
    python3 train.py --model-id gemma3_1b
    python3 train.py --model-id gemma4_4b_e4b --local-model-dir ./models/gemma-4-E4B-it-unsloth-bnb-4bit
"""
import argparse
import json
from pathlib import Path

import yaml

FINETUNE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FINETUNE_DIR.parents[1]
CONFIG_PATH = REPO_ROOT / "LangGraph" / "config" / "finetune_models.yml"
DATASET_PATH = REPO_ROOT / "LangGraph" / "data" / "finetune" / "es_dataset_alpaca.json"

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def load_model_config(model_id: str) -> dict:
    config = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    for m in config["models"]:
        if m["id"] == model_id:
            if not m["enabled"]:
                raise ValueError(f"Model '{model_id}' is present in {CONFIG_PATH} but disabled.")
            return m
    raise ValueError(f"No model with id '{model_id}' found in {CONFIG_PATH}")


def format_alpaca(examples, tokenizer):
    eos = tokenizer.eos_token
    texts = [
        ALPACA_PROMPT.format(instr, inp, out) + eos
        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=None, help="Model id from finetune_models.yml; defaults to settings.default_model")
    parser.add_argument("--local-model-dir", default=None, help="Local directory with the pre-downloaded model, overrides base_model (loads from disk, not the Hub)")
    parser.add_argument("--output-dir", default=str(FINETUNE_DIR / "outputs"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    config = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    model_id = args.model_id or config["settings"]["default_model"]
    model_cfg = load_model_config(model_id)

    print(f"Training model: {model_cfg['name']} (id={model_id})")

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    load_path = args.local_model_dir or model_cfg["base_model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,
    )

    lora_cfg = model_cfg["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    raw = json.load(open(DATASET_PATH, encoding="utf-8"))
    # strip the audit-only fields before handing to the trainer
    clean = [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]} for r in raw]
    dataset = Dataset.from_list(clean)
    dataset = dataset.map(lambda ex: format_alpaca(ex, tokenizer), batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_cfg["max_seq_length"],
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=not model_cfg["load_in_4bit"],
            bf16=False,
            logging_steps=10,
            optim="adamw_8bit",
            output_dir=args.output_dir,
            save_strategy="epoch",
        ),
    )

    trainer.train()

    adapter_dir = Path(args.output_dir) / f"{model_id}_lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
