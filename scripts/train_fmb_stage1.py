#!/usr/bin/env python3
"""FMB Stage 1 Training: LoRA fine-tune VLM to predict ActionSchema.

Trains Qwen3-VL (or any VLM) to predict structured action effects from
a single observation: (grid image, action_id) -> ActionSchema JSON.

This teaches the VLM the "induction primitive" — what an ActionSchema
looks like for each type of observed effect — before Stage 2 GRPO
teaches it to actively design experiments.

Usage:
    python scripts/train_fmb_stage1.py \
        --model models/Qwen3-VL-4B-Instruct \
        --data data/fmb_trajectories/train.jsonl \
        --output models/fmb_stage1 \
        --epochs 3 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# ── Data Loading ─────────────────────────────────────────────────

def load_dataset(jsonl_path: str) -> Dataset:
    """Load trajectory JSONL into HuggingFace Dataset."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ── Prompt Formatting ────────────────────────────────────────────

SYSTEM_PROMPT = """You are analyzing an unknown action system.
Given a grid image and an action index, predict the action's effect.
Output a JSON object with the effect type and parameters.

Effect types:
- translate: {"effect_type": "translate", "params": {"direction": "up"|"down"|"left"|"right"|"forward"|"backward"}}
- rotate: {"effect_type": "rotate", "params": {"rotation": "cw"|"ccw"}}
- state_change: {"effect_type": "state_change", "params": {"change": "inc"|"dec"|"color_move"|"color_interact"}}
- relational: {"effect_type": "relational", "params": {"relation": "nearest_diff"|"nearest_same"|"toward_brightest"|"flee_same"}}
- composite: {"effect_type": "composite", "params": {"primitives": [...]}}
- temporal: {"effect_type": "temporal", "params": {"same_dir": 0, "diff_dir": 1}}
- noop: {"effect_type": "noop", "params": {}}

Output ONLY the JSON object, no other text."""


def format_example(record: dict) -> dict:
    """Format a single training example as (messages, target_text)."""
    action = record["action"]
    prev_state = record["prev_state"]
    effect = record["effect"]

    user_prompt = (
        f"Action {action} was executed.\n"
        f"Initial state: position={prev_state['pos']}, "
        f"facing={['N','E','S','W'][prev_state['dir']]}, "
        f"color={prev_state['color']}\n"
        f"Observed effect: dr={effect['dr']}, dc={effect['dc']}, "
        f"ddir={effect['ddir']}, dcolor={effect['dcolor']}\n"
        f"\nWhat is the ActionSchema for Action {action}?"
    )

    target = json.dumps(record["ground_truth_schema"], ensure_ascii=False)

    return {
        "user_prompt": user_prompt,
        "target": target,
    }


# ── Model Setup ──────────────────────────────────────────────────

def setup_model(model_path: str, lora_r: int = 16, lora_alpha: int = 32):
    """Load base model with LoRA adapter."""
    print(f"Loading model from {model_path}...")

    # Use bfloat16 for A800
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    return model, processor


# ── Data Collator ────────────────────────────────────────────────

class SchemaDataCollator:
    """Collate (messages, target) pairs for training."""

    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        texts = []
        for item in batch:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{item['user_prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['target']}<|im_end|>"
            )
            texts.append(text)

        # Tokenize
        tokenized = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Labels = input_ids (standard causal LM training)
        tokenized["labels"] = tokenized["input_ids"].clone()

        # Mask non-assistant tokens (only compute loss on the schema JSON)
        for i, text in enumerate(texts):
            assistant_start = text.find("<|im_start|>assistant\n")
            if assistant_start >= 0:
                prefix_tokens = self.processor(
                    text=text[:assistant_start],
                    return_tensors="pt",
                )
                prefix_len = prefix_tokens["input_ids"].shape[1]
                tokenized["labels"][i, :prefix_len] = -100

        return tokenized


# ── Training ─────────────────────────────────────────────────────

def train(
    model_path: str,
    data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    max_length: int = 512,
):
    """Run Stage 1 LoRA training."""
    # Load data
    print(f"Loading data from {data_path}...")
    dataset = load_dataset(data_path)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    print(f"  {len(dataset)} training examples")

    # Split train/val
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = dataset["train"]
    val_ds = dataset["test"]
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Setup model
    model, processor = setup_model(model_path, lora_r=lora_r)
    collator = SchemaDataCollator(processor, max_length=max_length)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\nModel saved to {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="FMB Stage 1 Training")
    parser.add_argument("--model", type=str, required=True,
                        help="Base model path (e.g., models/Qwen3-VL-4B-Instruct)")
    parser.add_argument("--data", type=str, required=True,
                        help="Training data JSONL path")
    parser.add_argument("--output", type=str, default="models/fmb_stage1",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    train(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_r=args.lora_r,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
