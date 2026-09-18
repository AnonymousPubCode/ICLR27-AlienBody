#!/usr/bin/env python3
"""FMB Stage 1 Text-Modality Training.

Trains a text LLM (Qwen3.5) to predict ActionSchema from structured
text observations that include F4 relational context (cell colors,
nearest directions, brightness info).

Unlike the image-modality version, this bypasses the visual perception
bottleneck by feeding color values and spatial relations as text.

Usage:
    torchrun --nproc_per_node=8 scripts/train_fmb_stage1_text.py \
        --model /project/model/Qwen3.5-4B \
        --data data/fmb_trajectories/train_text.jsonl \
        --output models/fmb_stage1_text_4b \
        --epochs 10 --batch-size 4 --lr 1e-4
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
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ── Prompt Formatting ────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert at reverse-engineering action systems from observations.
Given structured observations from an unknown action space, predict the action's effect schema.
Output a JSON object with the effect type and parameters.

Effect types:
- translate: {"effect_type": "translate", "params": {"direction": "up"|"down"|"left"|"right"|"forward"|"backward"}}
- rotate: {"effect_type": "rotate", "params": {"rotation": "cw"|"ccw"}}
- state_change: {"effect_type": "state_change", "params": {"change": "inc"|"dec"|"color_move"|"color_interact"}}
- relational: {"effect_type": "relational", "params": {"relation": "nearest_diff"|"nearest_same"|"toward_brightest"|"flee_same"}}
- composite: {"effect_type": "composite", "params": {"primitives": [{"type": "translate", "params": {"direction": "forward"}}, {"type": "rotate", "params": {"rotation": "cw"}}]}}
- temporal: {"effect_type": "temporal", "params": {"same_dir": 0, "diff_dir": 1}}
- noop: {"effect_type": "noop", "params": {}}

Output ONLY the JSON object, no other text."""


def format_example_text(record: dict) -> str:
    """Format a training example as a full text prompt with F4 context."""
    action = record["action"]
    prev_state = record["prev_state"]
    effect = record["effect"]
    f4_ctx = record.get("f4_context", {})

    parts = [f"Action {action} was executed."]
    parts.append(f"Position: ({prev_state['pos'][0]}, {prev_state['pos'][1]})")
    parts.append(f"Facing: {['N','E','S','W'][prev_state['dir']]}")

    # F4 relational context
    if f4_ctx:
        cell_color = f4_ctx.get("cell_color", -1)
        color_name = f4_ctx.get("cell_color_name", f"c{cell_color}")
        parts.append(f"Cell color: {color_name}({cell_color})")

        if "nearest_diff_pos" in f4_ctx and f4_ctx["nearest_diff_pos"] is not None:
            nd = f4_ctx["nearest_diff_pos"]
            ndc = f4_ctx.get("nearest_diff_color", "?")
            parts.append(f"Nearest diff-color cell: ({nd[0]},{nd[1]}) color={ndc}")
        if "nearest_same_pos" in f4_ctx and f4_ctx["nearest_same_pos"] is not None:
            ns = f4_ctx["nearest_same_pos"]
            nsc = f4_ctx.get("nearest_same_color", "?")
            parts.append(f"Nearest same-color cell: ({ns[0]},{ns[1]}) color={nsc}")
        if "brightest_pos" in f4_ctx and f4_ctx["brightest_pos"] is not None:
            bp = f4_ctx["brightest_pos"]
            bc = f4_ctx.get("brightest_color", "?")
            parts.append(f"Brightest neighbor: ({bp[0]},{bp[1]}) color={bc}")
        if "flee_step" in f4_ctx and f4_ctx["flee_step"] is not None:
            fs = f4_ctx["flee_step"]
            parts.append(f"Flee direction step: ({fs[0]},{fs[1]})")

    # Observed effect
    dr, dc = effect.get("dr", 0), effect.get("dc", 0)
    ddir, dcolor = effect.get("ddir", 0), effect.get("dcolor", 0)
    dist = abs(dr) + abs(dc)
    if dist > 2:
        move_desc = f"TELEPORTED to relative ({dr:+d},{dc:+d})"
    elif dist > 0:
        dir_names = {(-1,0):"up",(1,0):"down",(0,-1):"left",(0,1):"right"}
        move_desc = f"moved {dir_names.get((dr,dc), f'({dr:+d},{dc:+d})')}"
    else:
        move_desc = "no position change"

    parts.append(f"Effect: {move_desc}")
    if ddir != 0:
        parts.append(f"Direction change: {'cw' if ddir in (1,-3) else 'ccw'}")
    if dcolor != 0:
        parts.append(f"Color change: {dcolor:+d}")

    parts.append(f"\nWhat is the ActionSchema for Action {action}?")

    return "\n".join(parts)


def load_dataset(jsonl_path: str) -> Dataset:
    """Load trajectory JSONL into HuggingFace Dataset."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for causal LM training."""
    texts = []
    for i in range(len(examples["action"])):
        f4_ctx = examples.get("f4_context", [None] * len(examples["action"]))
        if f4_ctx is None:
            f4_ctx = [None] * len(examples["action"])
        ctx = f4_ctx[i] if i < len(f4_ctx) else None
        rec = {
            "action": examples["action"][i],
            "prev_state": examples["prev_state"][i],
            "effect": examples["effect"][i],
            "f4_context": ctx or {},
        }
        user_prompt = format_example_text(rec)
        target = json.dumps(examples["ground_truth_schema"][i], ensure_ascii=False)

        full_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{target}<|im_end|>"
        )
        texts.append(full_text)

    tokenized = tokenizer(
        texts, truncation=True, padding="max_length", max_length=max_length
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ── Model Setup ──────────────────────────────────────────────────

def setup_model(model_path: str, lora_r: int = 16, lora_alpha: int = 32):
    """Load text model with LoRA adapter."""
    print(f"Loading text model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    for param in model.parameters():
        param.requires_grad = False

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
    return model


# ── Training ─────────────────────────────────────────────────────

def train(model_path, data_path, output_dir, epochs, batch_size, lr, lora_r):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(data_path)
    print(f"Loaded {len(dataset)} records")

    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    model = setup_model(model_path, lora_r=lora_r)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/fmb_stage1_text")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    args = parser.parse_args()

    train(args.model, args.data, args.output, args.epochs, args.batch_size, args.lr, args.lora_r)


if __name__ == "__main__":
    main()
