#!/usr/bin/env python3
"""CoT (Chain-of-Thought) FMB training — model learns to reason before outputting schema.

Unlike standard training which outputs raw JSON, this trains the model to:
  1. First output <reasoning> tags with step-by-step analysis
  2. Then output <schema> tags with the JSON

Usage:
    torchrun --nproc_per_node=1 scripts/train_fmb_cot.py \
        --model models/Qwen3.5-9B \
        --data data/fmb_trajectories/f4_cot_2000.jsonl \
        --output models/f4_cot_2000 --epochs 30
"""
import argparse, json, os, random, sys
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)

def load_cot_dataset(path):
    """Load CoT dataset where each record has a pre-formatted 'text' field."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records.append({"text": r["text"]})
    return Dataset.from_list(records)

def tokenize_fn(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="models/f4_cot")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42,
                   help="Global seed for the induction-frontier repro")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Repro seed: {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_cot_dataset(args.data)
    print(f"Loaded {len(ds)} CoT records")
    tokenized = ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True,
                       remove_columns=ds.column_names)

    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    for p in model.parameters():
        p.requires_grad = False

    lora = LoraConfig(r=16, lora_alpha=32,
                      target_modules=["q_proj","v_proj","k_proj","o_proj"],
                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ta = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=4,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=10, save_steps=200, save_total_limit=2,
        bf16=True, gradient_checkpointing=True, report_to="none",
        dataloader_num_workers=0, optim="adamw_torch_fused",
        seed=args.seed, data_seed=args.seed,
    )

    trainer = Trainer(model=model, args=ta, train_dataset=tokenized,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print(f"Starting CoT training: {args.epochs} epochs")
    trainer.train()

    final = os.path.join(args.output, "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Saved to {final}")

if __name__ == "__main__":
    main()
