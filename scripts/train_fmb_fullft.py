#!/usr/bin/env python3
"""Full Fine-tune (no LoRA) on F4 data.

ARCHIVAL RECORD — NOT EVIDENCE.  This trainer is shipped as the historical
record of the paper's full-FT row, which carries two defects found after the
original run:
  (1) the training examples include the ground-truth ``Schema:`` line, i.e.
      the target schema is part of the input (label leakage), and
  (2) the evaluation path fell back to base weights when no
      ``adapter_config.json`` was present, so the reported row was not a
      trained model.
The paper keeps this row as a record that contributes no evidence.  Do not
quote numbers produced by this script as a capability result.

Usage (single device; the original run used 4xA800 via torchrun):
    python scripts/train_fmb_fullft.py \
        --model models/Qwen3.5-4B \
        --data data/fmb_trajectories/f4_scale_2000.jsonl \
        --output models/f4_fullft_2000 --epochs 20
    # or: torchrun --nproc_per_node=4 scripts/train_fmb_fullft.py ...
"""
import argparse, json, os, random, sys
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ── Prompt (same as standard text training) ─────────────────────

SYSTEM_PROMPT = """You are an expert at reverse-engineering action systems from observations.
Given structured observations from an unknown action space, predict the action's effect schema.
Output a JSON object with the effect type and parameters.

Effect types:
- translate: {"effect_type": "translate", "params": {"direction": "up"|"down"|"left"|"right"|"forward"|"backward"}}
- rotate: {"effect_type": "rotate", "params": {"rotation": "cw"|"ccw"}}
- state_change: {"effect_type": "state_change", "params": {"change": "inc"|"dec"|"color_move"|"color_interact"}}
- relational: {"effect_type": "relational", "params": {"relation": "nearest_diff"|"nearest_same"|"toward_brightest"|"flee_same"}}
- composite: {"effect_type": "composite", "params": {"primitives": []}}
- temporal: {"effect_type": "temporal", "params": {"same_dir": 0, "diff_dir": 1}}
- noop: {"effect_type": "noop", "params": {}}

Output ONLY the JSON object, no other text."""


def format_example(rec):
    action = rec["action"]
    ps = rec["prev_state"]
    eff = rec["effect"]
    f4 = rec.get("f4_context", {})
    parts = [f"Action {action} was executed.",
             f"Position: ({ps['pos'][0]}, {ps['pos'][1]})",
             f"Facing: {['N','E','S','W'][ps['dir']]}"]
    if f4:
        parts.append(f"Cell color: c{f4.get('cell_color','?')}({f4.get('cell_color','?')})")
        if f4.get("nearest_diff_pos"):
            parts.append(f"Nearest diff: ({f4['nearest_diff_pos'][0]},{f4['nearest_diff_pos'][1]}) c{f4.get('nearest_diff_color','?')}")
        if f4.get("nearest_same_pos"):
            parts.append(f"Nearest same: ({f4['nearest_same_pos'][0]},{f4['nearest_same_pos'][1]}) c{f4.get('nearest_same_color','?')}")
        if f4.get("brightest_pos"):
            parts.append(f"Brightest: ({f4['brightest_pos'][0]},{f4['brightest_pos'][1]}) c{f4.get('brightest_color','?')}")
    parts.append(f"Effect: dr={eff['dr']}, dc={eff['dc']}, ddir={eff['ddir']}")
    gt = rec["ground_truth_schema"]
    parts.append(f"Schema: {json.dumps(gt)}")
    text = "\n".join(parts)
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nObserve:\n\n{text}<|im_end|>\n<|im_start|>assistant\n{json.dumps(gt)}<|im_end|>"


def load_data(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return Dataset.from_list(recs)


def tokenize_fn(examples, tokenizer):
    n = len(examples["action"])
    texts = []
    for i in range(n):
        rec = {k: examples[k][i] for k in ["action","prev_state","effect","ground_truth_schema"]}
        if "f4_context" in examples and examples["f4_context"]:
            rec["f4_context"] = examples["f4_context"][i] if i < len(examples["f4_context"]) else {}
        else:
            rec["f4_context"] = {}
        texts.append(format_example(rec))
    return tokenizer(texts, truncation=True, max_length=2048, padding=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="models/f4_fullft")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)  # lower LR for full FT
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

    ds = load_data(args.data)
    print(f"Loaded {len(ds)} records")
    tokenized = ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True,
                       remove_columns=ds.column_names)

    print(f"Loading model (FULL FT, no LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # ALL parameters are trainable (no freeze)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {total:,} (ALL)")

    ta = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=8,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=10, save_steps=200, save_total_limit=2,
        bf16=True, gradient_checkpointing=True,
        ddp_find_unused_parameters=False, report_to="none",
        dataloader_num_workers=0, optim="adamw_torch_fused",
        seed=args.seed, data_seed=args.seed,
    )

    trainer = Trainer(model=model, args=ta, train_dataset=tokenized,
                      data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    print(f"Starting FULL FT: {args.epochs} epochs, lr={args.lr}")
    trainer.train()

    final = os.path.join(args.output, "final")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Saved to {final}")

if __name__ == "__main__":
    main()
