#!/usr/bin/env python3
"""Multi-observation F4 induction training (reviewer B fix).

The original trained variants saw only the action's LAST observation
(fmb_text.py `records[-1]`). This trainer builds examples where the model
sees ALL Phase-1 observations of the action (12-27 records per action from
f4_scale_2000.jsonl, grouped by (env_id, action)) and must output the
action's ActionSchema. Same LoRA recipe as train_fmb_stage1_text.py.

Usage (on the A800 server):
  python scripts/train_fmb_multiobs.py \
      --model models/Qwen3.5-9B \
      --data data/fmb_trajectories/f4_scale_2000.jsonl \
      --output models/fmb_multiobs_9b
"""
from __future__ import annotations

import os
# Pin to GPU 6 (the only card with free memory on the training host);
# must run before any torch import.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)

SYSTEM_PROMPT_MULTIOBS = """You are identifying what an anonymous action does in a grid world with colored cells.
The action is one of the following types:
- {"effect_type": "translate", "params": {"direction": "up|down|left|right|forward|backward"}}
- {"effect_type": "rotate", "params": {"rotation": "cw|ccw"}}
- {"effect_type": "state_change", "params": {"change": "inc|dec|color_move|color_interact"}}
- {"effect_type": "relational", "params": {"relation": "nearest_diff|nearest_same|toward_brightest|flee_same|farthest_same|away_brightest|farthest_diff|flee_nearest_diff|toward_darkest|away_darkest|nearest_brighter|nearest_darker"}}
- {"effect_type": "noop", "params": {}}

Output ONLY the JSON object, no other text."""


def _dir_name(d: int) -> str:
    return {0: "N", 1: "E", 2: "S", 3: "W"}.get(d % 4, "?")


def _format_one_obs(action: int, rec: dict, ctx: dict) -> str:
    ps = rec["prev_state"]
    eff = rec["effect"]
    parts = [f"Action {action} was executed."]
    parts.append(f"Position: ({ps['pos'][0]}, {ps['pos'][1]})")
    parts.append(f"Facing: {_dir_name(ps['dir'])}")
    if ctx:
        cc = ctx.get("cell_color", -1)
        parts.append(f"Cell color: c{cc}")
        if ctx.get("nearest_diff_pos") is not None:
            nd = ctx["nearest_diff_pos"]
            parts.append(f"Nearest diff-color cell: ({nd[0]},{nd[1]}) color={ctx.get('nearest_diff_color','?')}")
        if ctx.get("nearest_same_pos") is not None:
            ns = ctx["nearest_same_pos"]
            parts.append(f"Nearest same-color cell: ({ns[0]},{ns[1]}) color={ctx.get('nearest_same_color','?')}")
        if ctx.get("brightest_pos") is not None:
            bp = ctx["brightest_pos"]
            parts.append(f"Brightest neighbor: ({bp[0]},{bp[1]}) color={ctx.get('brightest_color','?')}")
    dr, dc = eff.get("dr", 0), eff.get("dc", 0)
    dist = abs(dr) + abs(dc)
    if dist > 2:
        mv = f"TELEPORTED to relative ({dr:+d},{dc:+d})"
    elif dist > 0:
        mv = f"moved {_dir_name({(-1,0):0,(1,0):2,(0,-1):3,(0,1):1}.get((dr,dc), 0)).lower()}"
    else:
        mv = "no position change"
    parts.append(f"Effect: {mv}")
    return " | ".join(parts)


def build_multiobs_prompt(action: int, recs: list[dict], ctxs: list[dict]) -> str:
    """All observations of one action in one env -> schema question."""
    obs_lines = [f"Observation {i+1}: {_format_one_obs(action, r, ctxs[i] if i < len(ctxs) else {})}"
                 for i, r in enumerate(recs)]
    return ("Here are the Phase-1 observations for Action "
            f"{action}:\n" + "\n".join(obs_lines) +
            f"\n\nWhat is the ActionSchema for Action {action}?")


def load_grouped(jsonl_path: str) -> list[dict]:
    groups = defaultdict(list)
    for line in open(jsonl_path, encoding="utf-8"):
        r = json.loads(line)
        groups[(r["env_id"], r["action"])].append(r)
    examples = []
    for (env_id, action), recs in groups.items():
        # sort by original order (records are appended in exploration order)
        recs = sorted(recs, key=lambda r: r.get("_idx", 0))
        examples.append({
            "env_id": env_id, "action": action, "recs": recs,
        })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-9B")
    ap.add_argument("--data", default="data/fmb_trajectories/f4_scale_2000.jsonl")
    ap.add_argument("--output", default="models/fmb_multiobs_9b")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42,
                    help="Global seed for the induction-frontier repro")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"Repro seed: {args.seed}")

    examples = load_grouped(args.data)
    rows = []
    for ex in examples:
        recs = ex["recs"]
        ctxs = [r.get("f4_context", {}) for r in recs]
        user = build_multiobs_prompt(ex["action"], recs, ctxs)
        target = json.dumps(recs[0]["ground_truth_schema"])  # all recs share GT
        rows.append({
            "text": (f"<|im_start|>system\n{SYSTEM_PROMPT_MULTIOBS}<|im_end|>\n"
                     f"<|im_start|>user\n{user}<|im_end|>\n"
                     f"<|im_start|>assistant\n{target}<|im_end|>"),
        })
    ds = Dataset.from_list(rows)
    print(f"Loaded {len(ds)} multi-observation examples")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)

    def tokenize(ex):
        out = tok(ex["text"], truncation=True, max_length=4096)
        return out

    ds = ds.map(tokenize, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8, learning_rate=args.lr,
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=10, save_strategy="epoch", bf16=True,
        report_to="none",
        seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)
    trainer.train()
    trainer.save_model(str(Path(args.output) / "final"))
    print(f"saved to {args.output}/final")


if __name__ == "__main__":
    main()
