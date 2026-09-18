"""RLVR Proposer Training: verification-filtered expert iteration.

Trains a Qwen3.5-9B (text) LoRA proposer to induce action->effect mappings
from Phase-1 observations. Reward = verification agreement: replay the
proposed mapping through the TRUE transition function over all observations
(deterministic, 0-1).

Algorithm (ReST / STaR-style expert iteration):
  round k:
    1. sample proposals from the current policy (T=1.0, G per env)
    2. verify each against the simulator; keep proposals with score >= tau
    3. LoRA SFT on (prompt, accepted proposal) pairs
    4. evaluate on test envs: verified rate + end-to-end SR (proposal + BFS)
  Repeat until the verified rate plateaus.

If the trained proposer crosses the induction frontier (in-context GPT-4o
reaches 46% verified on F4), the verification-guided induction loop becomes
a trainable capability.

Usage (on server, GPUs 5,6 via CUDA_VISIBLE_DEVICES):
    python scripts/train_fmb_proposer_rlvr.py \
        --base /project/model/Qwen3.5-9B \
        --output models/proposer_rlvr_9b \
        --families 4,5 --rounds 6 --steps-per-round 120 \
        --group-size 8 --tau 0.8 --lr 1e-4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig, GridState, Position, Phase
from alienbody.env.actions import apply_action
from alienbody.agents.fmb_inductive import (
    RELATION_TYPES, COMPOSITE_TYPES, TEMPORAL_TYPES, _JSON_RE, _KV_RE,
)

# ── Prompt / parsing (mirrors fmb_inductive.py) ──────────────────

_SYSTEM_INDUCTION = """You are identifying what {n_actions} anonymous actions do in a grid world.
Each action is exactly one of the following types:

{vocab}

Below are the grid's cell colors (numbers 0-9; 0 = black) and the transitions
observed while testing the actions. Propose the complete mapping.

Grid colors (row, col -> color):
{grid}

Observed transitions (position -> action -> new position, with the color of the
cell the agent stood on):
{observations}

Output ONLY a JSON object mapping action index to type name, e.g.
{{"0": "move_to_nearest_same_color", "1": "move_toward_brightest", ...}}
Your answer:"""


def vocab_for(config: EnvConfig) -> dict:
    at = config.action_type
    if at == "C":
        return dict(list(RELATION_TYPES.items())[:config.n_actions])
    if at == "E":
        return COMPOSITE_TYPES
    if at == "F":
        return TEMPORAL_TYPES
    return RELATION_TYPES


def parse_mapping(response: str, config: EnvConfig) -> dict[int, str] | None:
    vocab = vocab_for(config)

    def _finish(raw: dict) -> dict[int, str] | None:
        mapping = {}
        for k, v in raw.items():
            try:
                a = int(str(k).strip())
            except ValueError:
                continue
            if a < 0 or a >= config.n_actions:
                continue
            name = str(v).strip()
            if name in vocab:
                mapping[a] = name
        return mapping if len(mapping) == config.n_actions else None

    m = _JSON_RE.search(response or "")
    if m:
        try:
            parsed = _finish(json.loads(m.group(0)))
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    pairs = {}
    for k, v in _KV_RE.findall(response or ""):
        a = int(k)
        if 0 <= a < config.n_actions and v in vocab:
            pairs[a] = v
    return _finish(pairs)


# ── Environment sampling ─────────────────────────────────────────

def run_phase1(config: EnvConfig) -> list[dict]:
    """Round-robin Phase-1 exploration; returns the observation sequence."""
    env = AlienBodyEnv(config, render_mode="both")
    obs, info = env.reset()
    seq = []
    last_action = -1
    for i in range(config.phase1_budget):
        a = i % config.n_actions
        prev_pos = info["agent_pos"]
        prev_dir = info["agent_dir"]
        obs, r, t, tr, info = env.step(a)
        seq.append({
            "action": a, "prev_pos": prev_pos,
            "new_pos": info["agent_pos"],
            "prev_action": last_action, "prev_dir": prev_dir,
        })
        last_action = a
    return seq


def build_prompt(config: EnvConfig, seq: list[dict]) -> str:
    vocab = vocab_for(config)
    grid_lines = []
    for r in range(config.grid_size):
        grid_lines.append(
            "  row %2d: %s" % (r, " ".join(str(config.cell_colors[r][c])
                                            for c in range(config.grid_size))))
    obs_lines = []
    for o in seq:
        pr, pc = o["prev_pos"]
        nr, nc = o["new_pos"]
        col = config.cell_colors[pr][pc]
        prev = o["prev_action"]
        prev_str = f"prev={prev}, " if prev >= 0 else ""
        obs_lines.append(
            f"  at ({pr},{pc}) color={col} facing={o['prev_dir']}, "
            f"{prev_str}Action {o['action']} -> ({nr},{nc})")
    return _SYSTEM_INDUCTION.format(
        n_actions=config.n_actions,
        vocab="\n".join(f"  {name}: {info[1]}" for name, info in vocab.items()),
        grid="\n".join(grid_lines),
        observations="\n".join(obs_lines) if obs_lines else "  (no observations)",
    )


def verify(mapping: dict[int, str] | None, config: EnvConfig,
           seq: list[dict]) -> float:
    """Deterministic verification reward in [0, 1]."""
    if not mapping:
        return 0.0
    correct = 0
    for o in seq:
        a = o["action"]
        pr, pc = o["prev_pos"]
        state = GridState(agent_pos=Position(pr, pc), agent_color=0,
                          agent_dir=o["prev_dir"], phase=Phase.EXECUTION,
                          step_count=0, phase1_steps=0, phase2_steps=0,
                          prev_action=o["prev_action"])
        temp_cfg = copy.deepcopy(config)
        object.__setattr__(temp_cfg, 'action_mapping',
                           tuple(mapping.get(i, "x")
                                 for i in range(config.n_actions)))
        new_state = apply_action(copy.deepcopy(state), temp_cfg, a)
        if (new_state.agent_pos.row, new_state.agent_pos.col) == tuple(o["new_pos"]):
            correct += 1
    return correct / len(seq) if seq else 0.0


# ── Model loading / generation / SFT ─────────────────────────────

def load_envs(families: list[int], split: str, limit: int) -> list[EnvConfig]:
    out = []
    base = Path(__file__).parent.parent / "data" / "envs"
    for fam in families:
        for f in sorted((base / f"family{fam}" / split).glob("env_*.json"))[:limit]:
            out.append(EnvConfig.from_file(str(f)))
    return out


def load_policy(base_model: str, lora_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    # Single-GPU: CUDA_VISIBLE_DEVICES picks the card; avoids device_map="auto"
    # hangs seen with multi-card Qwen generate.
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda()
    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tok


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], n_per: int,
                   max_new_tokens: int = 300, batch_size: int = 8) -> list[str]:
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        inputs = tok(chunk * n_per, return_tensors="pt", padding=True,
                     truncation=True, max_length=2048).to("cuda")
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=1.0, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
        for row in ids:
            out.append(tok.decode(row[inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
    return out


def sft_step(model, tok, pairs: list[tuple[str, str]], epochs: int,
             lr: float, out_dir: Path):
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset

    class PairDataset(Dataset):
        def __init__(self, pairs, tok):
            self.items = []
            for p, a in pairs:
                text = p + a
                enc = tok(text, truncation=True, max_length=1536,
                          return_tensors="pt")
                self.items.append(enc)
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            enc = self.items[i]
            return {"input_ids": enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                    "labels": enc["input_ids"][0].clone()}

    if not getattr(model, "is_peft_model", False):
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj", "up_proj",
                                          "down_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.train()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir), per_device_train_batch_size=1,
            gradient_accumulation_steps=4, num_train_epochs=epochs,
            learning_rate=lr, bf16=True, logging_steps=5,
            save_strategy="no", report_to=[]),
        train_dataset=PairDataset(pairs, tok))
    trainer.train()
    model.eval()
    return model


# ── Main loop ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/project/model/Qwen3.5-9B")
    ap.add_argument("--output", default="models/proposer_rlvr_9b")
    ap.add_argument("--families", default="4,5")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps-per-round", type=int, default=120)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.8,
                    help="final acceptance threshold (curriculum: starts at 0.3)")
    ap.add_argument("--bootstrap-epochs", type=int, default=2,
                    help="SFT epochs on (prompt, ground-truth) pairs before "
                         "self-training (0 to disable)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sft-epochs", type=int, default=2)
    ap.add_argument("--test-envs", type=int, default=50)
    args = ap.parse_args()

    fams = [int(x) for x in args.families.split(",")]
    train_envs = load_envs(fams, "train", limit=300)
    test_envs = load_envs(fams, "test", limit=args.test_envs)

    model, tok = load_policy(args.base)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-generate observation prompts for train/test envs
    def make_tasks(envs):
        tasks = []
        for cfg in envs:
            seq = run_phase1(cfg)
            tasks.append((cfg, seq, build_prompt(cfg, seq)))
        return tasks

    print(f"[RLVR] {len(train_envs)} train envs, {len(test_envs)} test envs")
    train_tasks = make_tasks(train_envs)
    test_tasks = make_tasks(test_envs)

    # GT bootstrap: SFT on (prompt, ground-truth mapping) pairs so the
    # policy learns the output format and vocabulary before self-training.
    if args.bootstrap_epochs > 0:
        gt_pairs = []
        for cfg, seq, prompt in train_tasks:
            gt_json = json.dumps(
                {str(i): cfg.action_mapping[i] for i in range(cfg.n_actions)})
            gt_pairs.append((prompt, gt_json))
        print(f"[RLVR] GT bootstrap: {len(gt_pairs)} pairs, "
              f"{args.bootstrap_epochs} epochs")
        model = sft_step(model, tok, gt_pairs, args.bootstrap_epochs,
                         args.lr, out_dir)

    history = []
    for rnd in range(args.rounds):
        tau = min(args.tau, 0.3 + 0.1 * rnd)  # curriculum on acceptance
        t0 = time.time()
        # Sample from the current policy
        rng = np.random.default_rng(rnd)
        chosen = [train_tasks[i] for i in rng.choice(len(train_tasks),
                                                     args.steps_per_round,
                                                     replace=True)]
        prompts = [t[2] for t in chosen]
        responses = generate_batch(model, tok, prompts, args.group_size)
        # Verify and filter
        pairs = []
        scores = []
        for (cfg, seq, prompt), resp in zip(chosen, responses):
            mapping = parse_mapping(resp, cfg)
            score = verify(mapping, cfg, seq)
            scores.append(score)
            if score >= tau:
                pairs.append((prompt, resp))
        mean_score = float(np.mean(scores))
        acc_rate = len(pairs) / len(scores)
        print(f"[round {rnd}] tau={tau:.2f} mean score={mean_score:.3f} "
              f"accepted={len(pairs)}/{len(scores)} ({acc_rate:.0%}) "
              f"({time.time()-t0:.0f}s)")
        # SFT on accepted pairs
        if len(pairs) < 8:
            print("[round %d] too few accepted pairs, stopping" % rnd)
            break
        model = sft_step(model, tok, pairs, args.sft_epochs, args.lr, out_dir)
        try:
            model.save_pretrained(str(out_dir / f"round_{rnd}"))
        except Exception as e:
            print(f"[round {rnd}] save failed: {e}", flush=True)
        # Evaluate on test (batched generation)
        test_prompts = [t[2] for t in test_tasks]
        test_resps = generate_batch(model, tok, test_prompts, 1)
        test_scores = [verify(parse_mapping(resp, cfg), cfg, seq)
                       for (cfg, seq, _), resp in zip(test_tasks, test_resps)]
        hist_entry = {
            "round": rnd, "train_mean_score": mean_score,
            "accepted": len(pairs), "test_verified_rate":
                float(np.mean([1.0 if s >= 0.99 else 0.0 for s in test_scores])),
            "test_mean_score": float(np.mean(test_scores)),
        }
        history.append(hist_entry)
        print(f"[round {rnd}] test: verified "
              f"{hist_entry['test_verified_rate']:.0%}, mean "
              f"{hist_entry['test_mean_score']:.3f}")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("[RLVR] done")


if __name__ == "__main__":
    main()
