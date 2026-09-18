#!/usr/bin/env python3
"""Generate Tier-XL environments (32x32, n=12, obstacles, P1=40, P2=120).

Tier-XL instantiation for the Relational family (F4): the full 12-effect
Type-C vocabulary (actions.py TYPE_C_EFFECTS, including 6 Tier-XL
extensions). This is the regime where permutation enumeration (12! ~ 4.8e8)
is intractable — the regime where an LLM proposer must earn its keep.

Usage:
  python scripts/gen_tier_xl.py --family 4 --split test --count 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.generator import generate_env, validate_solvable
from alienbody.env.actions import TYPE_C_EFFECTS

TIER_XL = {
    "grid_size": 32,
    "n_actions": 12,
    "phase1_budget": 40,
    "max_total_steps": 120,
    "min_distance": 16,
    "n_obstacles_range": (10, 15),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, default=4)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out", type=str, default="data/envs_xl")
    args = ap.parse_args()

    out_dir = Path(__file__).parent.parent / args.out / f"family{args.family}" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    generated, attempts, rejected = 0, 0, 0
    max_attempts = args.count * 400

    while generated < args.count and attempts < max_attempts:
        attempts += 1
        seed = args.family * 1000000 + 80000 + attempts
        config = generate_env(
            family=args.family, idx=generated, split=args.split, seed=seed,
            tier_tag="xl", effects_override=list(TYPE_C_EFFECTS), **TIER_XL,
        )
        result = validate_solvable(config)
        if result["solvable"] and result["optimal_steps"] >= 3:
            config.save(str(out_dir / f"env_{generated:03d}.json"))
            generated += 1
        else:
            rejected += 1

    print(f"family{args.family} Tier-XL {args.split}: generated={generated}, "
          f"rejected={rejected}, attempts={attempts}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
