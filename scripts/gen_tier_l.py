#!/usr/bin/env python3
"""Generate Tier-L environments (24x24, n=6, obstacles, P1=25, P2=80).

C3: instantiate the Tier-L difficulty tier (paper Table: tab:difficulty)
for the Relational family (F4) — the family on which the planning wall
and induction frontier live. Other families' effect vocabularies remain
n=4; F4's Type-C vocabulary has 6 effects (actions.py TYPE_C_EFFECTS).

Usage:
  python scripts/gen_tier_l.py --family 4 --split test --count 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.generator import generate_env, validate_solvable

# Tier-L canonical parameters (aligned with paper Table tab:difficulty)
TIER_L = {
    "grid_size": 24,
    "n_actions": 6,
    "phase1_budget": 25,
    "max_total_steps": 80,
    "min_distance": 8,
    "n_obstacles_range": (5, 8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", type=int, default=4)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--out", type=str, default="data/envs_l")
    args = ap.parse_args()

    out_dir = Path(__file__).parent.parent / args.out / f"family{args.family}" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    generated, attempts, rejected = 0, 0, 0
    max_attempts = args.count * 200

    while generated < args.count and attempts < max_attempts:
        attempts += 1
        seed = args.family * 1000000 + 70000 + attempts
        # Tier-L uses the full 6-effect relational vocabulary
        from alienbody.env.actions import TYPE_C_EFFECTS
        config = generate_env(
            family=args.family, idx=generated, split=args.split, seed=seed,
            tier_tag="l", effects_override=list(TYPE_C_EFFECTS), **TIER_L,
        )
        result = validate_solvable(config)
        if result["solvable"] and result["optimal_steps"] >= 3:
            config.save(str(out_dir / f"env_{generated:03d}.json"))
            generated += 1
        else:
            rejected += 1

    print(f"family{args.family} Tier-L {args.split}: generated={generated}, "
          f"rejected={rejected}, attempts={attempts}")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
