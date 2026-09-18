#!/usr/bin/env python3
"""Enrich training data with F4 relational context for text-modality training.

Reads standard trajectory JSONL, loads each env config, and adds
cell-color-aware context needed for F4 relational schema induction.

Usage:
    python scripts/enrich_training_data.py \
        --input data/fmb_trajectories/train_multi.jsonl \
        --output data/fmb_trajectories/train_text.jsonl \
        --data-dir data/envs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.grid import EnvConfig, Position
from alienbody.env.actions import _find_nearest_by_color, TYPE_C_EFFECTS
from alienbody.env.renderer import COLOR_NAMES

DIRECTION_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}


def compute_f4_context(pr: int, pc: int, config: EnvConfig) -> dict:
    """Compute F4 relational context for a given position."""
    gs = config.grid_size
    if not (0 <= pr < gs and 0 <= pc < gs):
        return {}

    agent_cell_color = config.cell_colors[pr][pc]
    ctx = {
        "cell_color": agent_cell_color,
        "cell_color_name": COLOR_NAMES.get(agent_cell_color, f"c{agent_cell_color}"),
    }

    # nearest different-color cell
    nd = _find_nearest_by_color(Position(pr, pc), config, same=False, ref_color=agent_cell_color)
    if nd:
        ctx["nearest_diff_pos"] = [nd.row, nd.col]
        ctx["nearest_diff_color"] = config.cell_colors[nd.row][nd.col]

    # nearest same-color cell
    ns = _find_nearest_by_color(Position(pr, pc), config, same=True, ref_color=agent_cell_color)
    if ns:
        ctx["nearest_same_pos"] = [ns.row, ns.col]
        ctx["nearest_same_color"] = config.cell_colors[ns.row][ns.col]

    # brightest 4-neighbor
    obstacles = set()
    if hasattr(config, 'obstacles') and config.obstacles:
        obstacles = {(r, c) for r, c in config.obstacles}
    from alienbody.env.grid import DIRECTION_DELTAS
    best_pos = None
    best_color = -1
    for d_idx in range(4):
        delta = DIRECTION_DELTAS[d_idx]
        nb_r, nb_c = pr + delta[0], pc + delta[1]
        if 0 <= nb_r < gs and 0 <= nb_c < gs and (nb_r, nb_c) not in obstacles:
            nc = config.cell_colors[nb_r][nb_c]
            if nc > best_color:
                best_color = nc
                best_pos = (nb_r, nb_c)
    if best_pos:
        ctx["brightest_pos"] = list(best_pos)
        ctx["brightest_color"] = best_color

    # flee direction
    if ns:
        flee_dr = pr - ns.row
        flee_dc = pc - ns.col
        if abs(flee_dr) >= abs(flee_dc):
            step = (1 if flee_dr > 0 else -1, 0)
        else:
            step = (0, 1 if flee_dc > 0 else -1)
        ctx["flee_step"] = list(step)

    return ctx


def enrich_record(record: dict, data_dir: str) -> dict:
    """Add F4 context to a single training record."""
    env_id = record["env_id"]
    family = record["family"]

    # Load env config
    parts = env_id.split("_")
    split_name = parts[1]  # "train" or "dev"
    env_num = parts[2]     # "000"
    env_path = Path(data_dir) / f"family{family}" / split_name / f"env_{env_num}.json"
    if not env_path.exists():
        return record

    config = EnvConfig.from_file(str(env_path))

    # Compute F4 context at the action's starting position
    pos = record["prev_state"]["pos"]
    f4_ctx = compute_f4_context(pos[0], pos[1], config)

    # Add to record
    enriched = dict(record)
    enriched["f4_context"] = f4_ctx
    enriched["action_type"] = config.action_type

    return enriched


def main():
    parser = argparse.ArgumentParser(description="Enrich training data with F4 context")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/envs")
    args = parser.parse_args()

    print(f"Loading records from {args.input}...")
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Enriching {len(records)} records with F4 context...")
    enriched = []
    for i, rec in enumerate(records):
        enriched.append(enrich_record(rec, args.data_dir))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(records)}")

    print(f"Writing {len(enriched)} records to {args.output}...")
    with open(args.output, "w") as f:
        for rec in enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print distribution
    has_f4 = sum(1 for r in enriched if r.get("f4_context", {}).get("cell_color", -1) >= 0)
    print(f"Records with F4 context: {has_f4}/{len(enriched)}")
    print("Done!")


if __name__ == "__main__":
    main()
