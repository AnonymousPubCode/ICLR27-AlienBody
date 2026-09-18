#!/usr/bin/env python3
"""Replay GB name-prior runs to recompute metrics (PyBoy is deterministic).

The original runs logged (buttons, response) per step but checkpoint hashes
used a buggy dHash. Replay the same button sequences to recompute:
  - max scroll_x (level progress)
  - checkpoint matches with the corrected 64-bit dHash
  - deaths / min health

Usage:
  python scripts/gb_replay_metrics.py --runs-dir results/gb_name_prior
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
VZG_ROOT = os.environ.get("VZG_ROOT", r"D:\Workspace\VideoGameZero")
sys.path.insert(0, VZG_ROOT)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from gb_name_prior import (RAM, ROM_PATH, BUTTONS, checkpoint_hashes, dh,  # noqa: E402
                           hamming, parse_buttons)


def replay_run(log: list[dict], cp_hashes: dict[int, str], warmup: int = 8) -> dict:
    from shared.emulators.gba.interface import GBAInterface

    env = GBAInterface(render=False)
    ok = env.load_game(ROM_PATH)
    assert ok, f"failed to load {ROM_PATH}"

    for _ in range(warmup):
        env.step({"START": True}, skip_frames=12)

    max_scroll = 0
    max_cp = 0
    deaths = 0
    min_health = 99
    prev_health = None
    cp_hit_at_step = {}

    for entry in log:
        buttons = entry.get("buttons") or parse_buttons(entry.get("response", "")) or ["A"]
        action = {b: True for b in buttons if b in BUTTONS}
        if not action:
            action = {"A": True}
        env.step(action, skip_frames=10)

        mem = env.read_memory_values(RAM)
        if mem.get("scroll_x", 0) is not None:
            max_scroll = max(max_scroll, mem["scroll_x"])
        health = mem.get("health")
        if health is not None:
            min_health = min(min_health, health)
            if prev_health is not None and health < prev_health and health < 6:
                deaths += 1
            prev_health = health
        if mem.get("game_state") in (0x06, 6):
            deaths += 1
            prev_health = None

        screen = env.get_screen()
        h = dh(screen)
        for idx, ref in cp_hashes.items():
            if hamming(h, ref) <= 12 and idx > max_cp:
                max_cp = idx
                cp_hit_at_step[idx] = entry.get("step")

    env.close()
    return {
        "max_scroll_x": max_scroll,
        "max_checkpoint": max_cp,
        "min_health": min_health,
        "deaths": deaths,
        "checkpoint_hits": cp_hit_at_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "..", "results", "gb_name_prior"))
    ap.add_argument("--checkpoint-dir",
                    default=os.path.join(VZG_ROOT, "rl", "configs",
                                         "kirby_dream_land_dx", "checkpoints"))
    args = ap.parse_args()

    cp_hashes = checkpoint_hashes(args.checkpoint_dir)
    print(f"checkpoints: {sorted(cp_hashes)}")

    for f in sorted(glob(os.path.join(args.runs_dir, "*_*.json"))):
        if "summary" in f:
            continue
        data = json.load(open(f))
        cond = data["result"]["condition"]
        r = replay_run(data["log"], cp_hashes)
        data["result_replayed"] = r
        with open(f, "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"{os.path.basename(f):<35} {cond:>10}: "
              f"scroll={r['max_scroll_x']:>3} cp={r['max_checkpoint']:>3} "
              f"deaths={r['deaths']} hits={r['checkpoint_hits']}")

    # Recompute summaries
    for cond in ["named", "category", "anonymous"]:
        runs = []
        for f in sorted(glob(os.path.join(args.runs_dir, f"{cond}_*.json"))):
            if "summary" in f:
                continue
            data = json.load(open(f))
            runs.append(data["result_replayed"])
        if not runs:
            continue
        agg = {
            "condition": cond,
            "mean_scroll_x": sum(r["max_scroll_x"] for r in runs) / len(runs),
            "mean_max_checkpoint": sum(r["max_checkpoint"] for r in runs) / len(runs),
            "mean_deaths": sum(r["deaths"] for r in runs) / len(runs),
            "runs": runs,
        }
        with open(os.path.join(args.runs_dir, f"summary_{cond}.json"), "w") as fh:
            json.dump(agg, fh, indent=1)
        print(f"{cond}: mean_scroll={agg['mean_scroll_x']:.1f} "
              f"mean_cp={agg['mean_max_checkpoint']:.1f} "
              f"mean_deaths={agg['mean_deaths']:.1f}")


if __name__ == "__main__":
    main()
