#!/usr/bin/env python3
"""Aggregate + paired-bootstrap CI for the Crafter name-prior experiment (P-S1).

Reads summary_*.json and per-run jsons from one or more output dirs, reports
per-condition metrics and paired bootstrap CIs over shared seeds.

Usage: python scripts/crafter_analyze.py [results_dir ...]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

CONDS = ["named", "category", "anonymous"]


def load_runs(dirs) -> dict:
    """Per-condition list of run records, one per seed (newest file wins).

    Accepts one or more directories; runs are merged by (condition, seed),
    keeping the file with the newest timestamp in its name.
    """
    if isinstance(dirs, str):
        dirs = [dirs]
    runs = {}
    for c in CONDS:
        by_seed = {}
        for d in dirs:
            fs = glob.glob(os.path.join(d, f"{c}_seed*.json"))
            fs = [f for f in fs if "summary" not in f]
            for f in fs:
                try:
                    seed = int(os.path.basename(f).split("_seed")[1].split("_")[0])
                except (IndexError, ValueError):
                    continue
                # keep the newest file per seed (timestamp in filename)
                if seed not in by_seed or os.path.basename(f) > os.path.basename(by_seed[seed]):
                    by_seed[seed] = f
        recs = []
        for seed in sorted(by_seed):
            data = json.load(open(by_seed[seed]))
            r = data["result"]
            recs.append({
                "seed": seed,
                "achievements": r["achievements"],
                "early": r["early_achievements"],
                "steps": r["steps_run"],
                "died": r["died"],
                "ca": (r.get("probe") or {}).get("ca"),
                "tokens": r["total_tokens"],
                "achievements_list": set(r["achievements_list"]),
            })
        runs[c] = recs
    return runs


def paired_ci(a, b, n_boot=10000, seed=0):
    """Paired bootstrap 95% CI and two-sided p for mean(a - b)."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    dist = d[idx].mean(axis=1)
    ci = (float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5)))
    p = 2.0 * min(float((dist <= 0).mean()), float((dist >= 0).mean()))
    return float(np.mean(d)), ci, min(float(p), 1.0)


def crafter_score(recs) -> float:
    n = len(recs)
    if n == 0:
        return 0.0
    # union of achievements observed across the condition
    all_ach = sorted(set().union(*(r["achievements_list"] for r in recs)))
    if not all_ach:
        return 0.0
    succ = {a: sum(1 for r in recs if a in r["achievements_list"]) / n
            for a in all_ach}
    # official score uses the full 22-achievement denominator
    full = {a: 0.0 for a in [
        "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
        "collect_sapling", "collect_stone", "collect_wood",
        "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
        "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe",
        "make_stone_sword", "make_wood_pickaxe", "make_wood_sword",
        "place_furnace", "place_plant", "place_stone", "place_table",
        "wake_up"]}
    full.update(succ)
    return math.exp(sum(math.log(1 + s) for s in full.values()) / len(full)) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="+",
                    default=[os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "results", "crafter_name_prior_gpt4o")])
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    n_per = {c: len(runs[c]) for c in CONDS}
    print(f"runs per condition: {n_per}\n")

    rows = {}
    for c in CONDS:
        recs = runs[c]
        if not recs:
            print(f"=== {c}: no runs found ===")
            continue
        ach = [r["achievements"] for r in recs]
        early = [r["early"] for r in recs]
        cas = [r["ca"] for r in recs if r["ca"] is not None]
        row = {
            "achievements": f"{np.mean(ach):.1f} ({min(ach)}-{max(ach)})",
            "early": f"{np.mean(early):.1f}",
            "score": f"{crafter_score(recs):.3f}",
            "steps": f"{np.mean([r['steps'] for r in recs]):.0f}",
            "died": f"{sum(r['died'] for r in recs)}/{len(recs)}",
            "probe_ca": f"{np.mean(cas):.2f}" if cas else "n/a",
            "tokens_total": f"{sum(r['tokens'] for r in recs):,}",
        }
        rows[c] = row
        print(f"=== {c} ===")
        for k, v in row.items():
            print(f"  {k:14s} {v}")
        print()

    # paired comparisons over shared seeds
    print("=== paired comparisons (mean delta, 95% CI, p) ===")
    for a, b in [("named", "anonymous"), ("category", "anonymous"),
                 ("named", "category")]:
        seeds_a = {r["seed"] for r in runs[a]}
        seeds_b = {r["seed"] for r in runs[b]}
        shared = sorted(seeds_a & seeds_b)
        if len(shared) < 2:
            print(f"  {a} vs {b}: insufficient shared seeds ({len(shared)})")
            continue
        da = {r["seed"]: r["achievements"] for r in runs[a]}
        db = {r["seed"]: r["achievements"] for r in runs[b]}
        va = [da[s] for s in shared]
        vb = [db[s] for s in shared]
        delta, ci, p = paired_ci(va, vb, n_boot=args.n_boot)
        print(f"  {a} - {b} (achievements, n={len(shared)}): "
              f"{delta:+.1f}  CI [{ci[0]:+.1f}, {ci[1]:+.1f}]  p={p:.3f}")


if __name__ == "__main__":
    main()
