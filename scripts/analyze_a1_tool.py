#!/usr/bin/env python3
"""A1 clean-planning analysis: L3 no-tool vs L3 + next_state simulator oracle.

Loads trajectories from the A1 control runs (results/a1_control_*) and tool
runs (results/a1_tool_*), and reports:

  - per-condition SR (the headline comparison)
  - tool-usage statistics: calls/episode, distinct positions queried,
    whether the target was ever reached *in simulation* (search success
    independent of execution)
  - oracle-plan agreement for tool runs (reuse OracleAgent BFS)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.grid import EnvConfig, Position
from alienbody.trajectory import load_trajectories


def summarize(path: Path) -> dict | None:
    if not path.exists():
        return None
    recs = load_trajectories(path)
    if not recs:
        return None
    n = len(recs)
    succ = sum(1 for r in recs if r.get("success"))
    n_tool = [r.get("n_tool_calls", 0) for r in recs]
    return {
        "n": n,
        "sr": 100.0 * succ / n,
        "success": succ,
        "tool_calls_mean": sum(n_tool) / n,
        "tool_calls_median": sorted(n_tool)[n // 2],
        "tool_calls_total": sum(n_tool),
        "recs": recs,
    }


def tool_search_stats(recs: list[dict], configs_dir: Path) -> dict:
    """Per-episode tool-search quality stats."""
    n_reach_target_sim = 0
    n_target_queried = 0
    distinct_pos = []
    for r in recs:
        cfg = EnvConfig.from_file(
            str(configs_dir / f"family{r['family']}" / "test" / f"env_{r['env_id'].split('_', 2)[-1]}.json")
        )
        target = tuple(cfg.target_pos)
        tl = r.get("tool_log", [])
        if not tl:
            distinct_pos.append(0)
            continue
        positions = {(q[0], q[1]) for q in (t["query"] for t in tl)}
        results = {(q[0], q[1]) for q in (t["result"] for t in tl)}
        distinct_pos.append(len(positions | results))
        if target in results:
            n_reach_target_sim += 1
        if target in positions:
            n_target_queried += 1
    n = len(recs)
    return {
        "n": n,
        "reach_target_in_sim": n_reach_target_sim,
        "target_queried": n_target_queried,
        "distinct_pos_mean": sum(distinct_pos) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data-dir", default="data/envs")
    args = ap.parse_args()

    results = Path(__file__).parent.parent / args.results
    data_dir = Path(__file__).parent.parent / args.data_dir

    models = [
        ("gpt4o", "fuxi_gpt-4o"),
        ("gemini", "fuxi_gemini-3-pro-preview"),
        ("dsv4", "fuxi_dsv4-lh"),
    ]

    print(f"{'model':8s} | {'cond':8s} | {'n':3s} | {'SR%':6s} | {'tool/ep':8s} | tool-med")
    print("-" * 60)
    for label, dirname in models:
        ctrl = summarize(results / f"a1c_control_{label}" / dirname / "family4_test.jsonl")
        tool = summarize(results / f"a1c_tool_{label}" / dirname / "family4_test.jsonl")
        for cond, s in (("no-tool", ctrl), ("tool", tool)):
            if s is None:
                print(f"{label:8s} | {cond:8s} |  -  |    -   |    -    | -")
                continue
            print(f"{label:8s} | {cond:8s} | {s['n']:3d} | "
                  f"{s['sr']:5.1f} | {s['tool_calls_mean']:6.1f} | {s['tool_calls_median']}")
        if tool:
            st = tool_search_stats(tool["recs"], data_dir)
            print(f"    tool search: reach-target-in-sim {st['reach_target_in_sim']}/{st['n']}, "
                  f"target-queried {st['target_queried']}/{st['n']}, "
                  f"distinct-pos/ep {st['distinct_pos_mean']:.1f}")
    print()

    # Aggregate JSON for the paper
    out = {}
    for label, dirname in models:
        ctrl = summarize(results / f"a1c_control_{label}" / dirname / "family4_test.jsonl")
        tool = summarize(results / f"a1c_tool_{label}" / dirname / "family4_test.jsonl")
        out[label] = {
            "no_tool": {k: v for k, v in (ctrl or {}).items() if k != "recs"},
            "tool": {k: v for k, v in (tool or {}).items() if k != "recs"},
        }
        if tool:
            out[label]["tool"]["search"] = tool_search_stats(tool["recs"], data_dir)
    out_path = results / "a1_tool_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
