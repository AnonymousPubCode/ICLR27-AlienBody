#!/usr/bin/env python3
"""Aggregate human study data and update HUMAN_BASELINE in eval/__init__.py.

Usage:
    # Aggregate all participant JSONL files
    python scripts/aggregate_human_data.py

    # Aggregate and update eval/__init__.py in-place
    python scripts/aggregate_human_data.py --update-eval

    # Specify custom data directory
    python scripts/aggregate_human_data.py --data-dir data/human_study

Output:
    - Per-family human baseline (total, phase1, phase2 steps)
    - Per-participant summary
    - Optionally updates alienbody/eval/__init__.py HUMAN_BASELINE dict
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.trajectory import load_trajectories
from alienbody.eval import compute_all_metrics


def aggregate_human_data(data_dir: Path) -> dict:
    """Aggregate all participant data into per-family baselines.

    Returns:
        {
            "participants": {pid: {"n_envs": N, "sr_pct": SR, ...}},
            "per_family": {
                1: {"total": avg, "phase1": avg, "phase2": avg, "sr_pct": sr, "n": N},
                ...
            },
            "overall": {"total": avg, "phase1": avg, "phase2": avg, "n_participants": N},
        }
    """
    jsonl_files = sorted(data_dir.glob("participant_*.jsonl"))
    if not jsonl_files:
        print(f"No participant_*.jsonl files found in {data_dir}")
        return {}

    print(f"Found {len(jsonl_files)} participant files")

    # Per-family accumulators
    family_steps: dict[int, list[float]] = defaultdict(list)
    family_p1: dict[int, list[float]] = defaultdict(list)
    family_p2: dict[int, list[float]] = defaultdict(list)
    family_success: dict[int, list[bool]] = defaultdict(list)

    participants = {}

    for jf in jsonl_files:
        pid = jf.stem.replace("participant_", "")
        trajectories = load_trajectories(jf)
        if not trajectories:
            print(f"  {pid}: empty file, skipping")
            continue

        n_success = sum(1 for t in trajectories if t.get("success", False))
        participants[pid] = {
            "n_envs": len(trajectories),
            "n_success": n_success,
            "sr_pct": n_success / len(trajectories) * 100 if trajectories else 0.0,
        }

        for traj in trajectories:
            family = traj.get("family", 1)
            if not traj.get("success", False):
                continue  # Only use successful episodes for baseline
            family_steps[family].append(traj["total_steps"])
            family_p1[family].append(traj["phase1_steps"])
            family_p2[family].append(traj["phase2_steps"])
            family_success[family].append(True)

        print(f"  {pid}: {len(trajectories)} envs, {n_success} success "
              f"({participants[pid]['sr_pct']:.0f}%)")

    # Compute per-family averages
    per_family = {}
    all_steps, all_p1, all_p2 = [], [], []
    for family in sorted(family_steps.keys()):
        steps = family_steps[family]
        p1 = family_p1[family]
        p2 = family_p2[family]
        n_success = len(steps)
        n_total = len(family_success.get(family, []))

        per_family[family] = {
            "total": sum(steps) / len(steps) if steps else 0.0,
            "phase1": sum(p1) / len(p1) if p1 else 0.0,
            "phase2": sum(p2) / len(p2) if p2 else 0.0,
            "sr_pct": n_success / n_total * 100 if n_total else 0.0,
            "n_success": n_success,
            "n_total": n_total,
        }
        all_steps.extend(steps)
        all_p1.extend(p1)
        all_p2.extend(p2)

    overall = {
        "total": sum(all_steps) / len(all_steps) if all_steps else 0.0,
        "phase1": sum(all_p1) / len(all_p1) if all_p1 else 0.0,
        "phase2": sum(all_p2) / len(all_p2) if all_p2 else 0.0,
        "n_participants": len(participants),
        "n_successful_episodes": len(all_steps),
    }

    return {
        "participants": participants,
        "per_family": per_family,
        "overall": overall,
    }


def print_summary(results: dict):
    """Pretty-print aggregation results."""
    if not results:
        return

    pf = results["per_family"]
    ov = results["overall"]

    print(f"\n{'='*70}")
    print(f"  HUMAN BASELINE SUMMARY")
    print(f"  Participants: {ov['n_participants']} | "
          f"Successful episodes: {ov['n_successful_episodes']}")
    print(f"{'='*70}")
    print(f"  {'Family':<20} {'Total':>8} {'P1':>8} {'P2':>8} {'SR%':>8} {'N':>6}")
    print(f"  {'-'*58}")

    family_names = {
        1: "F1 Remapped", 2: "F2 Directional", 3: "F3 State-Dep",
        4: "F4 Relational", 5: "F5 Composite", 6: "F6 Temporal",
    }
    for family in sorted(pf.keys()):
        f = pf[family]
        name = family_names.get(family, f"F{family}")
        print(f"  {name:<20} {f['total']:>8.1f} {f['phase1']:>8.1f} "
              f"{f['phase2']:>8.1f} {f['sr_pct']:>7.1f}% {f['n_success']:>5d}")

    print(f"  {'-'*58}")
    print(f"  {'OVERALL':<20} {ov['total']:>8.1f} {ov['phase1']:>8.1f} "
          f"{ov['phase2']:>8.1f}")
    print()

    # Python dict format for copy-paste into eval/__init__.py
    print("  # ── Copy into alienbody/eval/__init__.py ──")
    print("  HUMAN_BASELINE = {")
    for family in sorted(pf.keys()):
        f = pf[family]
        print(f"      {family}: {{\"total\": {f['total']:.1f}, "
              f"\"phase1\": {f['phase1']:.1f}, "
              f"\"phase2\": {f['phase2']:.1f}}},"
              f"  # {family_names.get(family, f'F{family}')}")
    print("  }")


def update_eval_module(results: dict, eval_path: Path):
    """Update HUMAN_BASELINE dict in eval/__init__.py."""
    if not results or not eval_path.exists():
        print(f"Error: {eval_path} not found")
        return

    pf = results["per_family"]
    ov = results["overall"]

    content = eval_path.read_text()

    # Build new HUMAN_BASELINE block
    lines = ["HUMAN_BASELINE = {"]
    family_names = {
        1: "Remapped (easiest)",
        2: "Directional",
        3: "State-dependent",
        4: "Relational",
        5: "Composite",
        6: "Temporal",
    }
    for family in sorted(pf.keys()):
        f = pf[family]
        name = family_names.get(family, f"F{family}")
        lines.append(
            f"    {family}: {{\"total\": {f['total']:.1f}, "
            f"\"phase1\": {f['phase1']:.1f}, "
            f"\"phase2\": {f['phase2']:.1f}}},"
            f"   # {name}"
        )
    lines.append("}")

    new_block = "\n".join(lines)

    # Find and replace HUMAN_BASELINE block
    import re
    pattern = r"HUMAN_BASELINE\s*=\s*\{[^}]+\}"
    if re.search(pattern, content):
        new_content = re.sub(pattern, new_block, content)
        eval_path.write_text(new_content)
    else:
        print("Warning: Could not find HUMAN_BASELINE block to replace")

    # Update HUMAN_AVG_DEFAULT
    new_default = f"HUMAN_AVG_DEFAULT = {ov['total']:.1f}  # fallback"
    content = eval_path.read_text()
    content = re.sub(
        r"HUMAN_AVG_DEFAULT\s*=\s*[\d.]+.*",
        new_default,
        content,
    )
    eval_path.write_text(content)

    print(f"Updated {eval_path}")
    print(f"  HUMAN_BASELINE ← {len(pf)} families with real data")
    print(f"  HUMAN_AVG_DEFAULT ← {ov['total']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate human study data for AlienBody"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/human_study",
        help="Directory containing participant_*.jsonl files",
    )
    parser.add_argument(
        "--update-eval", action="store_true",
        help="Update HUMAN_BASELINE in alienbody/eval/__init__.py",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save summary JSON to file",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / args.data_dir
    eval_path = Path(__file__).parent.parent / "alienbody" / "eval" / "__init__.py"

    results = aggregate_human_data(data_dir)
    if not results:
        print("\nNo data found. Run the human study first:")
        print("  streamlit run scripts/human_study.py")
        return

    print_summary(results)

    if args.update_eval:
        update_eval_module(results, eval_path)

    if args.output:
        output_path = Path(__file__).parent.parent / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
