#!/usr/bin/env python3
"""Master experiment orchestrator for AlienBody.

Usage:
    python scripts/run_all_experiments.py --experiment rq1
    python scripts/run_all_experiments.py --experiment all
    python scripts/run_all_experiments.py --experiment rq2 --dry-run
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SCRIPTS_DIR = Path(__file__).parent


# ── Experiment Definitions ─────────────────────────────────────────

EXPERIMENTS = {
    "rq1": {
        "name": "Main Results (Table 2)",
        "description": "Test all models on full test set",
        "runs": [
            # Control agents
            {"agent": "random", "family": "all", "split": "test"},
            {"agent": "oracle", "family": "all", "split": "test"},
            # Frontier VLMs (image modality)
            {"agent": "gpt-4o", "family": "all", "split": "test", "modality": "image"},
            {"agent": "claude-sonnet-4-20250514", "family": "all", "split": "test", "modality": "image"},
            {"agent": "gemini-2.5-pro", "family": "all", "split": "test", "modality": "image"},
            # Text-only LLMs
            {"agent": "gpt-4o", "family": "all", "split": "test", "modality": "text"},
            {"agent": "claude-sonnet-4-20250514", "family": "all", "split": "test", "modality": "text"},
        ],
    },
    "rq2": {
        "name": "Familiarity Effect (Table 3)",
        "description": "Compare familiar vs novel action types",
        "runs": [
            {"agent": "gpt-4o", "family": "1", "split": "test", "familiar": True},
            {"agent": "gpt-4o", "family": "1", "split": "test", "familiar": False},
            {"agent": "claude-sonnet-4-20250514", "family": "1", "split": "test", "familiar": True},
            {"agent": "claude-sonnet-4-20250514", "family": "1", "split": "test", "familiar": False},
        ],
    },
    "rq4": {
        "name": "Modality Comparison",
        "description": "VLM vs LLM for same model",
        "runs": [
            {"agent": "claude-sonnet-4-20250514", "family": "all", "split": "test", "modality": "image"},
            {"agent": "claude-sonnet-4-20250514", "family": "all", "split": "test", "modality": "text"},
        ],
    },
    "prompt": {
        "name": "Prompt Sensitivity",
        "description": "Compare prompt variants",
        "runs": [
            {"agent": "gpt-4o", "family": "1", "split": "dev", "prompt": "minimal"},
            {"agent": "gpt-4o", "family": "1", "split": "dev", "prompt": "cot"},
            {"agent": "gpt-4o", "family": "1", "split": "dev", "prompt": "explore_first"},
            {"agent": "gpt-4o", "family": "1", "split": "dev", "prompt": "expert_demo"},
        ],
    },
    "framework": {
        "name": "Agent Frameworks",
        "description": "ReAct and Reflexion wrappers",
        "runs": [
            {"agent": "gpt-4o", "family": "all", "split": "test"},
            {"agent": "react+gpt-4o", "family": "all", "split": "test"},
            {"agent": "reflexion+gpt-4o", "family": "all", "split": "test"},
        ],
    },
}


def build_command(run: dict) -> list[str]:
    """Build run_eval.py command from run specification."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "run_eval.py")]
    cmd += ["--agent", run["agent"]]
    cmd += ["--family", str(run.get("family", "all"))]
    cmd += ["--split", run.get("split", "test")]
    cmd += ["--modality", run.get("modality", "image")]
    cmd += ["--prompt", run.get("prompt", "minimal")]
    if run.get("familiar"):
        cmd += ["--familiar"]
    cmd += ["--resume"]
    return cmd


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AlienBody Experiment Orchestrator")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Experiment to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    if args.experiment == "all":
        exps = list(EXPERIMENTS.keys())
    else:
        exps = [args.experiment]

    for exp_name in exps:
        exp = EXPERIMENTS[exp_name]
        print(f"\n{'='*60}")
        print(f"  {exp['name']}")
        print(f"  {exp['description']}")
        print(f"{'='*60}")

        for i, run in enumerate(exp["runs"]):
            cmd = build_command(run)
            print(f"\n  [{i+1}/{len(exp['runs'])}] {' '.join(cmd)}")

            if args.dry_run:
                continue

            try:
                result = subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  ERROR: {e}")
            except KeyboardInterrupt:
                print("\n  Interrupted. Exiting.")
                return

    if not args.dry_run:
        print(f"\n{'='*60}")
        print("  All experiments complete.")
        print(f"  Generate tables: python scripts/generate_tables.py")
        print(f"  Generate figures: python scripts/generate_figures.py")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
