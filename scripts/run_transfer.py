#!/usr/bin/env python3
"""Cross-episode transfer experiment.

Tests whether agents improve at calibration across consecutive episodes
from the same family (learning-to-learn).

Usage:
    python scripts/run_transfer.py --agent gpt-4o --family 1 --n-episodes 10
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.core import AlienBodyEnv
from alienbody.env.generator import generate_env, validate_solvable
from alienbody.agents import run_episode
from alienbody.eval import compute_all_metrics
from alienbody.trajectory import TrajectoryWriter


def run_transfer_experiment(
    agent_name: str,
    family: int,
    n_episodes: int = 10,
    modality: str = "image",
    output_dir: str = "results/transfer",
):
    """Run n consecutive episodes, tracking CE improvement."""
    from scripts.run_eval import create_agent

    output_path = Path(output_dir) / f"{agent_name}_family{family}_transfer.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate n solvable environments from same family
    configs = []
    attempt = 0
    while len(configs) < n_episodes and attempt < n_episodes * 10:
        config = generate_env(family=family, idx=attempt, split="dev", seed=family * 9000 + attempt)
        sol = validate_solvable(config)
        if sol["solvable"]:
            configs.append(config)
        attempt += 1

    print(f"Transfer experiment: {agent_name} on Family {family}, {n_episodes} episodes")
    print(f"{'─'*50}")

    episode_metrics = []
    with TrajectoryWriter(output_path) as writer:
        for i, config in enumerate(configs):
            env = AlienBodyEnv(config, render_mode="both")
            agent = create_agent(agent_name, config, modality=modality)

            traj = run_episode(env, agent)
            metrics = compute_all_metrics(traj)
            traj["metrics"] = metrics
            traj["transfer_episode"] = i
            writer.write(traj, agent_name=agent_name, experiment="transfer")
            episode_metrics.append(metrics)

            status = "OK" if metrics["success"] else "FAIL"
            print(f"  Episode {i+1:2d}/{n_episodes}: CE={metrics['ce']:.3f} "
                  f"SR={status} P1={metrics['phase1_steps']} P2={metrics['phase2_steps']}")

    # Summary: trend
    print(f"\n{'─'*50}")
    print(f"  CE trend: {' → '.join(f'{m['ce']:.2f}' for m in episode_metrics)}")
    first_half = episode_metrics[:n_episodes // 2]
    second_half = episode_metrics[n_episodes // 2:]
    ce_first = sum(m["ce"] for m in first_half) / len(first_half) if first_half else 0
    ce_second = sum(m["ce"] for m in second_half) / len(second_half) if second_half else 0
    print(f"  First half avg CE: {ce_first:.3f}")
    print(f"  Second half avg CE: {ce_second:.3f}")
    print(f"  Improvement: {(ce_second - ce_first):.3f} ({(ce_second - ce_first) / max(ce_first, 0.01) * 100:+.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="random")
    parser.add_argument("--family", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--modality", default="image")
    parser.add_argument("--output", default="results/transfer")
    args = parser.parse_args()

    run_transfer_experiment(
        args.agent, args.family, args.n_episodes, args.modality, args.output
    )


if __name__ == "__main__":
    main()
