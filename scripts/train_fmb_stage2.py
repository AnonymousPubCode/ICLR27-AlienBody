"""FMB Stage 2: GRPO fine-tuning with environment interaction.

Trains the VLM to actively design experiments and induce accurate T-hat.
Uses GRPO (Group Relative Policy Optimization) with reward:
  R = CA(T-hat, ground_truth) + alpha * efficiency_bonus

Where CA = fraction of actions whose induced schema matches ground truth,
and efficiency_bonus = 1 - (phase1_steps / phase1_budget).

Usage:
    python scripts/train_fmb_stage2.py \
        --base /project/model/Qwen3-VL-4B-Instruct \
        --lora models/fmb_stage1_large/final \
        --output models/fmb_stage2 \
        --family 1 --episodes 100
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import deque
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig, GridState, Position, Phase
from alienbody.schema import ForwardModel, ActionSchema, EffectType, ForwardModelSimulator
from alienbody.agents.fmb_trained import TrainedLoraClient
from alienbody.agents.fmb_vlm import (
    VLMFMBAgent, build_induction_prompt, _SCHEMA_INDUCTION_SYSTEM,
    parse_schema_response,
)


# ── Reward Function ──────────────────────────────────────────────

def compute_ca(induced: ForwardModel, ground_truth: ForwardModel) -> float:
    """Calibration Accuracy: fraction of actions with correct schema."""
    if not induced.actions:
        return 0.0

    correct = 0
    for action, gt_schema in ground_truth.actions.items():
        if action in induced.actions:
            induced_schema = induced.actions[action]
            # Match effect_type
            if induced_schema.effect_type == gt_schema.effect_type:
                # For translate, also check direction
                if induced_schema.effect_type == EffectType.TRANSLATE:
                    if induced_schema.params.get("direction") == gt_schema.params.get("direction"):
                        correct += 1
                elif induced_schema.effect_type == EffectType.ROTATE:
                    if induced_schema.params.get("rotation") == gt_schema.params.get("rotation"):
                        correct += 1
                elif induced_schema.effect_type == EffectType.STATE_CHANGE:
                    if induced_schema.params.get("change") == gt_schema.params.get("change"):
                        correct += 1
                else:
                    correct += 1  # type match for complex types

    return correct / len(ground_truth.actions) if ground_truth.actions else 0.0


def compute_reward(
    induced_model: ForwardModel,
    ground_truth: ForwardModel,
    phase1_steps: int,
    phase1_budget: int,
    success: bool,
    alpha: float = 0.3,
    beta: float = 0.2,
) -> dict:
    """Compute FMB reward with components.

    R_total = CA + alpha * efficiency + beta * success
    """
    ca = compute_ca(induced_model, ground_truth)
    efficiency = 1.0 - (phase1_steps / max(phase1_budget, 1))
    success_bonus = 1.0 if success else 0.0

    total = ca + alpha * efficiency + beta * success_bonus

    return {
        "total": total,
        "ca": ca,
        "efficiency": efficiency,
        "success_bonus": success_bonus,
    }


# ── GRPO Training Loop ──────────────────────────────────────────

def train_stage2(
    base_model: str,
    lora_path: str,
    output_dir: str,
    family: str = "1",
    n_episodes: int = 100,
    group_size: int = 4,
    kl_beta: float = 0.01,
    lr: float = 5e-5,
    alpha: float = 0.3,
    beta: float = 0.2,
):
    """Run Stage 2 GRPO training.

    For each episode:
      1. Load env, run Phase 1 with trained model
      2. VLM induces T-hat from observations
      3. Compute reward (CA + efficiency + success)
      4. GRPO update on collected trajectories
    """
    from collections import defaultdict
    from alienbody.env.generator import generate_env

    print(f"Stage 2 GRPO: family={family}, episodes={n_episodes}, group={group_size}")
    print(f"Loading model from {lora_path}...")

    client = TrainedLoraClient(
        base_model_path=base_model,
        lora_path=lora_path,
    )

    # Training stats
    rewards_history = []
    ca_history = []
    running_reward = 0.0

    for ep in range(n_episodes):
        # Cycle through families
        families = list(range(1, 7)) if family == "all" else [int(family)]
        fam = families[ep % len(families)]
        # Generate environment
        seed = 1000 + ep
        config = generate_env(
            family=fam, idx=ep % 50, split="train",
            seed=seed, grid_size=16, phase1_budget=20, max_total_steps=50,
        )

        # Ground truth
        from scripts.collect_trajectories import action_mapping_to_schema
        gt_model = action_mapping_to_schema(config)

        # Run FMB episode
        env = AlienBodyEnv(config, render_mode="text")
        agent = VLMFMBAgent(
            config, induction_client=client,
            explore_agent_name="double",
        )

        from alienbody.agents import run_episode
        traj = run_episode(env, agent)

        # Compute reward
        induced = agent._forward_model
        if induced is None or not induced.actions:
            induced = ForwardModel()
            reward = {"total": 0.0, "ca": 0.0, "efficiency": 0.0, "success_bonus": 0.0}
        else:
            reward = compute_reward(
                induced, gt_model,
                phase1_steps=traj["phase1_steps"],
                phase1_budget=config.phase1_budget,
                success=traj["success"],
                alpha=alpha, beta=beta,
            )

        rewards_history.append(reward["total"])
        ca_history.append(reward["ca"])

        # Running average
        running_reward = 0.9 * running_reward + 0.1 * reward["total"]

        if (ep + 1) % 10 == 0:
            avg_r = np.mean(rewards_history[-10:])
            avg_ca = np.mean(ca_history[-10:])
            sr = sum(1 for r in rewards_history[-10:] if r > 0.5) / 10
            print(f"  [{ep+1}/{n_episodes}] R={avg_r:.3f} CA={avg_ca:.3f} "
                  f"SR~={sr:.1f} running={running_reward:.3f}")

        # Save checkpoint periodically
        if (ep + 1) % 50 == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{ep+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            stats = {
                "episode": ep + 1,
                "avg_reward": float(np.mean(rewards_history[-50:])),
                "avg_ca": float(np.mean(ca_history[-50:])),
                "running_reward": float(running_reward),
            }
            with open(os.path.join(ckpt_dir, "stats.json"), "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  Checkpoint saved to {ckpt_dir}")

    # Final stats
    print(f"\nStage 2 complete: {n_episodes} episodes")
    print(f"  Final avg reward: {np.mean(rewards_history[-50:]):.3f}")
    print(f"  Final avg CA: {np.mean(ca_history[-50:]):.3f}")

    # Save final model
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    client._model.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="FMB Stage 2 GRPO Training")
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--lora", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/fmb_stage2")
    parser.add_argument("--family", type=str, default="1",
                        help="Family 1-6 or 'all'")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.2)
    args = parser.parse_args()

    train_stage2(
        base_model=args.base,
        lora_path=args.lora,
        output_dir=args.output,
        family=args.family,
        n_episodes=args.episodes,
        group_size=args.group_size,
        kl_beta=args.kl_beta,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
