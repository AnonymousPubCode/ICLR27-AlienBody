#!/usr/bin/env python3
"""AlienBody demo: generate an environment, run agents, report metrics.

Usage:
    python scripts/demo.py
    python scripts/demo.py --family 2 --visual
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.core import AlienBodyEnv
from alienbody.env.generator import generate_env, validate_solvable
from alienbody.env.renderer import render_text
from alienbody.agents import RandomAgent, OracleAgent, DONE_EXPLORING
from alienbody.env.grid import Phase
from alienbody.eval import compute_all_metrics, aggregate_metrics


def run_episode(env: AlienBodyEnv, agent, verbose: bool = False) -> dict:
    """Run a full episode: Phase 1 (explore) → Phase 2 (navigate)."""
    agent.reset()
    obs, info = env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Environment: {env.config.env_id}")
        print(f"Family: {env.config.family} | Type: {env.config.action_type}")
        print(f"GT Mapping: {list(env.config.action_mapping)}")
        print(f"Agent start: {env.config.agent_start} → Target: {env.config.target_pos}")
        print(f"{'='*60}")

    step = 0
    while not env.done:
        action = agent.act(obs, info)

        # Handle DONE_EXPLORING signal
        if action == DONE_EXPLORING:
            if env.state.phase == Phase.CALIBRATION:
                obs = env.start_phase2()
                info = env._get_info()
                if verbose:
                    print(f"\n  --- Agent ended Phase 1 early. Target at {env.config.target_pos} ---\n")
                continue
            else:
                action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if verbose and step <= 10:  # Show first 10 steps
            phase_str = "P1" if info["phase"] == 1 else "P2"
            pos = info["agent_pos"]
            fb = obs.get("feedback", "")
            print(f"  [{phase_str}] Step {step}: Action {action} → {fb}")

        # Auto-transition to Phase 2 after Phase 1 budget
        if info["phase"] == 2 and env.state.phase2_steps == 0:
            if verbose:
                print(f"\n  --- Phase 2: Target revealed at {env.config.target_pos} ---\n")

    traj = env.get_trajectory()

    if verbose:
        status = "SUCCESS" if traj["success"] else "FAILED"
        print(f"\n  Result: {status} in {traj['total_steps']} steps "
              f"(P1={traj['phase1_steps']}, P2={traj['phase2_steps']})")

    return traj


def demo_family(family: int, n_envs: int = 5, verbose: bool = True):
    """Demo a family: generate environments, run agents, compare."""
    print(f"\n{'#'*60}")
    print(f"  FAMILY {family} DEMO")
    print(f"{'#'*60}")

    random_trajs = []
    oracle_trajs = []

    for i in range(n_envs):
        config = generate_env(family=family, idx=i, split="demo", seed=family * 1000 + i)

        # Validate solvability
        sol = validate_solvable(config)
        if not sol["solvable"]:
            print(f"  [SKIP] env_{i:03d} not solvable")
            continue

        if verbose and i == 0:
            # Show the first environment in detail
            print(f"\n  Oracle solution: {sol['optimal_steps']} steps, "
                  f"path: {sol['optimal_path'][:10]}...")

        env = AlienBodyEnv(config, render_mode="text")

        # Random agent
        random_agent = RandomAgent(n_actions=4, seed=i)
        traj = run_episode(env, random_agent, verbose=(verbose and i == 0))
        random_trajs.append(compute_all_metrics(traj))

        # Oracle agent
        oracle_agent = OracleAgent(config)
        traj = run_episode(env, oracle_agent, verbose=(verbose and i == 0))
        oracle_trajs.append(compute_all_metrics(traj))

    # Aggregate
    print(f"\n{'─'*60}")
    print(f"  RESULTS (Family {family}, {n_envs} environments)")
    print(f"{'─'*60}")

    for name, trajs in [("Random", random_trajs), ("Oracle", oracle_trajs)]:
        if not trajs:
            continue
        agg = aggregate_metrics(trajs)
        print(f"\n  {name:12s}: CE={agg['ce_mean']:.3f}  SR={agg['sr_pct']:.1f}%  "
              f"EC={agg['ec_mean']:.2f}  CA={agg['ca_mean']:.2f}  "
              f"P1={agg['p1_mean']:.1f}  P2={agg['p2_mean']:.1f}")


def demo_text_rendering(family: int = 1):
    """Show what the text observation looks like."""
    config = generate_env(family=family, idx=0, split="demo", seed=9999)
    env = AlienBodyEnv(config, render_mode="text")
    obs, info = env.reset()

    print(f"\n{'='*60}")
    print("  TEXT OBSERVATION (Phase 1 — target hidden)")
    print(f"{'='*60}")
    print(obs["text"])

    # Take a few actions
    for action in [0, 1, 2, 3]:
        obs, _, _, _, _ = env.step(action)
        print(f"\n  → Action {action}: {obs.get('feedback', '')}")

    # Transition to Phase 2
    obs = env.start_phase2()
    print(f"\n{'='*60}")
    print("  TEXT OBSERVATION (Phase 2 — target visible)")
    print(f"{'='*60}")
    print(obs["text"])


def demo_image_rendering(family: int = 1):
    """Generate and save a grid image."""
    config = generate_env(family=family, idx=0, split="demo", seed=9999)
    env = AlienBodyEnv(config, render_mode="image")
    obs, info = env.reset()

    img = obs["image"]
    print(f"\n  Image shape: {img.shape}, dtype: {img.dtype}")

    try:
        from PIL import Image
        pil_img = Image.fromarray(img)
        # Scale up for visibility
        pil_img = pil_img.resize((512, 512), Image.NEAREST)
        out_path = Path(__file__).parent.parent / "data" / "demo_grid.png"
        pil_img.save(str(out_path))
        print(f"  Saved to: {out_path}")
    except ImportError:
        print("  (PIL not available — skipping image save)")


def main():
    parser = argparse.ArgumentParser(description="AlienBody Demo")
    parser.add_argument("--family", type=int, default=0,
                        help="Family to demo (1-4, 0=all)")
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--visual", action="store_true",
                        help="Also generate image rendering")
    parser.add_argument("--text-demo", action="store_true",
                        help="Show text observation demo")
    args = parser.parse_args()

    print("=" * 60)
    print("  AlienBody Benchmark — Demo")
    print("=" * 60)

    if args.text_demo:
        demo_text_rendering()
        return

    if args.visual:
        demo_image_rendering(family=args.family or 1)

    families = [args.family] if args.family else [1, 2, 3, 4, 5, 6]
    for f in families:
        demo_family(f, n_envs=args.n_envs)


if __name__ == "__main__":
    main()
