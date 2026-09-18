#!/usr/bin/env python3
"""Play AlienBody interactively in the terminal.

Usage:
    python scripts/play.py                # random F1 Tier-S
    python scripts/play.py --family 4     # F4 (Relational) with colorful terrain
    python scripts/play.py --family 3 --grid 16   # F3 16x16
    python scripts/play.py --save-gif     # save episode as GIF when done
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.core import AlienBodyEnv
from alienbody.env.generator import generate_env
from alienbody.env.renderer import render_text, render_episode_gif
from alienbody.env.grid import Phase


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def play(family: int, grid_size: int, seed: int, save_gif: bool):
    config = generate_env(family=family, idx=0, split="play", seed=seed, grid_size=grid_size)
    env = AlienBodyEnv(config, render_mode="text")
    obs, info = env.reset()

    actions_taken = []

    print(f"\n{'='*50}")
    print(f"  ALIENBODY — Family {family} ({config.action_type})")
    print(f"  Grid: {grid_size}x{grid_size} | Actions: {config.n_actions}")
    print(f"  Phase 1 budget: {config.phase1_budget} | Max steps: {config.max_total_steps}")
    print(f"{'='*50}")
    print(f"\n  You have {config.n_actions} unlabeled buttons (Action 0-{config.n_actions-1}).")
    print(f"  Phase 1: Press buttons to learn what they do.")
    print(f"  Phase 2: A target appears — navigate to it!")
    print(f"\n  Controls:")
    print(f"    0-{config.n_actions-1}  = press that action button")
    print(f"    d      = done exploring (end Phase 1 early)")
    print(f"    q      = quit")
    input(f"\n  Press Enter to start...")

    step = 0
    while not env.done:
        clear_screen()
        show_target = (env.state.phase == Phase.EXECUTION)
        phase_str = "PHASE 1: CALIBRATION" if not show_target else "PHASE 2: NAVIGATE TO TARGET"

        print(f"\n  {phase_str}  |  Step {step}/{config.max_total_steps}")
        print(f"  {'─'*46}")
        print()
        print(render_text(env.state, config, show_target=show_target))
        print()

        if step > 0 and obs.get("feedback"):
            print(f"  Last: {obs['feedback']}")
        print()

        # Input
        prompt = f"  Action [0-{config.n_actions-1}"
        if env.state.phase == Phase.CALIBRATION:
            prompt += ", d=done"
        prompt += ", q=quit]: "

        try:
            raw = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            return

        if raw == "q":
            print("\n  Quit.")
            return
        if raw == "d" and env.state.phase == Phase.CALIBRATION:
            obs = env.start_phase2()
            info = env._get_info()
            print("\n  >>> Phase 2 started! Target revealed. <<<")
            continue
        try:
            action = int(raw)
            if action < 0 or action >= config.n_actions:
                continue
        except ValueError:
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        actions_taken.append(action)
        step += 1

    # Episode done
    clear_screen()
    traj = env.get_trajectory()
    success = traj["success"]

    print(f"\n  {'='*50}")
    if success:
        print(f"  ★ SUCCESS! ★")
    else:
        print(f"  ✗ FAILED (ran out of steps)")
    print(f"  {'='*50}")
    print(f"  Total steps: {traj['total_steps']}")
    print(f"  Phase 1: {traj['phase1_steps']} steps")
    print(f"  Phase 2: {traj['phase2_steps']} steps")
    print(f"  Ground truth: {list(config.action_mapping)}")
    print()

    if save_gif and actions_taken:
        gif_path = f"data/play_f{family}_{'win' if success else 'lose'}.gif"
        n = render_episode_gif(config, actions_taken, gif_path, scale=2, frame_duration=400)
        print(f"  GIF saved: {gif_path} ({n} frames)")


def main():
    parser = argparse.ArgumentParser(description="Play AlienBody interactively")
    parser.add_argument("--family", type=int, default=1, help="Family 1-6")
    parser.add_argument("--grid", type=int, default=8, help="Grid size (8, 16, 24)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-gif", action="store_true", help="Save episode as GIF")
    args = parser.parse_args()

    play(family=args.family, grid_size=args.grid, seed=args.seed, save_gif=args.save_gif)


if __name__ == "__main__":
    main()
