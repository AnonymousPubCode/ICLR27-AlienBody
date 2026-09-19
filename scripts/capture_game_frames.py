#!/usr/bin/env python3
"""Capture gameplay screenshots for the name-prior figure (P-S1 visual evidence).

Deterministically replays recorded trajectories (no API calls):
- Kirby: replays the logged button sequences of a named and an anonymous run
  through PyBoy (same ROM, same warmup, same skip-frames -> same screens).
- Crafter: replays the logged action sequence of named_seed0 and
  anonymous_seed0 through crafter.Env (seeded -> deterministic), rendering
  frames at chosen steps, upscaled 512x512.

Usage: python scripts/capture_game_frames.py
Output: fig/game_captures/{game}_{cond}_step{NNN}.png
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
VZG_ROOT = os.environ.get("VZG_ROOT", r"D:\Workspace\VideoGameZero")
sys.path.insert(0, VZG_ROOT)
sys.path.insert(0, os.path.join(HERE, ".."))

OUT = os.path.join(HERE, "..", "..", "fig", "game_captures")

KIRBY_ROM = os.path.join(VZG_ROOT, "roms", "kirby_dream_land_dx.gb")
CRAFTER_RESULTS = os.path.join(HERE, "..", "results", "crafter_name_prior_gpt4o")
CRAFTER_V2_RESULTS = os.path.join(HERE, "..", "results",
                                  "crafter_name_prior_v2_gpt4o")
KIRBY_RESULTS = os.path.join(HERE, "..", "results", "gb_name_prior_vision")


def save(img: Image.Image, name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    img.save(p)
    print("saved", p)
    return p


def pick_run(pattern: str, best: bool) -> str:
    fs = glob.glob(pattern)
    assert fs, f"no runs for {pattern}"
    if best:
        best_f, best_v = None, -1
        for f in fs:
            r = json.load(open(f))["result"]
            v = r.get("max_scroll_x", r.get("achievements", 0))
            if v > best_v:
                best_f, best_v = f, v
        return best_f
    # median-ish: middle file by metric
    scored = sorted((json.load(open(f))["result"].get("max_scroll_x", 0), f)
                    for f in fs)
    return scored[len(scored) // 2][1]


def pick_v2_seed(cond: str) -> tuple[int, str, int]:
    """Symmetric per-condition rule: the seed at the condition's median
    achievement count (ties -> lowest seed).  Returns (seed, path, achievements)."""
    rows = []
    for f in glob.glob(os.path.join(CRAFTER_V2_RESULTS, f"{cond}_seed*_full20.json")):
        r = json.load(open(f))["result"]
        rows.append((r["seed"], f, r["achievements"]))
    assert rows, f"no v2 runs for {cond}"
    rows.sort()
    counts = sorted(v for _, _, v in rows)
    median = counts[(len(counts) - 1) // 2]           # lower median, 20 seeds
    for seed, f, v in rows:
        if v == median:
            return seed, f, v
    raise AssertionError


def capture_crafter_v2():
    """Crafter frames from the v2 (gloss-free, shared-history) protocol.

    Replays each run under its *own* seed (the v1 helper's fixed seed=0 replay
    was a bug) and reports the replayed end state so the caption text can be
    written from the trace rather than from the image.
    """
    import crafter

    for cond in ["named", "anonymous", "category"]:
        seed, path, ach = pick_v2_seed(cond)
        data = json.load(open(path))
        log = [e for e in data["log"] if "delta" in e]
        env = crafter.Env(seed=seed)
        names = list(env.action_names)
        env.reset()
        pos0 = tuple(int(v) for v in env._player.pos)
        info = {"achievements": {}, "inventory": {}}
        acted = 0
        for step, e in enumerate(log):
            idx = e["action_index"] if e["action_index"] is not None else 0
            _, _, done, info = env.step(idx)
            if e["parse_ok"]:
                acted += 1
            if done:
                break
        pos_end = tuple(int(v) for v in info.get("player_pos", env._player.pos))
        ach_end = sorted(n for n, c in info["achievements"].items() if c > 0)
        frame = Image.fromarray(env.render()).resize((512, 512), Image.NEAREST)
        name = f"crafter_v2_{cond}_seed{seed}_step{step:03d}.png"
        save(frame, name)
        print(f"{cond}: seed {seed} ({path.split(os.sep)[-1]}), median-ach {ach}, "
              f"{len(log)} logged steps, {acted} acted, "
              f"pos {pos0}->{pos_end} (tiles moved "
              f"{abs(pos0[0]-pos_end[0])+abs(pos0[1]-pos_end[1])}), "
              f"achievements {ach_end}, "
              f"inventory { {k: v for k, v in info['inventory'].items() if v} }")
    print("crafter v2 done")


def capture_kirby():
    from shared.emulators.gba.interface import GBAInterface as GameBoyInterface

    steps_wanted = {0, 60, 120, 180, 240, 299}
    for cond, best in [("named", True), ("anonymous", False)]:
        f = pick_run(os.path.join(KIRBY_RESULTS, f"{cond}_*.json"), best)
        data = json.load(open(f))
        log = [e for e in data["log"] if "buttons" in e]
        env = GameBoyInterface(render=False)
        assert env.load_game(KIRBY_ROM)
        for _ in range(8):  # same warmup as gb_name_prior.run_episode
            env.step({"START": True}, skip_frames=12)
        for step, e in enumerate(log):
            if step in steps_wanted:
                screen = env.get_screen()
                img = screen.resize((640, 576), Image.NEAREST) \
                    if not isinstance(screen, Image.Image) else screen
                save(img, f"kirby_{cond}_step{step:03d}.png")
            if step >= 299:
                break
            env.step({b: True for b in e["buttons"]}, skip_frames=10)
        env.close()
    print("kirby done")


def capture_crafter():
    import crafter

    steps_wanted = {0, 75, 150, 225, 299}
    for cond in ["named", "anonymous"]:
        # newest file per seed, then the longest run among seeds
        by_seed = {}
        for f in glob.glob(os.path.join(CRAFTER_RESULTS, f"{cond}_seed*.json")):
            try:
                seed = int(os.path.basename(f).split("_seed")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            if seed not in by_seed or f > by_seed[seed]:
                by_seed[seed] = f
        best_f = None
        best_n = -1
        for f in by_seed.values():
            data = json.load(open(f))
            n = len([e for e in data["log"] if "action" in e])
            if n > best_n:
                best_f, best_n = f, n
        print(f"{cond}: using {os.path.basename(best_f)} ({best_n} steps)")
        data = json.load(open(best_f))
        env = crafter.Env(seed=0)
        names = list(env.action_names)
        env.reset()
        log = [e for e in data["log"] if "action" in e]
        achieved = set()
        info = {"achievements": {}, "player_pos": (0, 0)}
        for step, e in enumerate(log):
            if step in steps_wanted:
                frame = Image.fromarray(env.render()).resize(
                    (512, 512), Image.NEAREST)
                save(frame, f"crafter_{cond}_step{step:03d}.png")
                print(f"  step {step}: achievements={sorted(achieved)} "
                      f"pos={info['player_pos']}")
            obs, r, done, info = env.step(names.index(e["action"]))
            achieved |= {n for n, c in info["achievements"].items() if c > 0}
    print("crafter done")


if __name__ == "__main__":
    capture_crafter_v2()
    capture_kirby()
