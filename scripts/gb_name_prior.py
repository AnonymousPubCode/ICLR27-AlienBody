#!/usr/bin/env python3
"""GB name-prior bridge (R4): Kirby's Dream Land — named vs anonymous buttons.

External-validity bridge for the AlienBody name-prior ablation: Game Boy
buttons carry strong pretraining priors ("A = jump"). We run the same
screenshot→action loop under three prompt conditions and measure objective
level progress via RAM (scroll_x / health) plus checkpoint hash matches.

Run locally (PyBoy on Windows via VideoGameZero) or anywhere with pyboy;
the VLM is called through AlienBody's FuxiClient (GPT-4o image).

Usage:
  python scripts/gb_name_prior.py --condition named --n-runs 3
  python scripts/gb_name_prior.py --condition category --n-runs 3
  python scripts/gb_name_prior.py --condition anonymous --n-runs 3
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
VZG_ROOT = os.environ.get("VZG_ROOT", r"D:\Workspace\VideoGameZero")
sys.path.insert(0, VZG_ROOT)
sys.path.insert(0, os.path.join(HERE, ".."))

# Kirby RAM map (from VideoGameZero rl/src/env.py, DataCrystal)
RAM = {
    "game_state": 0xD02C,
    "scroll_x": 0xD051,
    "kirby_x": 0xD05C,
    "kirby_y": 0xD05D,
    "health": 0xD086,
    "lives": 0xD089,
}

ROM_PATH = os.path.join(VZG_ROOT, "roms", "kirby_dream_land_dx.gb")

BUTTONS = ["LEFT", "RIGHT", "UP", "DOWN", "A", "B", "START", "SELECT"]

CONDITIONS = {
    "named": """You are playing Kirby's Dream Land on Game Boy.
Controls:
- LEFT/RIGHT: Move
- A: Jump (press UP after jumping to fly)
- B: Inhale enemies (press DOWN to swallow, B again to spit as projectile)
- UP: Enter doors (must be standing on ground)
- START: Pause
Key tips:
- Inhale enemies with B, then spit them at other enemies to deal damage.
- To enter doors, stand in front and press UP (do NOT float above them).
- Shining stars end the level - walk into them.
- Avoid touching enemies directly or you take damage.
- If the screen looks the same for several steps, try a different direction or action.""",
    "category": """You are playing a side-scrolling platform game on a handheld console.
You have 8 buttons. Their effects fall into four categories: moving left/right,
jumping/flying, inhaling objects, and entering doors — but you do NOT know
which button does which. Figure out what each button does by trying them,
then make progress to the right.""",
    "anonymous": """You are playing a game on a handheld console.
You have 8 buttons, labeled B0..B7. You do NOT know what they do.
Figure out what each button does by trying them, then make progress in the game.""",
}

TASK_LINE = "\n\nEach turn, output ONLY a button or button combination, e.g. \"RIGHT\" or \"RIGHT+A\", nothing else."


def parse_buttons(text: str) -> list[str]:
    """Extract button presses from a model response."""
    out = []
    upper = text.upper()
    for b in BUTTONS:
        if b in upper:
            out.append(b)
    if not out:
        # fall back to single-letter scan on last line
        last = upper.strip().split("\n")[-1]
        for b in BUTTONS:
            if b in last:
                out.append(b)
    return out[:2]  # at most 2 buttons


def screenshot_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def checkpoint_hashes(cp_dir: str) -> dict[int, str]:
    """Load checkpoint screenshots as perceptual hashes (dHash)."""
    out = {}
    if not os.path.isdir(cp_dir):
        return out
    from PIL import Image
    for fname in sorted(os.listdir(cp_dir)):
        if not fname.endswith(".png"):
            continue
        try:
            idx = int(fname.split(".")[0])
        except ValueError:
            continue
        out[idx] = dh(Image.open(os.path.join(cp_dir, fname)))
    return out


def dh(img) -> str:
    """dHash of a screen (for checkpoint matching)."""
    from PIL import Image
    g = img.convert("L").resize((9, 8))
    px = list(g.getdata())
    diff = 0
    for r in range(8):
        for c in range(8):
            if px[r * 9 + c] > px[r * 9 + c + 1]:
                diff |= 1 << (r * 8 + c)
    return str(diff)


def hamming(a: str, b: str) -> int:
    return bin(int(a) ^ int(b)).count("1")


def run_episode(condition: str, model: str, max_steps: int,
                cp_hashes: dict[int, str], out_dir: str) -> dict:
    from shared.emulators.gba.interface import GBAInterface as GameBoyInterface
    from alienbody.agents.fuxi_client import FuxiClient

    client = FuxiClient(model=model)
    env = GameBoyInterface(render=False)
    ok = env.load_game(ROM_PATH)
    assert ok, f"failed to load {ROM_PATH}"

    system_prompt = CONDITIONS[condition] + TASK_LINE
    log = []

    # load_game() already boots the ROM to its initial state
    # Warmup: Kirby boots to a title screen — press START twice to reach
    # the first level (identical for all conditions, so the comparison stays fair)
    for _ in range(8):
        env.step({"START": True}, skip_frames=12)
    max_scroll = 0
    max_cp = 0
    min_health = 99
    deaths = 0
    prev_health = None

    for step in range(max_steps):
        screen = env.get_screen()
        b64 = screenshot_b64(screen)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Step {step}. Current screen:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        t0 = time.time()
        try:
            resp = client.complete(messages, max_tokens=64)
        except Exception as e:
            log.append({"step": step, "error": str(e)})
            break
        buttons = parse_buttons(resp)
        if not buttons:
            buttons = ["A"]
        log.append({"step": step, "response": resp[:80], "buttons": buttons,
                    "latency": round(time.time() - t0, 1)})

        # Execute: press combo for 10 frames (step handles press->release)
        action = {b: True for b in buttons}
        env.step(action, skip_frames=10)

        # RAM metrics
        mem = env.read_memory_values(RAM)
        if mem.get("scroll_x", 0) is not None:
            max_scroll = max(max_scroll, mem["scroll_x"])
        health = mem.get("health")
        if health is not None:
            min_health = min(min_health, health)
            if prev_health is not None and health < prev_health and health < 6:
                deaths += 1
            prev_health = health

        # Checkpoint match
        h = dh(screen)
        for idx, ref in cp_hashes.items():
            if hamming(h, ref) <= 10 and idx > max_cp:
                max_cp = idx

        if mem.get("game_state") in (0x06, 6):
            log.append({"step": step, "event": "died"})
            deaths += 1
            prev_health = None

    env.close()

    result = {
        "condition": condition, "model": model, "max_steps": max_steps,
        "max_scroll_x": max_scroll, "max_checkpoint": max_cp,
        "min_health": min_health, "deaths": deaths, "steps_run": len(log),
    }
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, f"{condition}_{ts}.json"), "w") as f:
        json.dump({"result": result, "log": log}, f, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITIONS), default="named")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--checkpoint-dir",
                    default=os.path.join(VZG_ROOT, "rl", "configs",
                                         "kirby_dream_land_dx", "checkpoints"))
    ap.add_argument("--output", default=os.path.join(HERE, "..", "results", "gb_name_prior"))
    args = ap.parse_args()

    cp_hashes = checkpoint_hashes(args.checkpoint_dir)
    print(f"condition={args.condition} model={args.model} "
          f"runs={args.n_runs} max_steps={args.max_steps} "
          f"checkpoints={len(cp_hashes)}")

    results = []
    for i in range(args.n_runs):
        print(f"\n=== run {i+1}/{args.n_runs} ===")
        r = run_episode(args.condition, args.model, args.max_steps,
                        cp_hashes, args.output)
        results.append(r)
        print(f"  scroll_x={r['max_scroll_x']} checkpoint={r['max_checkpoint']} "
              f"min_health={r['min_health']} deaths={r['deaths']}")

    agg = {
        "condition": args.condition,
        "mean_scroll_x": sum(r["max_scroll_x"] for r in results) / len(results),
        "mean_max_checkpoint": sum(r["max_checkpoint"] for r in results) / len(results),
        "mean_deaths": sum(r["deaths"] for r in results) / len(results),
        "runs": results,
    }
    out_p = os.path.join(args.output, f"summary_{args.condition}.json")
    os.makedirs(args.output, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(agg, f, indent=1)
    print(f"\nwrote {out_p}")
    print(json.dumps({k: v for k, v in agg.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
