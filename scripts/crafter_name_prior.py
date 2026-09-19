#!/usr/bin/env python3
"""Crafter name-prior bridge (P-S1): named vs category vs anonymous actions.

External-validity experiment on a published RL benchmark (Crafter, Hafner
NeurIPS'21). Same screenshot→action loop as gb_name_prior.py under three
system-prompt conditions; the ONLY thing that differs is the action naming.
World screen = image (64x64 upscaled to 512x512), status/inventory = text for
all conditions. Post-episode calibration probe asks the model what each action
does and scores the answer against ground truth (LLM-judged FULL/PARTIAL/NONE).

Metrics:
  1. Main: achievements unlocked (0-22) per episode, mean/min/max over runs.
  2. Secondary (official): Crafter score S = exp(mean ln(1+succ)) - 1.
  3. Calibration probe CA = (FULL + 0.5*PARTIAL) / 17 -- measures "does the
     model know action semantics", immune to gameplay floor effects.

Paired seeds: run i uses seed_start+i; run the SAME --seed-start for all
three conditions so differences come from naming, not world luck.

Usage:
  # sanity: 25 steps, no probe (check image readability + $/call)
  python scripts/crafter_name_prior.py --condition named --max-steps 25 --no-probe
  # pilot: 1 run x 300 steps per condition (run the 3 conditions in parallel)
  python scripts/crafter_name_prior.py --condition named --max-steps 300
  python scripts/crafter_name_prior.py --condition category --max-steps 300
  python scripts/crafter_name_prior.py --condition anonymous --max-steps 300
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import crafter

# Canonical 22-achievement order (crafter.Env info['achievements'] keys).
ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood",
    "defeat_skeleton", "defeat_zombie", "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword", "make_stone_pickaxe",
    "make_stone_sword", "make_wood_pickaxe", "make_wood_sword",
    "place_furnace", "place_plant", "place_stone", "place_table",
    "wake_up",
]
# Early-game achievements (R1 floor-effect mitigation: look at these when
# total achievement count is near zero for every condition).
EARLY = {"collect_wood", "collect_stone", "collect_drink", "collect_sapling",
         "wake_up", "place_table"}

GOAL_LIST = ("wake_up, collect_wood, collect_stone, collect_coal, collect_iron, "
             "collect_diamond, collect_drink, collect_sapling, place_table, "
             "place_furnace, place_plant, place_stone, make_wood_pickaxe, "
             "make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, "
             "make_stone_sword, make_iron_sword, eat_plant, eat_cow, "
             "defeat_zombie, defeat_skeleton")

# Ground truth for the calibration probe (judge reference).
TRUTH = {
    "noop": "do nothing",
    "move_left": "walk one tile left",
    "move_right": "walk one tile right",
    "move_up": "walk one tile up",
    "move_down": "walk one tile down",
    "do": "interact with the object in front: chop trees, collect materials or water, attack creatures",
    "sleep": "sleep to recover energy",
    "place_stone": "place a stone block",
    "place_table": "place a crafting table",
    "place_furnace": "place a furnace for smelting iron",
    "place_plant": "plant a sapling",
    "make_wood_pickaxe": "craft a wooden pickaxe from wood at a table",
    "make_stone_pickaxe": "craft a stone pickaxe from stone at a table",
    "make_iron_pickaxe": "craft an iron pickaxe from iron and coal at a table and furnace",
    "make_wood_sword": "craft a wooden sword from wood at a table",
    "make_stone_sword": "craft a stone sword from stone at a table",
    "make_iron_sword": "craft an iron sword from iron and coal at a table and furnace",
}

# System prompts. Identical except for how actions are presented. The
# achievement list is shared across conditions so goal information is not
# confounded with naming.
CONDITIONS = {
    "named": f"""You are playing Crafter, a 2D open-world survival game. Each turn you see the game screen and your status; you output one action.

Actions (output exactly one name):
- noop: do nothing
- move_left / move_right / move_up / move_down: walk one tile
- do: use the object in front of you (chop trees, collect water, attack creatures)
- sleep: rest to recover energy
- place_stone / place_table / place_furnace / place_plant: place the object in front of you
- make_wood_pickaxe / make_stone_pickaxe / make_iron_pickaxe: craft the pickaxe (needs materials; stone/iron need a table and furnace)
- make_wood_sword / make_stone_sword / make_iron_sword: craft the sword (needs materials; stone/iron need a table and furnace)

Achievements to unlock: {GOAL_LIST}.

Keep health, food, drink and energy up. Avoid zombies and skeletons.""",
    "category": f"""You are playing a 2D open-world survival game. Each turn you see the game screen and your status; you output one action number.

Your 17 actions are numbered 0 to 16:
- 0, 1, 2, 3: move (walk one tile; each is a different direction)
- 4: interact (do something with the object in front of you)
- 5: rest (recover)
- 6, 7, 8, 9: place (put down some kind of object in front of you)
- 10, 11, 12, 13, 14, 15, 16: craft (make some kind of tool; probably needs materials and equipment)

Within each category the exact effect of each action differs; you do NOT know the specifics. Experiment to figure them out.

Achievements to unlock: {GOAL_LIST}.

Keep health, food, drink and energy up. Avoid hostile creatures.""",
    "anonymous": f"""You are playing a 2D game. Each turn you see the game screen and your status; you output one action.

Your 17 actions are labeled action_0 through action_16. You do NOT know what any of them does.

Goals to achieve: {GOAL_LIST}.

Keep your status values up. Experiment to figure out what each action does.""",
}

TASK_LINE = {
    "named": "\n\nEach turn, output ONLY the action name, nothing else.",
    "category": "\n\nEach turn, output ONLY the action number (0-16), nothing else.",
    "anonymous": "\n\nEach turn, output ONLY \"action_N\" where N is the number, nothing else.",
}


def parse_action(text: str, condition: str, names: list[str]) -> int:
    """Map a model response to an action index. 0 = noop on failure."""
    if condition == "named":
        for name in sorted(names, key=len, reverse=True):
            if name == "noop":
                continue
            spaced = name.replace("_", r"[ _]*")
            if re.search(rf"(?<![a-z0-9_]){spaced}(?![a-z0-9_])", text, re.I):
                return names.index(name)
        return 0
    if condition == "anonymous":
        m = re.search(r"action[_\s]*(\d{1,2})", text, re.I)
        if m:
            return min(max(int(m.group(1)), 0), 16)
        return 0
    # category: first standalone integer in 0-16
    m = re.search(r"(?<!\d)(1[0-6]|[0-9])(?!\d)", text)
    if m:
        return min(max(int(m.group(1)), 0), 16)
    return 0


def screenshot_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def format_inventory(inv: dict) -> str:
    """Text status line; identical for all conditions (no action semantics)."""
    parts = []
    for k in ("health", "food", "drink", "energy"):
        if inv.get(k):
            parts.append(f"{k.capitalize()} {inv[k]}")
    for k, v in inv.items():
        if k not in ("health", "food", "drink", "energy") and v:
            parts.append(f"{k} {v}")
    return ", ".join(parts) if parts else "no status"


def _probe_question(condition: str, idx: int, name: str) -> str:
    """Condition-aware probe question: ask in the vocabulary the condition used."""
    if condition == "named":
        return (f"In a 2D survival game, what does the action '{name}' do? "
                "Describe the specific effect in one short phrase.")
    if condition == "category":
        return ("In the game you played, the 17 actions were grouped: "
                "0-3 move, 4 interact, 5 rest, 6-9 place, 10-16 craft. "
                f"What does the action numbered {idx} do? "
                "Describe the specific effect in one short phrase.")
    return ("In a 2D survival game with 17 actions labeled action_0 through "
            f"action_16, what does the action called action_{idx} do? "
            "Describe the specific effect in one short phrase.")


def probe_calibration(client, condition: str, names: list[str]) -> dict:
    """Ask what each action does, then LLM-judge against TRUTH.

    CA = (FULL + 0.5*PARTIAL) / 17. 'named' is a ceiling sanity check
    (should approach 1.0); 'anonymous' measures whether the label alone
    carries any semantics. No 'unknown' escape hatch: the judge rules on
    wrong or vague answers (NONE), which avoids over-cautious hedging.
    """
    items = []
    for idx, name in enumerate(names):
        try:
            ans = client.complete([
                {"role": "system", "content":
                 "Answer in ONE short phrase describing the effect. Be specific."},
                {"role": "user", "content": _probe_question(condition, idx, name)},
            ], max_tokens=40)
        except Exception as e:
            items.append({"action": name, "answer": f"ERROR: {e}", "verdict": "NONE"})
            continue
        try:
            verdict = client.complete([
                {"role": "system", "content":
                 "You judge whether a student's description of a game action matches "
                 "its true effect. Respond with exactly one word: FULL if the specific "
                 "effect is correct, PARTIAL if only the right category is captured, "
                 "NONE if wrong or unknown."},
                {"role": "user", "content":
                 f"True effect: {TRUTH[name]}\nStudent answer: {ans}"},
            ], max_tokens=8)
        except Exception as e:
            verdict = f"ERROR: {e}"
        v = "NONE"
        for tag in ("FULL", "PARTIAL", "NONE"):
            if tag in verdict.upper():
                v = tag
                break
        items.append({"action": name, "answer": ans[:80], "verdict": v})
    ca = sum({"FULL": 1.0, "PARTIAL": 0.5}.get(i["verdict"], 0.0)
             for i in items) / len(names)
    return {"ca": round(ca, 3), "items": items}


def run_episode(condition: str, model: str, seed: int, max_steps: int,
                out_dir: str, do_probe: bool = True) -> dict:
    from alienbody.agents.gateway_client import GatewayClient

    client = GatewayClient(model=model)
    env = crafter.Env(seed=seed)
    names = list(env.action_names)

    system_prompt = CONDITIONS[condition] + TASK_LINE[condition]
    log = []
    achieved = set()
    died = False
    n_parse_fail = 0
    t0_total = time.time()

    obs = env.reset()
    info = {"inventory": {}, "achievements": {}}

    for step in range(max_steps):
        img = Image.fromarray(env.render()).resize((512, 512), Image.NEAREST)
        b64 = screenshot_b64(img)
        inv_text = format_inventory(info.get("inventory", {}))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text":
                    f"Step {step}. Achievements unlocked: {len(achieved)}.\n"
                    f"Status: {inv_text}\nCurrent screen:"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        t0 = time.time()
        try:
            resp = client.complete(messages, max_tokens=16)
        except Exception as e:
            log.append({"step": step, "error": str(e)})
            break
        idx = parse_action(resp, condition, names)
        if idx == 0 and condition != "named":
            # indistinguishable from a real noop, but count text that parsed
            # to nothing as a failure for the parse-failure metric
            if condition == "anonymous" and not re.search(r"action[_\s]*\d", resp, re.I):
                n_parse_fail += 1
            elif condition == "category" and not re.search(r"\d", resp):
                n_parse_fail += 1
        log.append({"step": step, "response": resp[:60], "action": names[idx],
                    "latency": round(time.time() - t0, 1)})

        obs, reward, done, info = env.step(idx)
        achieved |= {n for n, c in info["achievements"].items() if c > 0}
        if done:
            died = bool(info.get("discount") == 0)
            break

    probe = probe_calibration(client, condition, names) if do_probe else None

    result = {
        "condition": condition, "model": model, "seed": seed,
        "max_steps": max_steps, "steps_run": len(log), "died": died,
        "achievements": len(achieved),
        "early_achievements": len(achieved & EARLY),
        "achievements_list": sorted(achieved),
        "parse_failures": n_parse_fail,
        "total_tokens": client.total_tokens,
        "wall_seconds": round(time.time() - t0_total, 1),
        "probe": probe,
    }
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, f"{condition}_seed{seed}_{ts}.json"), "w") as f:
        json.dump({"result": result, "log": log}, f, indent=1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITIONS), default="named")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--n-runs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--probe-only", default=None,
                    help="path to a run json: re-run the calibration probe only")
    ap.add_argument("--output",
                    default=os.path.join(HERE, "..", "results",
                                         "crafter_name_prior_gpt4o"))
    args = ap.parse_args()

    if args.probe_only:
        from alienbody.agents.gateway_client import GatewayClient
        with open(args.probe_only) as f:
            data = json.load(f)
        r = data["result"]
        client = GatewayClient(model=r["model"])
        names = list(crafter.Env(seed=0).action_names)
        probe = probe_calibration(client, r["condition"], names)
        data["result"]["probe"] = probe
        with open(args.probe_only, "w") as f:
            json.dump(data, f, indent=1)
        print(f"CA={probe['ca']} for {r['condition']} (run {args.probe_only})")
        for it in probe["items"]:
            print(f"  {it['action']:20s} {it['verdict']:8s} {it['answer'][:60]}")
        return

    print(f"condition={args.condition} model={args.model} "
          f"runs={args.n_runs} max_steps={args.max_steps} "
          f"seed_start={args.seed_start} probe={not args.no_probe}")

    results = []
    for i in range(args.n_runs):
        seed = args.seed_start + i
        print(f"\n=== run {i+1}/{args.n_runs} (seed {seed}) ===")
        r = run_episode(args.condition, args.model, seed, args.max_steps,
                        args.output, do_probe=not args.no_probe)
        results.append(r)
        ca = r["probe"]["ca"] if r["probe"] else None
        print(f"  achievements={r['achievements']} early={r['early_achievements']} "
              f"steps={r['steps_run']} died={r['died']} "
              f"parse_fail={r['parse_failures']} tokens={r['total_tokens']} "
              f"CA={ca} wall={r['wall_seconds']}s")

    n = len(results)
    ach = [r["achievements"] for r in results]
    early = [r["early_achievements"] for r in results]
    cas = [r["probe"]["ca"] for r in results if r["probe"]]
    succ = {a: sum(1 for r in results if a in r["achievements_list"]) / n
            for a in ACHIEVEMENTS}
    score = (math.exp(sum(math.log(1 + s) for s in succ.values()) / len(succ)) - 1
             if n else 0.0)
    agg = {
        "condition": args.condition, "model": args.model, "n_runs": n,
        "mean_achievements": round(sum(ach) / n, 2),
        "min_achievements": min(ach), "max_achievements": max(ach),
        "mean_early": round(sum(early) / n, 2),
        "crafter_score": round(score, 3),
        "achievement_success_rates": {a: round(s, 3)
                                      for a, s in succ.items() if s > 0},
        "mean_probe_ca": round(sum(cas) / len(cas), 3) if cas else None,
        "mean_steps": round(sum(r["steps_run"] for r in results) / n, 1),
        "died_runs": sum(1 for r in results if r["died"]),
        "total_tokens": sum(r["total_tokens"] for r in results),
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
