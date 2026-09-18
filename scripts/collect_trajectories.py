#!/usr/bin/env python3
"""Collect offline babbling trajectories for FMB Stage 1 training.

Runs OracleAgent's Phase 1 on training environments, recording:
  - Grid image (PNG)
  - Action taken
  - Observed effect (position delta, direction change, color change)
  - Ground-truth ActionSchema

Output: JSONL file with one record per (env, action) pair.
Used to train the VLM to predict ActionSchema from observations.

Usage:
    python scripts/collect_trajectories.py --family 1 --split train --n-envs 50
    python scripts/collect_trajectories.py --family all --split train --n-envs 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig
from alienbody.env.actions import get_all_effect_names
from alienbody.schema import ActionSchema, EffectType, ForwardModel
from alienbody.agents import OracleAgent, DONE_EXPLORING


def action_mapping_to_schema(config: EnvConfig) -> ForwardModel:
    """Convert ground-truth action_mapping to ForwardModel."""
    model = ForwardModel()
    action_type = config.action_type

    for i, effect_name in enumerate(config.action_mapping):
        schema = _effect_to_schema(effect_name, action_type)
        model.actions[i] = schema

    return model


def _effect_to_schema(effect_name: str, action_type: str) -> ActionSchema:
    """Map effect name → ActionSchema."""
    # Type B: cardinal directions
    if effect_name in ("up", "down", "left", "right"):
        return ActionSchema(EffectType.TRANSLATE, {"direction": effect_name})

    # Type A: rotation-based
    if effect_name in ("rotate_cw", "rotate_ccw"):
        rot = "cw" if effect_name == "rotate_cw" else "ccw"
        return ActionSchema(EffectType.ROTATE, {"rotation": rot})
    if effect_name in ("forward", "backward"):
        return ActionSchema(EffectType.TRANSLATE, {"direction": effect_name})

    # Type D: state-modifying
    if effect_name in ("color_inc", "color_dec", "color_move", "color_interact"):
        return ActionSchema(EffectType.STATE_CHANGE, {"change": effect_name})

    # Type C: relational
    if effect_name in ("move_to_nearest_diff_color", "move_to_nearest_same_color",
                       "move_toward_brightest", "flee_same_color"):
        relation = effect_name.replace("move_to_", "").replace("move_", "")
        return ActionSchema(EffectType.RELATIONAL, {"relation": relation})

    # Type E: composite
    if effect_name in ("move_and_rotate_cw", "rotate_and_strafe",
                       "double_forward", "retreat_and_spin"):
        # Decompose based on name
        if effect_name == "move_and_rotate_cw":
            primitives = [
                {"type": "translate", "params": {"direction": "forward"}},
                {"type": "rotate", "params": {"rotation": "cw"}},
            ]
        elif effect_name == "rotate_and_strafe":
            primitives = [
                {"type": "rotate", "params": {"rotation": "ccw"}},
                {"type": "translate", "params": {"direction": "forward"}},
            ]
        elif effect_name == "double_forward":
            primitives = [
                {"type": "translate", "params": {"direction": "forward"}},
                {"type": "translate", "params": {"direction": "forward"}},
            ]
        elif effect_name == "retreat_and_spin":
            primitives = [
                {"type": "translate", "params": {"direction": "backward"}},
                {"type": "rotate", "params": {"rotation": "cw"}},
                {"type": "rotate", "params": {"rotation": "cw"}},
            ]
        return ActionSchema(EffectType.COMPOSITE, {"primitives": primitives})

    # Type F: temporal
    if effect_name.startswith("temporal_"):
        return ActionSchema(EffectType.TEMPORAL, {"effect_name": effect_name})

    # Fallback
    return ActionSchema(EffectType.TRANSLATE, {"direction": "up"})


def run_phase1_oracle(config: EnvConfig) -> list[dict]:
    """Run Oracle Phase 1 exploration and return observation records."""
    env = AlienBodyEnv(config, render_mode="image")

    # Phase 1: Oracle systematically tests each action once
    agent = OracleAgent(config)
    agent.reset()
    obs, info = env.reset()

    records = []
    tested = set()
    prev_pos = info["agent_pos"]
    prev_dir = info["agent_dir"]
    prev_color = info["agent_color"]

    while env.phase.value == 1 and env.state.phase1_steps < config.phase1_budget:
        action = agent.act(obs, info)

        if action == DONE_EXPLORING:
            break

        obs, reward, term, trunc, info = env.step(action)

        new_pos = info["agent_pos"]
        new_dir = info["agent_dir"]
        new_color = info["agent_color"]

        # Only record first observation of each action (clean, no-condition data)
        if action not in tested:
            tested.add(action)
            records.append({
                "env_id": config.env_id,
                "family": config.family,
                "action_type": config.action_type,
                "action": action,
                "prev_pos": list(prev_pos),
                "new_pos": list(new_pos),
                "prev_dir": prev_dir,
                "new_dir": new_dir,
                "prev_color": prev_color,
                "new_color": new_color,
                "dr": new_pos[0] - prev_pos[0],
                "dc": new_pos[1] - prev_pos[1],
                "ddir": (new_dir - prev_dir) % 4,
                "dcolor": new_color - prev_color,
            })

        prev_pos = new_pos
        prev_dir = new_dir
        prev_color = new_color

    return records


def collect_dataset(
    family: int | str,
    split: str,
    output_path: str,
    n_envs: int = -1,
    data_dir: str = "data/envs",
    include_image: bool = False,
):
    """Collect offline babbling dataset.

    Args:
        family: 1-6 or "all"
        split: "train", "dev", or "test"
        output_path: output JSONL file
        n_envs: max environments (-1 = all)
        data_dir: environment config directory
        include_image: if True, save base64 PNG images (large files!)
    """
    import base64
    import io
    from PIL import Image

    data_dir = Path(data_dir)
    families = list(range(1, 7)) if family == "all" else [int(family)]
    configs = []

    for fam in families:
        env_dir = data_dir / f"family{fam}" / split
        if not env_dir.exists():
            print(f"  Warning: {env_dir} not found, skipping")
            continue
        for json_file in sorted(env_dir.glob("env_*.json")):
            configs.append(EnvConfig.from_file(str(json_file)))

    if n_envs > 0:
        configs = configs[:n_envs]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, config in enumerate(configs):
            # Get ground-truth schema
            gt_model = action_mapping_to_schema(config)

            # Run Phase 1 exploration
            records = run_phase1_oracle(config)

            for rec in records:
                action = rec["action"]
                gt_schema = gt_model.actions.get(action)
                if gt_schema is None:
                    continue

                record = {
                    "env_id": rec["env_id"],
                    "family": rec["family"],
                    "action_type": rec["action_type"],
                    "action": action,
                    "prev_state": {
                        "pos": rec["prev_pos"],
                        "dir": rec["prev_dir"],
                        "color": rec["prev_color"],
                    },
                    "effect": {
                        "dr": rec["dr"],
                        "dc": rec["dc"],
                        "ddir": rec["ddir"],
                        "dcolor": rec["dcolor"],
                    },
                    "ground_truth_schema": gt_schema.to_dict(),
                }

                if include_image:
                    # Re-render the initial state image
                    env = AlienBodyEnv(config, render_mode="image")
                    env.reset()
                    # Navigate to the state before this action
                    # (simplified: just render current state)
                    obs, _ = env.reset()
                    img = obs.get("image")
                    if img is not None:
                        buf = io.BytesIO()
                        Image.fromarray(img).save(buf, format="PNG")
                        record["image_b64"] = base64.b64encode(buf.getvalue()).decode()

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(configs)}] {total_records} records collected")

    print(f"\nDone: {total_records} records -> {output_path}")
    return total_records


def main():
    parser = argparse.ArgumentParser(description="Collect FMB training data")
    parser.add_argument("--family", type=str, default="all")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-envs", type=int, default=-1)
    parser.add_argument("--output", type=str, default="data/fmb_trajectories/train.jsonl")
    parser.add_argument("--data-dir", type=str, default="data/envs")
    parser.add_argument("--include-image", action="store_true",
                        help="Include base64 PNG images (large files)")
    args = parser.parse_args()

    collect_dataset(
        family=args.family,
        split=args.split,
        output_path=args.output,
        n_envs=args.n_envs,
        data_dir=args.data_dir,
        include_image=args.include_image,
    )


if __name__ == "__main__":
    main()
