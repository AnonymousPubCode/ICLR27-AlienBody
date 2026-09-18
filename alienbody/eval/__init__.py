"""Evaluation metrics for AlienBody.

Primary: Calibration Efficiency (CE)
Secondary: Success Rate (SR), Calibration Accuracy (CA),
           Exploration Coverage (EC), Phase 1/2 efficiency

CA inference implemented for all 4 action types: A, B, C, D, and combos.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from alienbody.env.grid import DIRECTION_NAMES, DIRECTION_DELTAS

# ── Per-Family Human Baselines ─────────────────────────────────────
# From paper Section 4.1.4: n=18 participants, 40 environments each
# Format: {"total": avg_total_actions, "phase1": avg_p1, "phase2": avg_p2}

HUMAN_BASELINE = {
    1: {"total": 13.5, "phase1": 4.2, "phase2": 9.3},   # Remapped (easiest)
    2: {"total": 15.0, "phase1": 5.8, "phase2": 9.2},   # Directional
    3: {"total": 16.5, "phase1": 6.5, "phase2": 10.0},  # State-dependent
    4: {"total": 15.5, "phase1": 5.7, "phase2": 9.8},   # Relational
    5: {"total": 17.0, "phase1": 7.2, "phase2": 9.8},   # Composite (needs decomposition)
    6: {"total": 18.5, "phase1": 8.5, "phase2": 10.0},  # Temporal (needs history tracking)
}

HUMAN_AVG_DEFAULT = 15.0  # fallback


def get_human_baseline(family: int) -> float:
    """Get human average total actions for a family."""
    return HUMAN_BASELINE.get(family, {}).get("total", HUMAN_AVG_DEFAULT)


# ── Primary Metrics ────────────────────────────────────────────────

def calibration_efficiency(human_avg_actions: float, ai_actions: int) -> float:
    """CE = Human_Avg_Actions / AI_Actions. CE=1.0 means AI matches human."""
    if ai_actions <= 0:
        return 0.0
    return human_avg_actions / ai_actions


def success_rate(trajectories: list[dict]) -> float:
    """Percentage of episodes completed successfully."""
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t["success"]) / len(trajectories) * 100


# ── Exploration Metrics ────────────────────────────────────────────

def exploration_coverage(trajectory: dict, n_actions: int = 4) -> float:
    """Fraction of unique actions tried during Phase 1."""
    phase1_actions = [
        h["action"] for h in trajectory["history"]
        if h["phase"] == 1
    ]
    if not phase1_actions:
        return 0.0
    unique = len(set(phase1_actions))
    return unique / n_actions


def information_gain_curve(trajectory: dict, n_actions: int = 4) -> list[float]:
    """Cumulative fraction of unique actions discovered over Phase 1 steps."""
    seen = set()
    curve = []
    for record in trajectory["history"]:
        if record["phase"] != 1:
            break
        seen.add(record["action"])
        curve.append(len(seen) / n_actions)
    return curve


def exploration_efficiency(trajectory: dict, n_actions: int = 4) -> dict:
    """Detailed exploration analysis for Phase 1."""
    phase1 = [h for h in trajectory["history"] if h["phase"] == 1]
    if not phase1:
        return {"steps": 0, "unique_actions": 0, "repetition_rate": 0.0,
                "first_full_coverage_step": -1}

    actions = [h["action"] for h in phase1]
    seen = set()
    first_full = -1
    repeats = 0

    for i, a in enumerate(actions):
        if a in seen:
            repeats += 1
        seen.add(a)
        if len(seen) == n_actions and first_full == -1:
            first_full = i + 1

    return {
        "steps": len(actions),
        "unique_actions": len(set(actions)),
        "repetition_rate": repeats / len(actions) if actions else 0.0,
        "first_full_coverage_step": first_full,
    }


# ── Calibration Accuracy ──────────────────────────────────────────

def calibration_accuracy(trajectory: dict) -> float:
    """Measure how well the agent learned action mappings from Phase 1.

    Infers the agent's internal model from Phase 1 observations,
    then compares with ground truth. Returns fraction correct (0-1).

    Supports all action types: A, B, C, D, and combos (A+D).
    """
    gt_mapping = trajectory.get("ground_truth_mapping", [])
    if not gt_mapping:
        return 0.0

    action_type = trajectory.get("action_type", "B")
    inferred = _infer_action_model(trajectory, action_type)

    correct = 0
    for action_idx, gt_effect in enumerate(gt_mapping):
        if action_idx in inferred and inferred[action_idx] == gt_effect:
            correct += 1

    return correct / len(gt_mapping)


def _infer_action_model(trajectory: dict, action_type: str) -> dict[int, str]:
    """Infer what the agent learned about each action from Phase 1 observations.

    Returns {action_idx: inferred_effect_name}.
    """
    if "+" in action_type:
        return _infer_combo_model(trajectory, action_type)

    inferred = {}
    for record in trajectory["history"]:
        if record["phase"] != 1:
            continue
        action = record["action"]
        if action in inferred:
            continue  # first observation is most reliable

        result = None
        if action_type == "B":
            result = _infer_type_b(record)
        elif action_type == "A":
            result = _infer_type_a(record)
        elif action_type == "C":
            result = _infer_type_c(record)
        elif action_type == "D":
            result = _infer_type_d(record)
        elif action_type == "E":
            result = _infer_type_e(record)
        elif action_type == "F":
            result = _infer_type_f(record, trajectory["history"])

        if result:
            inferred[action] = result

    return inferred


def _infer_type_b(record: dict) -> str | None:
    """Type B: infer from position delta → cardinal direction."""
    if not record["position_changed"]:
        return None
    dr = record["new_pos"][0] - record["prev_pos"][0]
    dc = record["new_pos"][1] - record["prev_pos"][1]
    delta_to_name = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
    return delta_to_name.get((dr, dc))


def _infer_type_a(record: dict) -> str | None:
    """Type A: infer rotation vs movement from observation changes."""
    if record["direction_changed"] and not record["position_changed"]:
        prev_d = record["prev_dir"]
        new_d = record["new_dir"]
        delta = (new_d - prev_d) % 4
        if delta == 1:
            return "rotate_cw"
        elif delta == 3:
            return "rotate_ccw"
    elif record["position_changed"] and not record["direction_changed"]:
        dr = record["new_pos"][0] - record["prev_pos"][0]
        dc = record["new_pos"][1] - record["prev_pos"][1]
        facing = DIRECTION_DELTAS[record["prev_dir"]]
        if (dr, dc) == facing:
            return "forward"
        elif (dr, dc) == (-facing[0], -facing[1]):
            return "backward"
    return None


def _infer_type_c(record: dict) -> str | None:
    """Type C: infer relational actions from movement patterns.

    Key signals:
    - Large teleport (Manhattan dist > 2) → nearest_diff/same_color
    - Small move (1-2 cells) → move_toward_brightest or flee_same_color
    - No move → no-op (couldn't find valid target)
    """
    if not record["position_changed"]:
        return None

    dr = abs(record["new_pos"][0] - record["prev_pos"][0])
    dc = abs(record["new_pos"][1] - record["prev_pos"][1])
    manhattan = dr + dc

    if manhattan > 2:
        # Large teleport: nearest_diff_color or nearest_same_color
        # Can't distinguish without knowing cell colors, but we can flag as teleport
        return "move_to_nearest_diff_color"  # most common large-distance action
    elif manhattan == 1:
        # Small step: could be move_toward_brightest or flee_same_color
        return "move_toward_brightest"  # default for 1-step relational
    elif manhattan == 2:
        # 2-step: likely flee_same_color
        return "flee_same_color"
    return None


def _infer_type_d(record: dict) -> str | None:
    """Type D: infer from color changes and conditional behavior.

    Key signals:
    - Color changed, no position change → color_inc or color_dec
    - Position changed, no color change → color_move or color_interact
    - Direction changed (rotation) → color_interact (mismatch case)
    """
    if record["color_changed"] and not record["position_changed"] and not record["direction_changed"]:
        # Pure color change
        prev_c = record["prev_color"]
        new_c = record["new_color"]
        if (new_c - prev_c) % 4 == 1:
            return "color_inc"
        elif (new_c - prev_c) % 4 == 3:
            return "color_dec"
    elif record["position_changed"] and not record["color_changed"]:
        # Movement without color change — could be color_move or color_interact (match)
        # Distinguish: color_move direction depends on color, color_interact moves forward
        return "color_move"  # default for position-only change in Type D
    elif record["direction_changed"] and not record["position_changed"] and not record["color_changed"]:
        # Rotation without movement or color change → color_interact (mismatch case)
        return "color_interact"
    return None


def _infer_type_e(record: dict) -> str | None:
    """Type E: infer composite actions from combined position+direction changes.

    Composite actions produce two-step effects, so we look at the net result:
    - move_and_rotate_cw: position changes AND direction +1
    - rotate_and_strafe: direction changes AND position changes (in new facing dir)
    - double_forward: position changes by 2 cells in facing direction
    - retreat_and_spin: position backward AND direction +2 (180°)
    """
    pos_changed = record.get("position_changed", False)
    dir_changed = record.get("direction_changed", False)

    if pos_changed and dir_changed:
        prev_d = record["prev_dir"]
        new_d = record["new_dir"]
        dir_delta = (new_d - prev_d) % 4

        if dir_delta == 1:
            return "move_and_rotate_cw"
        elif dir_delta == 2:
            return "retreat_and_spin"
        elif dir_delta == 3:
            return "rotate_and_strafe"

    elif pos_changed and not dir_changed:
        # Could be double_forward (2-cell displacement) or single move
        dr = record["new_pos"][0] - record["prev_pos"][0]
        dc = record["new_pos"][1] - record["prev_pos"][1]
        if abs(dr) + abs(dc) >= 2:
            return "double_forward"

    return None


def _infer_type_f(record: dict, all_history: list[dict]) -> str | None:
    """Type F: infer temporal actions by comparing same-prev vs diff-prev behavior.

    This requires looking at multiple observations of the same action
    to detect the prev-action dependency pattern.
    """
    pos_changed = record.get("position_changed", False)
    if not pos_changed:
        return None  # no movement → could be repeat/inverse with no prior

    dr = record["new_pos"][0] - record["prev_pos"][0]
    dc = record["new_pos"][1] - record["prev_pos"][1]
    delta_to_dir = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    move_dir = delta_to_dir.get((dr, dc))
    if move_dir is None:
        return None

    # For temporal actions, the effect name encodes the action index
    # We can infer: if move_dir matches TYPE_F_SAME_DIR pattern → temporal_N
    # But we need the prev_action context to distinguish same/diff
    # Simple heuristic: map observed direction to most likely temporal_N
    action_idx = record["action"]

    # Check if this was a "same" press (action == prev_action)
    step_idx = next((i for i, h in enumerate(all_history) if h is record), -1)
    if step_idx > 0:
        prev_act = all_history[step_idx - 1]["action"]
        is_same = (action_idx == prev_act)
    else:
        is_same = False

    # Map: temporal_0 same→north(0), diff→east(1)
    #       temporal_1 same→south(2), diff→west(3)
    if is_same:
        if move_dir == 0:
            return "temporal_0"
        elif move_dir == 2:
            return "temporal_1"
    else:
        if move_dir == 1:
            return "temporal_0"
        elif move_dir == 3:
            return "temporal_1"

    # temporal_2/3 (repeat/inverse) are harder to infer from single obs
    return None


def _infer_combo_model(trajectory: dict, action_type: str) -> dict[int, str]:
    """Infer action model for combo types like 'A+D'.

    Split actions at midpoint: first half uses first type, second half uses second.
    """
    types = action_type.split("+")
    n_actions = len(trajectory.get("ground_truth_mapping", []))
    split = n_actions // len(types)

    inferred = {}
    for record in trajectory["history"]:
        if record["phase"] != 1:
            continue
        action = record["action"]
        if action in inferred:
            continue

        type_idx = min(action // split, len(types) - 1)
        sub_type = types[type_idx]

        result = None
        if sub_type == "A":
            result = _infer_type_a(record)
        elif sub_type == "B":
            result = _infer_type_b(record)
        elif sub_type == "C":
            result = _infer_type_c(record)
        elif sub_type == "D":
            result = _infer_type_d(record)

        if result:
            inferred[action] = result

    return inferred


# ── Aggregate Metrics ──────────────────────────────────────────────

def compute_all_metrics(
    trajectory: dict,
    human_avg_actions: float | None = None,
    n_actions: int = 4,
) -> dict:
    """Compute all metrics for a single trajectory.

    If human_avg_actions is None, auto-selects per-family baseline.
    """
    family = trajectory.get("family", 1)
    if human_avg_actions is None:
        human_avg_actions = get_human_baseline(family)

    total = trajectory["total_steps"]
    human_bl = HUMAN_BASELINE.get(family, {})

    return {
        "env_id": trajectory["env_id"],
        "family": family,
        "success": trajectory["success"],
        "total_steps": total,
        "phase1_steps": trajectory["phase1_steps"],
        "phase2_steps": trajectory["phase2_steps"],
        "ce": calibration_efficiency(human_avg_actions, total) if trajectory["success"] else 0.0,
        "sr": 1.0 if trajectory["success"] else 0.0,
        "ec": exploration_coverage(trajectory, n_actions),
        "ca": calibration_accuracy(trajectory),
        # Phase decomposition (for diagnostic ladder analysis)
        "human_p1": human_bl.get("phase1", 5.0),
        "human_p2": human_bl.get("phase2", 10.0),
        "p1_efficiency": human_bl.get("phase1", 5.0) / max(trajectory["phase1_steps"], 1),
        "p2_efficiency": human_bl.get("phase2", 10.0) / max(trajectory["phase2_steps"], 1) if trajectory["success"] else 0.0,
        **exploration_efficiency(trajectory, n_actions),
    }


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """Aggregate metrics across episodes."""
    if not all_metrics:
        return {}

    n = len(all_metrics)
    successful = [m for m in all_metrics if m["success"]]

    result = {
        "n_episodes": n,
        "n_success": len(successful),
        "ce_mean": sum(m["ce"] for m in all_metrics) / n,
        "sr_pct": sum(m["sr"] for m in all_metrics) / n * 100,
        "ec_mean": sum(m["ec"] for m in all_metrics) / n,
        "ca_mean": sum(m["ca"] for m in all_metrics) / n,
        "p1_mean": sum(m["phase1_steps"] for m in all_metrics) / n,
        "p2_mean": sum(m["phase2_steps"] for m in all_metrics) / n,
        "avg_repetition_rate": sum(m["repetition_rate"] for m in all_metrics) / n,
        # Per-success metrics (excluding failures)
        "ce_success_mean": (sum(m["ce"] for m in successful) / len(successful)) if successful else 0.0,
    }

    # Phase decomposition metrics (for diagnostic ladder)
    if "p1_efficiency" in all_metrics[0]:
        result["p1_eff_mean"] = sum(m["p1_efficiency"] for m in all_metrics) / n
        result["p2_eff_mean"] = sum(m["p2_efficiency"] for m in all_metrics) / n

    return result
