"""Failure mode auto-classifier for AlienBody.

Classifies failed episodes into 6 mutually exclusive categories
(paper Table 4). Rule-based for reproducibility.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from alienbody.eval import (
    exploration_coverage, calibration_accuracy, exploration_efficiency,
)


class FailureMode(str, Enum):
    INSUFFICIENT_EXPLORATION = "insufficient_exploration"
    RANDOM_EXPLORATION = "random_exploration"
    EXECUTION_ERROR = "execution_error"
    REASONING_ERROR = "reasoning_error"
    MEMORY_FAILURE = "memory_failure"
    PERCEPTION_ERROR = "perception_error"


# Classification thresholds
EC_THRESHOLD = 1.0      # full coverage = all 4 actions tried
CA_THRESHOLD = 0.5      # half correct = reasonable understanding
CA_GOOD_THRESHOLD = 0.75  # mostly correct
REPETITION_THRESHOLD = 0.6  # >60% repeated actions = random
EFFICIENCY_RATIO = 2.0  # >2x oracle path = execution error


def classify_failure(
    trajectory: dict,
    oracle_path_length: int | None = None,
    n_actions: int = 4,
) -> FailureMode | None:
    """Classify a failed episode into one failure mode.

    Returns None if episode succeeded.
    Categories are MUTUALLY EXCLUSIVE, applied in priority order.
    """
    if trajectory.get("success", False):
        return None

    ec = exploration_coverage(trajectory, n_actions)
    ca = calibration_accuracy(trajectory)
    ee = exploration_efficiency(trajectory, n_actions)

    # 1. Insufficient Exploration: didn't try all actions AND used few Phase 1 steps
    if ec < EC_THRESHOLD and ee["steps"] < n_actions * 2:
        return FailureMode.INSUFFICIENT_EXPLORATION

    # 2. Random Exploration: high repetition rate, no systematic strategy
    if ee["repetition_rate"] > REPETITION_THRESHOLD:
        return FailureMode.RANDOM_EXPLORATION

    # 3. Execution Error: good calibration but inefficient navigation
    if ca >= CA_GOOD_THRESHOLD:
        if oracle_path_length and trajectory["phase2_steps"] > oracle_path_length * EFFICIENCY_RATIO:
            return FailureMode.EXECUTION_ERROR
        # Even without oracle, if CA is high but failed, it's execution
        return FailureMode.EXECUTION_ERROR

    # 4. Reasoning Error: explored all actions but wrong model
    if ec >= EC_THRESHOLD and ca < CA_THRESHOLD:
        return FailureMode.REASONING_ERROR

    # 5. Memory Failure: check for Phase 2 behavior contradicting Phase 1
    if _detect_memory_failure(trajectory, n_actions):
        return FailureMode.MEMORY_FAILURE

    # 6. Perception Error: catch-all
    return FailureMode.PERCEPTION_ERROR


def _detect_memory_failure(trajectory: dict, n_actions: int) -> bool:
    """Check if Phase 2 actions contradict Phase 1 learned mappings.

    If agent used an action successfully in Phase 1 (observed its effect)
    but then uses it inconsistently in Phase 2, that's memory failure.
    """
    history = trajectory.get("history", [])

    # Build Phase 1 action → effect mapping
    phase1_effects: dict[int, list[tuple]] = {}
    for record in history:
        if record["phase"] != 1:
            break
        action = record["action"]
        if action not in phase1_effects:
            phase1_effects[action] = []
        if record["position_changed"]:
            dr = record["new_pos"][0] - record["prev_pos"][0]
            dc = record["new_pos"][1] - record["prev_pos"][1]
            phase1_effects[action].append((dr, dc))

    if not phase1_effects:
        return False

    # Check Phase 2: does the agent's action choice make sense given Phase 1?
    phase2_records = [r for r in history if r["phase"] == 2]
    if len(phase2_records) < 4:
        return False

    # Count contradictions: agent uses an action but goes wrong direction
    contradictions = 0
    for record in phase2_records[len(phase2_records) // 2:]:  # check second half
        action = record["action"]
        if action in phase1_effects and phase1_effects[action]:
            learned_delta = phase1_effects[action][0]
            if record["position_changed"]:
                actual_dr = record["new_pos"][0] - record["prev_pos"][0]
                actual_dc = record["new_pos"][1] - record["prev_pos"][1]
                # For Type B, delta should be consistent
                if (actual_dr, actual_dc) != learned_delta:
                    contradictions += 1

    # If >30% of late Phase 2 actions contradict Phase 1, it's memory failure
    return contradictions > len(phase2_records) * 0.15


def analyze_failures(
    trajectories: list[dict],
    oracle_lengths: dict[str, int] | None = None,
    n_actions: int = 4,
) -> dict:
    """Analyze failure mode distribution across episodes.

    Args:
        trajectories: list of trajectory dicts
        oracle_lengths: {env_id: optimal_path_length} for execution error detection
        n_actions: number of actions

    Returns dict with counts and percentages per failure mode.
    """
    failed = [t for t in trajectories if not t.get("success", False)]
    if not failed:
        return {"total_failed": 0, "distribution": {}}

    counts = {mode: 0 for mode in FailureMode}

    for traj in failed:
        oracle_len = None
        if oracle_lengths and traj.get("env_id") in oracle_lengths:
            oracle_len = oracle_lengths[traj["env_id"]]

        mode = classify_failure(traj, oracle_len, n_actions)
        if mode:
            counts[mode] += 1

    total = len(failed)
    distribution = {
        mode.value: {"count": count, "pct": count / total * 100 if total > 0 else 0}
        for mode, count in counts.items()
    }

    return {
        "total_episodes": len(trajectories),
        "total_failed": total,
        "total_success": len(trajectories) - total,
        "distribution": distribution,
    }
