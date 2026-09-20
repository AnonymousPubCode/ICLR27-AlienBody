"""Canonical, permission-audited state serialization for Interface Ladder v2.

``STATE_V2`` is deliberately a plain UTF-8 JSON block.  It is the only
environment-state payload an LLM arm may receive in the v2 evaluation.  The
renderer has no model-specific branches, so every arm can record and audit
the exact same bytes at every decision point.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from alienbody.env.grid import DIRECTION_NAMES, EnvConfig, GridState, Phase


STATE_V2_VERSION = "STATE_V2"


def actions_remaining(config: EnvConfig, state: GridState) -> int:
    """Return the number of legal environment actions still available."""
    return max(0, config.max_total_steps - state.step_count)


def state_v2_payload(config: EnvConfig, state: GridState) -> dict[str, Any]:
    """Return the complete, JSON-serializable public observation contract.

    The action mapping is intentionally included: Interface Ladder v2 measures
    planning after the mapping has been fixed, rather than mapping induction.
    Static layout information is represented only here; callers must not add
    config fields or Python objects to an LLM transcript.
    """
    target_visible = state.phase == Phase.EXECUTION
    payload: dict[str, Any] = {
        "contract": STATE_V2_VERSION,
        "grid_size": config.grid_size,
        "cell_colors_row_major": [list(row) for row in config.cell_colors],
        "obstacles": [list(pos) for pos in sorted(config.obstacles)],
        "agent": {
            "position": [state.agent_pos.row, state.agent_pos.col],
            "color": state.agent_color,
            "direction": DIRECTION_NAMES[state.agent_dir],
        },
        "target": list(config.target_pos) if target_visible else None,
        "phase": "execution" if target_visible else "calibration",
        "n_actions": config.n_actions,
        "action_mapping": {
            str(index): effect for index, effect in enumerate(config.action_mapping)
        },
        "actions_remaining": actions_remaining(config, state),
    }
    return payload


def render_state_v2(config: EnvConfig, state: GridState) -> str:
    """Render canonical bytes passed to an Interface Ladder v2 model.

    Compact separators and sorted keys are part of the contract: changing
    either intentionally changes the audit digest and requires a protocol
    version bump.
    """
    encoded = json.dumps(
        state_v2_payload(config, state),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{STATE_V2_VERSION}\n{encoded}"


def state_v2_sha256(config: EnvConfig, state: GridState) -> str:
    """SHA-256 digest of exactly the state bytes delivered to the model."""
    return hashlib.sha256(render_state_v2(config, state).encode("utf-8")).hexdigest()
