"""AlienBody environment — Gymnasium-compatible.

Design follows Gymnasium API: reset() -> (obs, info), step(action) -> (obs, reward, term, trunc, info).
Observation is a dict with 'image' (numpy array) and 'text' (str) for dual-modality support.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from alienbody.env.grid import EnvConfig, GridState, Phase, Position
from alienbody.env.actions import apply_action
from alienbody.env.renderer import render_image, render_text


class AlienBodyEnv:
    """AlienBody benchmark environment.

    Usage:
        config = EnvConfig.from_file("data/envs/family1/train/env_001.json")
        env = AlienBodyEnv(config)
        obs, info = env.reset()
        # Phase 1: explore
        for _ in range(20):
            obs, reward, term, trunc, info = env.step(action)
        # Phase 2: navigate
        env.start_phase2()
        while not done:
            obs, reward, term, trunc, info = env.step(action)
    """

    metadata = {"render_modes": ["image", "text", "both"]}

    def __init__(self, config: EnvConfig, render_mode: str = "both"):
        self.config = config
        self.render_mode = render_mode
        self.state: Optional[GridState] = None
        self._phase2_started = False

    @property
    def n_actions(self) -> int:
        return self.config.n_actions

    @property
    def phase(self) -> Phase:
        return self.state.phase if self.state else Phase.CALIBRATION

    @property
    def done(self) -> bool:
        return self.state.done if self.state else False

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        """Reset environment to initial state. Returns (observation, info)."""
        self.state = self.config.make_initial_state()
        self._phase2_started = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Execute action. Returns (obs, reward, terminated, truncated, info).

        - terminated: agent reached target (success)
        - truncated: exceeded action budget (failure)
        """
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        if self.state.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Record pre-action state
        prev_pos = self.state.agent_pos
        prev_dir = self.state.agent_dir
        prev_color = self.state.agent_color

        # Apply action
        self.state = apply_action(self.state, self.config, action)
        self.state.step_count += 1

        # Update phase-specific counters
        if self.state.phase == Phase.CALIBRATION:
            self.state.phase1_steps += 1
        else:
            self.state.phase2_steps += 1

        # Record action history
        self.state.action_history.append({
            "step": self.state.step_count,
            "phase": int(self.state.phase),
            "action": action,
            "prev_pos": (prev_pos.row, prev_pos.col),
            "new_pos": (self.state.agent_pos.row, self.state.agent_pos.col),
            "prev_dir": prev_dir,
            "new_dir": self.state.agent_dir,
            "prev_color": prev_color,
            "new_color": self.state.agent_color,
            "position_changed": self.state.agent_pos != prev_pos,
            "direction_changed": self.state.agent_dir != prev_dir,
            "color_changed": self.state.agent_color != prev_color,
        })

        # Check termination conditions
        terminated = False
        truncated = False
        reward = 0.0

        if self.state.phase == Phase.EXECUTION:
            target = Position(*self.config.target_pos)
            if self.state.agent_pos == target:
                terminated = True
                self.state.success = True
                reward = 1.0

        # Check budget limits
        if self.state.phase == Phase.CALIBRATION:
            if self.state.phase1_steps >= self.config.phase1_budget:
                # Auto-transition to Phase 2
                self._transition_to_phase2()
        if self.state.step_count >= self.config.max_total_steps:
            truncated = True

        if terminated or truncated:
            self.state.done = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def start_phase2(self) -> dict:
        """Manually transition to Phase 2. Returns new observation.

        Call this when agent declares 'done exploring'.
        If Phase 1 budget exhausted, this happens automatically.
        """
        if self.state.phase == Phase.EXECUTION:
            return self._get_obs()
        self._transition_to_phase2()
        return self._get_obs()

    def _transition_to_phase2(self):
        """Internal: switch from Phase 1 to Phase 2."""
        self.state.phase = Phase.EXECUTION
        self._phase2_started = True

    def _get_obs(self) -> dict:
        """Build observation dict with both modalities."""
        show_target = self.state.phase == Phase.EXECUTION
        obs = {}
        if self.render_mode in ("image", "both"):
            obs["image"] = render_image(self.state, self.config, show_target=show_target)
        if self.render_mode in ("text", "both"):
            obs["text"] = render_text(self.state, self.config, show_target=show_target)
        obs["phase"] = int(self.state.phase)
        obs["step"] = self.state.step_count
        # Last action feedback (if any)
        if self.state.action_history:
            last = self.state.action_history[-1]
            obs["feedback"] = self._format_feedback(last)
        return obs

    def _format_feedback(self, record: dict) -> str:
        """Human-readable feedback for last action."""
        parts = [f"Action {record['action']} executed."]
        if record["position_changed"]:
            parts.append(
                f"Position: ({record['prev_pos'][0]},{record['prev_pos'][1]}) "
                f"→ ({record['new_pos'][0]},{record['new_pos'][1]})"
            )
        else:
            parts.append("Position unchanged (wall or no-op).")
        if record["direction_changed"]:
            from alienbody.env.grid import DIRECTION_NAMES
            parts.append(f"Now facing: {DIRECTION_NAMES[record['new_dir']]}")
        if record["color_changed"]:
            parts.append(f"Color changed: {record['prev_color']} → {record['new_color']}")
        return " ".join(parts)

    def _get_info(self) -> dict:
        """Metadata for evaluation (not given to agent)."""
        info = {
            "env_id": self.config.env_id,
            "family": self.config.family,
            "phase": int(self.state.phase),
            "step_count": self.state.step_count,
            "phase1_steps": self.state.phase1_steps,
            "phase2_steps": self.state.phase2_steps,
            "agent_pos": (self.state.agent_pos.row, self.state.agent_pos.col),
            "agent_dir": self.state.agent_dir,
            "agent_color": self.state.agent_color,
            "prev_action": self.state.prev_action,
            "done": self.state.done,
            "success": self.state.success,
        }
        # Include last action record for agents that need structured history
        if self.state.action_history:
            info["last_record"] = self.state.action_history[-1]
        return info

    def get_trajectory(self) -> dict:
        """Full episode trajectory for analysis."""
        return {
            "env_id": self.config.env_id,
            "family": self.config.family,
            "action_type": self.config.action_type,
            "ground_truth_mapping": list(self.config.action_mapping),
            "history": list(self.state.action_history),
            "total_steps": self.state.step_count,
            "phase1_steps": self.state.phase1_steps,
            "phase2_steps": self.state.phase2_steps,
            "success": self.state.success,
        }
