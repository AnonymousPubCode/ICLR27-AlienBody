"""Agent interface + baselines.

Agent protocol:
  - act(observation, info) -> int (action index) or DONE_EXPLORING (-1)
  - reset() — reset agent state for new episode
  - DONE_EXPLORING signal: agent returns -1 to end Phase 1 early

Agent zoo:
  Control:       RandomAgent, OracleAgent, SystematicExplorer
  Algorithmic:   BayesianExplorer, MemoryAugmentedExplorer  (algorithmic_agents.py)
  LLM/VLM:       LLMAgent via make_agent()                  (llm_agent.py)
  Frameworks:    ReActAgent, ReflexionAgent,
                 PlanAndExecuteAgent, InnerMonologueAgent    (framework_agents.py)
  RL:            SB3Agent, MetaRLAgent                       (rl_agent.py)
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from alienbody.env.grid import Position, GridState, Phase, DIRECTION_DELTAS
from alienbody.env.actions import apply_action

# Special action: agent declares "done exploring" in Phase 1
DONE_EXPLORING = -1


class Agent(ABC):
    """Base agent interface."""

    @abstractmethod
    def act(self, observation: dict, info: dict | None = None) -> int:
        """Choose an action given current observation.

        Returns:
            int >= 0: action index to execute
            DONE_EXPLORING (-1): signal to end Phase 1 early
        """
        ...

    def reset(self):
        """Reset agent state for new episode."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class RandomAgent(Agent):
    """Random action selection. Lower bound baseline."""

    def __init__(self, n_actions: int = 4, seed: int = 42):
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def act(self, observation: dict, info: dict | None = None) -> int:
        return self.rng.randint(0, self.n_actions - 1)

    def reset(self):
        pass


class OracleAgent(Agent):
    """Oracle agent: knows action mapping + BFS optimal planning.

    Phase 1: systematically tries each action once, then signals DONE_EXPLORING.
    Phase 2: BFS shortest path from current position to target.
    """

    def __init__(self, config):
        from alienbody.env.grid import EnvConfig
        self.config: EnvConfig = config
        self._plan: list[int] = []
        self._phase1_count: int = 0

    def reset(self):
        self._plan = []
        self._phase1_count = 0

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)

        if phase == 1:
            # Systematic: try action 0, 1, 2, 3 in order, then signal done
            if self._phase1_count < self.config.n_actions:
                action = self._phase1_count
                self._phase1_count += 1
                return action
            return DONE_EXPLORING

        # Phase 2: follow BFS plan
        if not self._plan:
            self._plan = self._bfs_plan(info or {})
        if self._plan:
            return self._plan.pop(0)
        return 0

    def _bfs_plan(self, info: dict) -> list[int]:
        """BFS from current position to target with known action mapping."""
        cur_pos = info.get("agent_pos", self.config.agent_start)
        if isinstance(cur_pos, (list, tuple)) and len(cur_pos) == 2:
            cur_r, cur_c = cur_pos
        else:
            cur_r, cur_c = self.config.agent_start
        cur_dir = info.get("agent_dir", self.config.agent_start_dir)
        cur_color = info.get("agent_color", self.config.agent_start_color)
        prev_act = info.get("prev_action", -1)
        cur_lmd = info.get("last_move_delta", None)
        if cur_lmd is not None:
            cur_lmd = tuple(cur_lmd)
        target = Position(*self.config.target_pos)

        start_key = (cur_r, cur_c, cur_dir, cur_color, prev_act, cur_lmd)
        queue = deque([(start_key, [])])
        visited = {start_key}

        while queue:
            (r, c, d, color, pa, lmd), path = queue.popleft()
            if len(path) >= 40:
                continue
            for a in range(self.config.n_actions):
                state = GridState(
                    agent_pos=Position(r, c), agent_color=color,
                    agent_dir=d, phase=Phase.EXECUTION,
                    step_count=0, phase1_steps=0, phase2_steps=0,
                    prev_action=pa, last_move_delta=lmd,
                )
                state = apply_action(state, self.config, a)
                nk = (state.agent_pos.row, state.agent_pos.col,
                      state.agent_dir, state.agent_color, state.prev_action,
                      state.last_move_delta)
                if state.agent_pos == target:
                    return path + [a]
                if nk not in visited:
                    visited.add(nk)
                    queue.append((nk, path + [a]))
        return []


class SystematicExplorer(Agent):
    """Systematic exploration + greedy navigation.

    Phase 1: try each action exactly once, record effects.
    Phase 2: at each step, pick the action whose observed effect
             best aligns with the direction to target.
    """

    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions
        self._phase1_count: int = 0
        self._learned: dict[int, tuple[int, int]] = {}  # action → (dr, dc) observed delta
        self._target: tuple[int, int] | None = None

    def reset(self):
        self._phase1_count = 0
        self._learned = {}
        self._target = None

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)

        if phase == 1:
            if self._phase1_count < self.n_actions:
                action = self._phase1_count
                self._phase1_count += 1
                return action
            return DONE_EXPLORING

        # Phase 2: greedy navigate using learned deltas
        if info and "agent_pos" in info:
            return self._greedy_navigate(info)
        return 0

    def record_step(self, action: int, prev_pos: tuple, new_pos: tuple,
                    prev_dir: int = 0, new_dir: int = 0,
                    prev_color: int = 0, new_color: int = 0):
        """Called externally by run_episode to record observation.

        Records position deltas and direction changes for all action types.
        """
        dr = new_pos[0] - prev_pos[0]
        dc = new_pos[1] - prev_pos[1]
        if action not in self._learned:
            # Store position delta (for navigation)
            if dr != 0 or dc != 0:
                self._learned[action] = (dr, dc)
            # For rotation-only actions (Type A), record direction change
            elif new_dir != prev_dir:
                self._learned[action] = (0, 0)  # no movement, but action has effect

    def set_target(self, target: tuple[int, int]):
        """Set target position for Phase 2 navigation."""
        self._target = target

    def _greedy_navigate(self, info: dict) -> int:
        """Pick action whose learned delta best aligns with direction to target."""
        if not self._learned or not self._target:
            return 0

        cur_r, cur_c = info["agent_pos"]
        tr, tc = self._target
        goal_dr = tr - cur_r
        goal_dc = tc - cur_c

        if goal_dr == 0 and goal_dc == 0:
            return 0

        # Normalize goal direction
        goal_len = math.sqrt(goal_dr ** 2 + goal_dc ** 2)
        gdr, gdc = goal_dr / goal_len, goal_dc / goal_len

        best_action = 0
        best_score = -float("inf")

        for action, (dr, dc) in self._learned.items():
            # Dot product = alignment score
            score = dr * gdr + dc * gdc
            if score > best_score:
                best_score = score
                best_action = action

        return best_action


# ── Episode Runner Utility ─────────────────────────────────────────

def run_episode(env, agent: Agent, verbose: bool = False) -> dict:
    """Run a complete episode with proper DONE_EXPLORING handling.

    This is the canonical way to run an agent on an AlienBody environment.
    Handles Phase 1 → Phase 2 transition, including early termination.
    """
    agent.reset()
    obs, info = env.reset()

    if verbose:
        print(f"  Env: {env.config.env_id} | Family: {env.config.family} | Type: {env.config.action_type}")

    while not env.done:
        action = agent.act(obs, info)

        if action == DONE_EXPLORING:
            # Agent wants to end Phase 1 early
            if env.phase == Phase.CALIBRATION:
                obs = env.start_phase2()
                info = env._get_info()

                # Notify SystematicExplorer of target
                if isinstance(agent, SystematicExplorer):
                    agent.set_target(env.config.target_pos)

                if verbose:
                    print(f"  → Agent ended Phase 1 early at step {info['step_count']}")
                continue
            else:
                action = 0  # fallback if in Phase 2

        # Record pre-step position for agents that need it
        prev_pos = info.get("agent_pos", (0, 0))
        prev_dir = info.get("agent_dir", 0)
        prev_color = info.get("agent_color", 0)

        obs, reward, terminated, truncated, info = env.step(action)

        # Notify agents that track observations
        if hasattr(agent, "record_step"):
            try:
                agent.record_step(
                    action, prev_pos, info["agent_pos"],
                    prev_dir=prev_dir, new_dir=info.get("agent_dir", 0),
                    prev_color=prev_color, new_color=info.get("agent_color", 0),
                )
            except TypeError:
                agent.record_step(action, prev_pos, info["agent_pos"])

        # Auto phase 2 transition notification
        if info["phase"] == 2 and hasattr(agent, "set_target"):
            if hasattr(agent, "_target") and agent._target is None:
                agent.set_target(env.config.target_pos)

        if verbose and info["step_count"] <= 8:
            fb = obs.get("feedback", "")
            phase_str = "P1" if info["phase"] == 1 else "P2"
            print(f"  [{phase_str}] Step {info['step_count']}: Action {action} → {fb}")

    trajectory = env.get_trajectory()

    if verbose:
        status = "SUCCESS" if trajectory["success"] else "FAILED"
        print(f"  → {status} in {trajectory['total_steps']} steps "
              f"(P1={trajectory['phase1_steps']}, P2={trajectory['phase2_steps']})")

    return trajectory
