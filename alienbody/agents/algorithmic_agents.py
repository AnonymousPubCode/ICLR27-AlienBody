"""Algorithmic agents: Bayesian Explorer and Memory-Augmented Explorer.

These agents use principled exploration strategies without LLM calls —
pure algorithmic baselines that highlight what systematic reasoning achieves.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Optional

from alienbody.agents import Agent, DONE_EXPLORING
from alienbody.env.grid import Position, DIRECTION_DELTAS, DIRECTION_NAMES


# ═══════════════════════════════════════════════════════════════════
#  BayesianExplorer
# ═══════════════════════════════════════════════════════════════════

# Observable effect fingerprints: (dr, dc, ddir, dcolor) → effect name
_CARDINAL_EFFECTS = {
    (-1, 0, 0, 0): "up",
    (1, 0, 0, 0): "down",
    (0, -1, 0, 0): "left",
    (0, 1, 0, 0): "right",
}

_ROTATION_EFFECTS = {
    (0, 0, 1, 0): "rotate_cw",
    (0, 0, -1, 0): "rotate_ccw",
    (0, 0, 3, 0): "rotate_ccw",  # -1 mod 4 = 3
}


class BayesianExplorer(Agent):
    """Maintains a posterior over action→effect mappings.

    Phase 1: Chooses actions by maximum entropy (most uncertain first).
    Phase 2: Uses MAP estimates to plan via BFS-like greedy navigation.

    Works best on families with observable position/direction effects (F1, F2, F5).
    Falls back to greedy heuristic for families with state-dependent effects.
    """

    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions
        self._observations: dict[int, list[dict]] = defaultdict(list)
        self._action_model: dict[int, tuple[int, int]] = {}
        self._test_order: list[int] = []
        self._tested: set[int] = set()
        self._target: Optional[tuple[int, int]] = None
        self._phase1_count = 0

    @property
    def name(self) -> str:
        return "BayesianExplorer"

    def reset(self):
        self._observations = defaultdict(list)
        self._action_model = {}
        self._test_order = list(range(self.n_actions))
        random.shuffle(self._test_order)
        self._tested = set()
        self._target = None
        self._phase1_count = 0

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)

        if phase == 1:
            return self._act_phase1(observation, info)
        return self._act_phase2(observation, info)

    def _act_phase1(self, obs: dict, info: dict | None) -> int:
        self._phase1_count += 1

        # Record effect of last action
        if info and self._phase1_count > 1:
            self._record_last_effect(obs)

        # Choose next action: untested first, then highest uncertainty
        if self._test_order:
            action = self._test_order.pop(0)
            self._tested.add(action)
            self._current_action = action
            self._pre_state = self._extract_state(info)
            return action

        # All tested — signal done if we have a good model
        if len(self._action_model) >= 2:
            return DONE_EXPLORING

        # Re-test the most uncertain action
        uncertainties = []
        for a in range(self.n_actions):
            n_obs = len(self._observations[a])
            uncertainties.append((1.0 / (n_obs + 1), a))
        uncertainties.sort(reverse=True)
        action = uncertainties[0][1]
        self._current_action = action
        self._pre_state = self._extract_state(info)
        return action

    def _act_phase2(self, obs: dict, info: dict | None) -> int:
        if not info:
            return 0

        if self._target is None:
            self._target = info.get("target_pos")

        cur_r, cur_c = info["agent_pos"]
        if self._target is None:
            # Try to parse target from text observation
            text = obs.get("text", "")
            if "Target:" in text:
                import re
                m = re.search(r"Target:\s*\((\d+),(\d+)\)", text)
                if m:
                    self._target = (int(m.group(1)), int(m.group(2)))

        if self._target is None:
            return 0

        tr, tc = self._target

        # Greedy: pick action whose learned (dr, dc) moves closest to target
        goal_dr = tr - cur_r
        goal_dc = tc - cur_c

        if goal_dr == 0 and goal_dc == 0:
            return 0

        best_action = 0
        best_score = -float("inf")

        for action, (dr, dc) in self._action_model.items():
            # Dot product alignment
            score = dr * goal_dr + dc * goal_dc
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _record_last_effect(self, obs: dict):
        """Record what happened after the last action."""
        if not hasattr(self, '_current_action') or not hasattr(self, '_pre_state'):
            return

        # Parse current state from feedback or observation text
        text = obs.get("text", "")
        feedback = obs.get("feedback", "")

        # Extract position change from feedback
        import re
        move_match = re.search(
            r"Position.*?\((\d+),(\d+)\).*?→.*?\((\d+),(\d+)\)", feedback
        )
        if move_match:
            pr, pc = int(move_match.group(1)), int(move_match.group(2))
            nr, nc = int(move_match.group(3)), int(move_match.group(4))
            dr, dc = nr - pr, nc - pc
            self._action_model[self._current_action] = (dr, dc)
            self._observations[self._current_action].append({
                "dr": dr, "dc": dc, "prev_pos": (pr, pc), "new_pos": (nr, nc),
            })
        elif "unchanged" in feedback:
            self._observations[self._current_action].append({"dr": 0, "dc": 0})

    def _extract_state(self, info: dict | None) -> dict:
        if not info:
            return {}
        return {
            "pos": info.get("agent_pos"),
            "dir": info.get("agent_dir"),
            "color": info.get("agent_color"),
        }

    def record_step(self, action: int, prev_pos: tuple, new_pos: tuple):
        """Called externally by run_episode to record observation."""
        dr = new_pos[0] - prev_pos[0]
        dc = new_pos[1] - prev_pos[1]
        if dr != 0 or dc != 0:
            self._action_model[action] = (dr, dc)
        self._observations[action].append({"dr": dr, "dc": dc})

    def set_target(self, target: tuple[int, int]):
        self._target = target


# ═══════════════════════════════════════════════════════════════════
#  MemoryAugmentedExplorer
# ═══════════════════════════════════════════════════════════════════

class MemoryAugmentedExplorer(Agent):
    """Explicit memory bank of (action, before, after) tuples.

    Phase 1: Systematically tests each action, storing full state transitions.
    Phase 2: Uses the learned transition model to simulate forward and plan
             a path via greedy + lookahead search.

    The key advantage over SystematicExplorer: handles direction-dependent
    effects (Type A) by tracking direction changes in the memory bank.
    """

    def __init__(self, n_actions: int = 4, lookahead: int = 3):
        self.n_actions = n_actions
        self.lookahead = lookahead
        self._memory: list[dict] = []
        self._action_effects: dict[int, list[dict]] = defaultdict(list)
        self._phase1_count = 0
        self._target: Optional[tuple[int, int]] = None
        self._plan: list[int] = []

    @property
    def name(self) -> str:
        return "MemoryAugmented"

    def reset(self):
        self._memory = []
        self._action_effects = defaultdict(list)
        self._phase1_count = 0
        self._target = None
        self._plan = []

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)

        if phase == 1:
            self._phase1_count += 1
            if self._phase1_count <= self.n_actions:
                return self._phase1_count - 1
            # Test each action twice for direction-dependent families
            if self._phase1_count <= self.n_actions * 2:
                return (self._phase1_count - 1) % self.n_actions
            return DONE_EXPLORING

        # Phase 2
        return self._act_phase2(info)

    def _act_phase2(self, info: dict | None) -> int:
        if not info or not self._target:
            return 0

        # If we have a plan, follow it
        if self._plan:
            return self._plan.pop(0)

        cur_r, cur_c = info["agent_pos"]
        cur_dir = info.get("agent_dir", 0)
        tr, tc = self._target

        # Build a plan using lookahead search
        self._plan = self._search_plan(cur_r, cur_c, cur_dir, tr, tc)
        if self._plan:
            return self._plan.pop(0)

        # Fallback: greedy by learned deltas
        return self._greedy_step(cur_r, cur_c, tr, tc)

    def _search_plan(self, sr: int, sc: int, sdir: int, tr: int, tc: int) -> list[int]:
        """BFS using learned transition model, limited depth."""
        from collections import deque

        best_plan = []
        best_dist = abs(sr - tr) + abs(sc - tc)

        queue = deque([(sr, sc, sdir, [], best_dist)])
        visited = {(sr, sc, sdir)}

        while queue:
            r, c, d, path, dist = queue.popleft()
            if len(path) >= self.lookahead:
                continue

            for action in range(self.n_actions):
                nr, nc, nd = self._simulate(action, r, c, d)

                new_dist = abs(nr - tr) + abs(nc - tc)
                if new_dist == 0:
                    return path + [action]

                if new_dist < best_dist:
                    best_dist = new_dist
                    best_plan = path + [action]

                key = (nr, nc, nd)
                if key not in visited:
                    visited.add(key)
                    queue.append((nr, nc, nd, path + [action], new_dist))

        return best_plan

    def _simulate(self, action: int, r: int, c: int, direction: int) -> tuple[int, int, int]:
        """Simulate an action using learned transition model."""
        effects = self._action_effects.get(action, [])
        if not effects:
            return r, c, direction

        # Use the most common effect for this action
        # Group by (dr, dc, ddir) and pick majority
        from collections import Counter
        deltas = Counter()
        for e in effects:
            key = (e.get("dr", 0), e.get("dc", 0), e.get("ddir", 0))
            deltas[key] += 1

        best_delta = deltas.most_common(1)[0][0]
        dr, dc, dd = best_delta
        return r + dr, c + dc, (direction + dd) % 4

    def _greedy_step(self, cr: int, cc: int, tr: int, tc: int) -> int:
        """Pick action that moves closest to target."""
        goal_dr = tr - cr
        goal_dc = tc - cc
        best_action = 0
        best_score = -float("inf")

        for action, effects in self._action_effects.items():
            if not effects:
                continue
            avg_dr = sum(e.get("dr", 0) for e in effects) / len(effects)
            avg_dc = sum(e.get("dc", 0) for e in effects) / len(effects)
            score = avg_dr * goal_dr + avg_dc * goal_dc
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def record_step(self, action: int, prev_pos: tuple, new_pos: tuple,
                    prev_dir: int = 0, new_dir: int = 0,
                    prev_color: int = 0, new_color: int = 0):
        """Called externally to record full state transition."""
        dr = new_pos[0] - prev_pos[0]
        dc = new_pos[1] - prev_pos[1]
        ddir = (new_dir - prev_dir) % 4
        if ddir == 3:
            ddir = -1

        record = {
            "action": action,
            "prev_pos": prev_pos, "new_pos": new_pos,
            "dr": dr, "dc": dc,
            "prev_dir": prev_dir, "new_dir": new_dir, "ddir": ddir,
            "prev_color": prev_color, "new_color": new_color,
        }
        self._memory.append(record)
        self._action_effects[action].append(record)

    def set_target(self, target: tuple[int, int]):
        self._target = target
