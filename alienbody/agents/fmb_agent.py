"""FMB Agent: Forward-Model Babbling (heuristic baseline).

Phase 1 (Active Babbling): Multi-round exploration with family-aware testing.
  Round 1: test each action once → detect family type
  Round 2 (if ego-centric): rotate 90°, re-test movement actions
  Round 2 (if state-dependent): cycle colors, re-test state actions

Phase 2 (Plan-over-Model): BFS using ForwardModelSimulator with induced T-hat.

See fmb_vlm.py for VLM-powered schema induction variant.
"""
from __future__ import annotations

import copy
from collections import defaultdict, deque
from typing import Optional

from alienbody.agents import Agent, DONE_EXPLORING
from alienbody.env.grid import GridState, EnvConfig, Position, Phase, DIRECTION_DELTAS
from alienbody.schema import (
    ForwardModel, ActionSchema, EffectType, ForwardModelSimulator,
)


class FMBAgent(Agent):
    """Heuristic FMB: family-aware multi-round exploration + pattern-matching induction."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.n_actions = config.n_actions

        self._observations: dict[int, list[dict]] = defaultdict(list)
        self._forward_model: Optional[ForwardModel] = None
        self._simulator: Optional[ForwardModelSimulator] = None

        # Phase 1: two rounds
        self._phase1_step = 0
        self._round = 1                                    # 1 or 2
        self._tested_once: set[int] = set()                # actions tested in round 1
        self._need_round2: bool = False
        self._round2_queue: list[int] = []                 # actions to re-test in round 2
        self._round2_mode: str = ""                        # "ego", "color", "relational", "unknown"
        self._rotation_action: Optional[int] = None        # which action rotates CW
        self._color_inc_action: Optional[int] = None        # which action increments color
        self._max_relational_rounds: int = 2               # max rounds for F4/unknown families
        self._relational_round_count: int = 0              # current round counter

        # Phase 2
        self._plan: list[int] = []
        self._target: Optional[tuple[int, int]] = None

    @property
    def name(self) -> str:
        return "FMBAgent"

    def reset(self):
        self._observations = defaultdict(list)
        self._forward_model = None
        self._simulator = None
        self._phase1_step = 0
        self._round = 1
        self._tested_once = set()
        self._need_round2 = False
        self._round2_queue = []
        self._round2_mode = ""
        self._rotation_action = None
        self._color_inc_action = None
        self._max_relational_rounds = 2
        self._relational_round_count = 0
        self._plan = []
        self._target = None

    # ── Phase 1 ──────────────────────────────────────────────────

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)
        if phase == 1:
            return self._act_phase1(observation, info)
        return self._act_phase2(observation, info)

    def _act_phase1(self, obs: dict, info: dict | None) -> int:
        self._phase1_step += 1

        # Round 1: test each action exactly once
        if self._round == 1:
            if self._phase1_step <= self.n_actions:
                return self._phase1_step - 1  # Action 0, 1, 2, 3

            # Round 1 complete — analyze and decide whether round 2 is needed
            self._analyze_round1()
            if self._need_round2 and self._round2_queue:
                self._round = 2
                self._phase1_step = 0
                # fall through to round 2
            else:
                return DONE_EXPLORING

        # Round 2+: re-test actions from different positions
        if self._round >= 2:
            if self._round2_queue:
                return self._round2_queue.pop(0)

            # Queue exhausted — continue for multi-round families
            if self._round2_mode in ("relational", "unknown"):
                self._relational_round_count += 1
                if self._relational_round_count < self._max_relational_rounds:
                    # Refill queue for another round
                    for a in range(self.n_actions):
                        self._round2_queue.append(a)
                    return self._round2_queue.pop(0)

            return DONE_EXPLORING

        return DONE_EXPLORING

    def _analyze_round1(self):
        """After round 1 testing, detect family type and plan round 2 if needed."""
        # Detect rotation actions (F2 ego-centric)
        for action in range(self.n_actions):
            for rec in self._observations.get(action, []):
                if rec.get("ddir", 0) != 0 and rec.get("dr", 0) == 0 and rec.get("dc", 0) == 0:
                    if rec.get("ddir", 0) in (1, -3):
                        self._rotation_action = action
                if rec.get("dcolor", 0) in (1, -3):
                    self._color_inc_action = action

        # Detect F4 relational: any action with large jump (teleport)
        has_jump = any(
            abs(rec.get("dr", 0)) > 1 or abs(rec.get("dc", 0)) > 1
            for action in range(self.n_actions)
            for rec in self._observations.get(action, [])
        )

        # Family F2: ego-centric (has rotation + movement actions)
        if self._rotation_action is not None:
            self._need_round2 = True
            self._round2_mode = "ego"
            # Re-test movement actions after rotating
            self._round2_queue = [self._rotation_action]  # rotate first
            for a in range(self.n_actions):
                if a != self._rotation_action:
                    self._round2_queue.append(a)

        # Family F3: state-dependent (has color changes)
        elif self._color_inc_action is not None:
            self._need_round2 = True
            self._round2_mode = "color"
            # Cycle color, then re-test all actions
            self._round2_queue = [self._color_inc_action]  # change color first
            for a in range(self.n_actions):
                if a != self._color_inc_action:
                    self._round2_queue.append(a)

        # Family F4: relational (has teleport actions)
        elif has_jump:
            self._need_round2 = True
            self._round2_mode = "relational"
            # F4 needs many observations from diverse positions for permutation
            # disambiguation. With budget=20, we can do 4 extra rounds (16 more steps).
            self._max_relational_rounds = 5  # total rounds (1 initial + 4 extra)
            self._relational_round_count = 1
            for a in range(self.n_actions):
                self._round2_queue.append(a)

        # Unidentified family with positional effects: re-test for disambiguation
        # (covers F4 where jumps are small, F5 composite, F6 temporal)
        elif not self._need_round2:
            has_position_change = any(
                rec.get("dr", 0) != 0 or rec.get("dc", 0) != 0
                for action in range(self.n_actions)
                for rec in self._observations.get(action, [])
            )
            if has_position_change:
                self._need_round2 = True
                self._round2_mode = "unknown"
                # Use full budget: 4 extra rounds = 16 more observations
                self._max_relational_rounds = 5
                self._relational_round_count = 1
                for a in range(self.n_actions):
                    self._round2_queue.append(a)

    # ── Phase 2 ──────────────────────────────────────────────────

    def _act_phase2(self, obs: dict, info: dict | None) -> int:
        if not info:
            return 0

        if self._forward_model is None:
            self._forward_model = self._induce_forward_model()
            self._simulator = ForwardModelSimulator(self._forward_model, self.config)

        if self._target is None:
            self._target = info.get("target_pos", self.config.target_pos)

        if self._plan:
            return self._plan.pop(0)

        if self._simulator and self._forward_model.is_complete(self.n_actions):
            self._plan = self._bfs_plan_over_model(info)
            if self._plan:
                return self._plan.pop(0)

        return self._greedy_fallback(info)

    # ── Schema Induction ─────────────────────────────────────────

    def _induce_forward_model(self) -> ForwardModel:
        model = ForwardModel()
        for action in range(self.n_actions):
            records = self._observations.get(action, [])
            model.actions[action] = self._infer_schema(action, records) if records else ActionSchema(EffectType.NOOP)

        # Fallback: if we're in "unknown" mode with many observations and the
        # actions show position-DEPENDENT effects (different deltas from different
        # positions), try permutation enumeration. This catches F4 environments
        # where relational actions produce only small moves.
        # F1 (cardinal) has position-INDEPENDENT effects: same (dr,dc) always.
        if self._round2_mode == "unknown" and self._max_relational_rounds >= 5:
            # Check for position-dependent effects
            has_position_dependence = False
            for action in range(self.n_actions):
                records = self._observations.get(action, [])
                if len(records) >= 2:
                    deltas = set()
                    for rec in records:
                        dr, dc = rec.get("dr", 0), rec.get("dc", 0)
                        if dr != 0 or dc != 0:
                            deltas.add((dr, dc))
                    if len(deltas) >= 2:
                        has_position_dependence = True
                        break

            if has_position_dependence:
                from alienbody.env.actions import TYPE_C_EFFECTS
                perm_model = ForwardModel()
                for action in range(self.n_actions):
                    records = self._observations.get(action, [])
                    if records:
                        perm_model.actions[action] = self._infer_relational_schema(action, records)
                    else:
                        perm_model.actions[action] = ActionSchema(EffectType.NOOP)
                model = perm_model

        return model

    def _infer_schema(self, action: int, records: list[dict]) -> ActionSchema:
        if not records:
            return ActionSchema(EffectType.NOOP)

        # ── Family detection from all observations ───────────────
        all_recs = []
        for a in range(self.n_actions):
            all_recs.extend(self._observations.get(a, []))

        has_rotation = any(r.get("ddir", 0) != 0 and r.get("dr", 0) == 0 and r.get("dc", 0) == 0
                          for r in all_recs)
        has_color = any(r.get("dcolor", 0) != 0 for r in all_recs)
        has_jump = any(abs(r.get("dr", 0)) > 1 or abs(r.get("dc", 0)) > 1 for r in all_recs)

        # ── Ego-centric detection (F2) ──────────────────────────
        dir_effects = defaultdict(set)
        for rec in records:
            dr, dc = rec.get("dr", 0), rec.get("dc", 0)
            if dr != 0 or dc != 0:
                dir_effects[rec.get("prev_dir", 0)].add((dr, dc))

        if len(dir_effects) > 1:
            facing_to_expected = {
                0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            facing_to_bkw = {
                0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}
            fwd = sum(1 for d, deltas in dir_effects.items()
                      if facing_to_expected.get(d) in deltas)
            bkw = sum(1 for d, deltas in dir_effects.items()
                      if facing_to_bkw.get(d) in deltas)
            if fwd > bkw:
                return ActionSchema(EffectType.TRANSLATE, {"direction": "forward"})
            elif bkw > 0:
                return ActionSchema(EffectType.TRANSLATE, {"direction": "backward"})

        # ── Color-dependent detection (F3) ──────────────────────
        color_effects = defaultdict(set)
        for rec in records:
            dr, dc = rec.get("dr", 0), rec.get("dc", 0)
            if dr != 0 or dc != 0:
                color_effects[rec.get("prev_color", 0)].add((dr, dc))

        if len(color_effects) > 1:
            return ActionSchema(EffectType.STATE_CHANGE, {"change": "color_move"})

        # ── Single-effect classification ────────────────────────
        effects = defaultdict(list)
        for rec in records:
            effects[(rec.get("dr", 0), rec.get("dc", 0),
                     rec.get("ddir", 0), rec.get("dcolor", 0))].append(rec)

        if len(effects) == 1:
            (dr, dc, ddir, dcolor) = list(effects.keys())[0]

            if dcolor != 0 and dr == 0 and dc == 0:
                return ActionSchema(EffectType.STATE_CHANGE,
                    {"change": "inc" if dcolor in (1, -3) else "dec"})
            if ddir != 0 and dr == 0 and dc == 0:
                return ActionSchema(EffectType.ROTATE,
                    {"rotation": "cw" if ddir in (1, -3) else "ccw"})
            if abs(dr) <= 1 and abs(dc) <= 1:
                # Only route to permutation for confirmed relational mode (big jumps detected)
                if self._round2_mode == "relational":
                    return self._infer_relational_schema(action, records)
                if has_rotation:
                    return ActionSchema(EffectType.TRANSLATE, {"direction": "forward"})
                return ActionSchema(EffectType.TRANSLATE,
                    {"direction": self._delta_to_direction(dr, dc)})
            if abs(dr) > 1 or abs(dc) > 1:
                # F4 relational: big jump → nearest_diff or nearest_same
                # Use cell colors at start position to disambiguate
                return self._infer_relational_schema(action, records)

        # ── Multi-effect fallback ────────────────────────────────
        if len(effects) >= 2:
            if self._round2_mode == "relational":
                return self._infer_relational_schema(action, records)
            has_move = any(abs(k[0]) + abs(k[1]) > 0 for k in effects)
            has_rot = any(k[2] != 0 for k in effects)
            if has_move and has_rot:
                return ActionSchema(EffectType.COMPOSITE, {"primitives": [
                    {"type": "translate", "params": {"direction": "forward"}},
                    {"type": "rotate", "params": {"rotation": "cw"}}]})
            return ActionSchema(EffectType.TEMPORAL, {"same_dir": 0, "diff_dir": 1})

        if self._round2_mode == "relational":
            return self._infer_relational_schema(action, records)
        return ActionSchema(EffectType.TRANSLATE, {"direction": "up"})

    def _infer_relational_schema(self, action: int, records: list[dict]) -> ActionSchema:
        """F4 relational induction via permutation enumeration + ground-truth simulation.

        Enumerate all 4! = 24 possible action→relation mappings.
        For each candidate, call the REAL apply_type_c to simulate outcomes
        and score by prediction accuracy. Returns best-matching relation
        for the specified action.

        Key: uses apply_type_c directly (not a hand-written simulation),
        guaranteeing the predictions match the environment's actual behavior.
        """
        from alienbody.env.actions import apply_type_c, TYPE_C_EFFECTS
        from alienbody.env.grid import GridState, Position, EnvConfig
        import itertools
        import copy

        if self.n_actions > 6:
            # n! enumeration intractable (Tier-XL: 12! ~ 4.8e8) -> degrade to
            # the positional-delta heuristic; the honest behavior at scale.
            return ActionSchema(EffectType.RELATIONAL, {"relation": "nearest_diff"})

        gs = self.config.grid_size

        # Effect vocabulary: the first n_actions effects. Tier-M (n=4)
        # environments were generated when TYPE_C_EFFECTS had exactly 4
        # entries, so the slice reproduces the historical vocabulary; Tier-L
        # (n=6) uses the full list. (Permuting the full 6-effect vocabulary
        # for n=4 envs under-determines the fit and collapses induction.)
        effects_vocab = list(TYPE_C_EFFECTS[:self.n_actions])

        # Collect all observations: (action_idx, prev_pos, new_pos)
        observations = []
        for a in range(self.n_actions):
            for rec in self._observations.get(a, []):
                pr, pc = rec.get("prev_pos", (0, 0))
                nr, nc = rec.get("new_pos", (0, 0))
                if 0 <= pr < gs and 0 <= pc < gs:
                    observations.append((a, pr, pc, nr, nc))

        if not observations:
            return ActionSchema(EffectType.RELATIONAL, {"relation": "nearest_diff"})

        # Pre-compute for each observation position, what each effect does
        # Cache: (pr, pc, effect_name) → (nr, nc)
        # Use object.__setattr__ because EnvConfig is @dataclass(frozen=True)
        effect_cache = {}
        for _, pr, pc, _, _ in observations:
            for effect in effects_vocab:
                key = (pr, pc, effect)
                if key not in effect_cache:
                    temp_cfg = copy.deepcopy(self.config)
                    object.__setattr__(temp_cfg, 'action_mapping', tuple([effect] * self.n_actions))
                    state = GridState(
                        agent_pos=Position(pr, pc), agent_dir=0, agent_color=0,
                        phase=0, step_count=0, phase1_steps=0, phase2_steps=0,
                        prev_action=-1)
                    new_state = apply_type_c(copy.deepcopy(state), temp_cfg, 0)
                    effect_cache[key] = (new_state.agent_pos.row, new_state.agent_pos.col)

        # Enumerate all n! permutations and score
        best_score = -1
        best_mapping = None

        for perm in itertools.permutations(effects_vocab):
            # perm[i] = effect assigned to action i
            score = 0
            for a, pr, pc, nr, nc in observations:
                pred = effect_cache.get((pr, pc, perm[a]), (pr, pc))
                if pred == (nr, nc):
                    score += 1
                elif abs(pred[0] - nr) + abs(pred[1] - nc) <= 1:
                    score += 0.3

            if score > best_score:
                best_score = score
                best_mapping = perm

        if best_mapping is None:
            return ActionSchema(EffectType.RELATIONAL, {"relation": "nearest_diff"})

        # best_mapping[i] is the full effect name like "move_to_nearest_diff_color"
        # Convert to short relation name for ActionSchema params
        full_name = best_mapping[action]
        short_map = {
            "move_to_nearest_diff_color": "nearest_diff",
            "move_to_nearest_same_color": "nearest_same",
            "move_toward_brightest": "toward_brightest",
            "flee_same_color": "flee_same",
            "move_to_farthest_same_color": "farthest_same",
            "move_away_from_brightest": "away_brightest",
            "move_to_farthest_diff_color": "farthest_diff",
            "flee_nearest_diff_color": "flee_nearest_diff",
            "move_toward_darkest": "toward_darkest",
            "move_away_from_darkest": "away_darkest",
            "move_to_nearest_brighter": "nearest_brighter",
            "move_to_nearest_darker": "nearest_darker",
        }
        relation = short_map.get(full_name, "nearest_diff")
        return ActionSchema(EffectType.RELATIONAL, {"relation": relation})

    # ── BFS Planning ─────────────────────────────────────────────

    def _bfs_plan_over_model(self, info: dict) -> list[int]:
        if not self._simulator:
            return []

        cur_pos = info.get("agent_pos", (0, 0))
        cur_r, cur_c = cur_pos if isinstance(cur_pos, tuple) else (0, 0)
        cur_dir = info.get("agent_dir", 0)
        cur_color = info.get("agent_color", 0)
        prev_act = info.get("prev_action", -1)
        cur_lmd = info.get("last_move_delta")
        if cur_lmd is not None:
            cur_lmd = tuple(cur_lmd)
        target = info.get("target_pos", self.config.target_pos)
        tr, tc = target if isinstance(target, tuple) else (0, 0)

        start = GridState(
            agent_pos=Position(cur_r, cur_c), agent_color=cur_color,
            agent_dir=cur_dir, phase=Phase.EXECUTION,
            step_count=0, phase1_steps=0, phase2_steps=0,
            prev_action=prev_act, last_move_delta=cur_lmd)

        queue = deque([(start, [])])
        visited = {(cur_r, cur_c, cur_dir, cur_color, prev_act, cur_lmd)}

        while queue:
            state, path = queue.popleft()
            if len(path) >= 40:
                continue
            for a in range(self.n_actions):
                ns = copy.deepcopy(state)
                ns = self._simulator.predict(ns, a)
                ns.prev_action = a
                nk = (ns.agent_pos.row, ns.agent_pos.col,
                      ns.agent_dir, ns.agent_color, ns.prev_action,
                      ns.last_move_delta)
                if ns.agent_pos.row == tr and ns.agent_pos.col == tc:
                    return path + [a]
                if nk not in visited:
                    visited.add(nk)
                    queue.append((ns, path + [a]))
        return []

    def _greedy_fallback(self, info: dict) -> int:
        cur = info.get("agent_pos", (0, 0))
        target = info.get("target_pos", (0, 0))
        tr, tc = target if isinstance(target, tuple) else (0, 0)
        cr, cc = cur if isinstance(cur, tuple) else (0, 0)
        best_a, best_s = 0, -float("inf")
        for a in range(self.n_actions):
            recs = self._observations.get(a, [])
            if not recs: continue
            adr = sum(r.get("dr", 0) for r in recs) / len(recs)
            adc = sum(r.get("dc", 0) for r in recs) / len(recs)
            s = adr * (tr - cr) + adc * (tc - cc)
            if s > best_s:
                best_s, best_a = s, a
        return best_a

    @staticmethod
    def _delta_to_direction(dr: int, dc: int) -> str:
        return {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}.get((dr, dc), "up")

    def record_step(self, action: int, prev_pos: tuple, new_pos: tuple,
                    prev_dir: int = 0, new_dir: int = 0,
                    prev_color: int = 0, new_color: int = 0):
        dr = new_pos[0] - prev_pos[0]
        dc = new_pos[1] - prev_pos[1]
        ddir = (new_dir - prev_dir) % 4
        if ddir == 3: ddir = -1
        self._observations[action].append({
            "dr": dr, "dc": dc, "ddir": ddir,
            "dcolor": new_color - prev_color,
            "prev_pos": prev_pos, "new_pos": new_pos,
            "prev_dir": prev_dir, "new_dir": new_dir,
            "prev_color": prev_color, "new_color": new_color,
        })

    def set_target(self, target: tuple[int, int]):
        self._target = target
