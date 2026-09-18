"""Enumerate+Verify baseline (FM_enum) — the no-LLM answer to "why not just
enumerate?".

For each family's effect vocabulary (relational / composite / temporal,
4! = 24 permutations each at Tier-M), enumerate every action→effect mapping,
verify each candidate by exact sequential replay against the REAL transition
function (the same env primitives the environment itself uses), and plan with
the best-scoring mapping via the standard BFS. No LLM, no hand-written family
heuristics — pure search over the hypothesis space + verification.

Runs with the same Phase-1 exploration budget as FM_heur (systematic round +
retest rounds), so comparisons are like-for-like.
"""
from __future__ import annotations

import copy
import itertools

from alienbody.agents.fmb_agent import FMBAgent
from alienbody.env.actions import (
    apply_type_c, apply_type_e, apply_type_f,
    TYPE_C_EFFECTS, TYPE_E_EFFECTS, TYPE_F_EFFECTS,
)
from alienbody.env.grid import GridState, Position
from alienbody.schema import ActionSchema, EffectType, ForwardModel

_COMPOSITE_PRIMS = {
    "move_and_rotate_cw": [
        {"type": "translate", "params": {"direction": "forward"}},
        {"type": "rotate", "params": {"rotation": "cw"}}],
    "rotate_and_strafe": [
        {"type": "rotate", "params": {"rotation": "ccw"}},
        {"type": "translate", "params": {"direction": "forward"}}],
    "double_forward": [
        {"type": "translate", "params": {"direction": "forward"}},
        {"type": "translate", "params": {"direction": "forward"}}],
    "retreat_and_spin": [
        {"type": "translate", "params": {"direction": "backward"}},
        {"type": "rotate", "params": {"rotation": "cw"}},
        {"type": "rotate", "params": {"rotation": "cw"}}],
}
_TEMPORAL_PARAMS = {
    "temporal_0": {"same_dir": 0, "diff_dir": 1},
    "temporal_1": {"same_dir": 2, "diff_dir": 3},
    "temporal_2": {"repeat": True},
    "temporal_3": {"inverse": True},
}
RELATIONAL_SHORT = {
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

APPLIERS = {"C": apply_type_c, "E": apply_type_e, "F": apply_type_f}
VOCABS = {"C": TYPE_C_EFFECTS, "E": TYPE_E_EFFECTS, "F": TYPE_F_EFFECTS}


class EnumerateVerifyAgent(FMBAgent):
    """FM_enum: permutation enumeration + exact replay verification."""

    def reset(self):
        super().reset()
        self._trajectory = []       # ordered records
        self._prev_act = -1         # action before the current record
        import os
        self._max_relational_rounds = int(os.environ.get(
            "ENUM_ROUNDS", "2"))    # same Phase-1 budget as FM_heur (12 steps)
        self._rel_effect_cache = {}
        self._rel_cache_traj_len = -1
        self._last_induce_meta = None

    def record_step(self, action, prev_pos, new_pos, prev_dir=0, new_dir=0,
                    prev_color=0, new_color=0):
        super().record_step(action, prev_pos, new_pos, prev_dir, new_dir,
                            prev_color, new_color)
        if action is not None and action >= 0:
            self._trajectory.append({
                "action": action,
                "prev_pos": prev_pos, "new_pos": new_pos,
                "prev_dir": prev_dir, "new_dir": new_dir,
                "prev_action": self._prev_act,
            })
            self._prev_act = action

    # ── Enumeration + verification ────────────────────────────────

    def _replay_score(self, kind: str, perm: tuple[str, ...]) -> float:
        """Exact sequential replay: apply the candidate mapping to the real
        transition function and count matched (pos, dir) outcomes."""
        if not self._trajectory:
            return -1.0

        # Fast path for relational (type C): effects are Markovian in position,
        # so score independent transitions via a one-time effect cache.
        if kind == "C":
            return self._replay_score_relational_cached(perm)

        temp_cfg = copy.deepcopy(self.config)
        object.__setattr__(temp_cfg, "action_mapping", tuple(perm))
        applier = APPLIERS[kind]

        first = self._trajectory[0]
        state = GridState(
            agent_pos=Position(first["prev_pos"][0], first["prev_pos"][1]),
            agent_dir=first["prev_dir"], agent_color=0,
            phase=1, step_count=0, phase1_steps=0, phase2_steps=0,
            prev_action=-1)

        score = 0.0
        for rec in self._trajectory:
            state.prev_action = rec["prev_action"]
            s2 = applier(copy.deepcopy(state), temp_cfg, rec["action"])
            got = (s2.agent_pos.row, s2.agent_pos.col, s2.agent_dir)
            want = (rec["new_pos"][0], rec["new_pos"][1], rec["new_dir"])
            if (got[0], got[1]) == (want[0], want[1]):
                score += 1.0
                if got[2] == want[2]:
                    score += 0.2
            elif abs(got[0] - want[0]) + abs(got[1] - want[1]) <= 1:
                score += 0.3
            state = s2
        return score

    def _ensure_relational_cache(self):
        """Cache (pos, effect_name) → next_pos for all traj positions × vocab."""
        traj_len = len(self._trajectory)
        if getattr(self, "_rel_cache_traj_len", -1) == traj_len:
            return
        from alienbody.env.actions import apply_type_c, TYPE_C_EFFECTS
        vocab = list(TYPE_C_EFFECTS[: self.n_actions])
        cache: dict[tuple, tuple[int, int]] = {}
        gs = self.config.grid_size
        for rec in self._trajectory:
            pr, pc = rec["prev_pos"]
            if not (0 <= pr < gs and 0 <= pc < gs):
                continue
            for effect in vocab:
                key = (pr, pc, effect)
                if key in cache:
                    continue
                temp_cfg = copy.deepcopy(self.config)
                object.__setattr__(
                    temp_cfg, "action_mapping", tuple([effect] * self.n_actions)
                )
                state = GridState(
                    agent_pos=Position(pr, pc), agent_dir=0, agent_color=0,
                    phase=0, step_count=0, phase1_steps=0, phase2_steps=0,
                    prev_action=-1,
                )
                new_state = apply_type_c(copy.deepcopy(state), temp_cfg, 0)
                cache[key] = (new_state.agent_pos.row, new_state.agent_pos.col)
        self._rel_effect_cache = cache
        self._rel_cache_traj_len = traj_len

    def _replay_score_relational_cached(self, perm: tuple[str, ...]) -> float:
        self._ensure_relational_cache()
        cache = self._rel_effect_cache
        score = 0.0
        for rec in self._trajectory:
            pr, pc = rec["prev_pos"]
            effect = perm[rec["action"]]
            got = cache.get((pr, pc, effect))
            if got is None:
                continue
            want = (rec["new_pos"][0], rec["new_pos"][1])
            if got == want:
                score += 1.0
            elif abs(got[0] - want[0]) + abs(got[1] - want[1]) <= 1:
                score += 0.3
        return score

    # n! cost: 6!=720, 8!=40320 (fast with relational cache), 10!=3.6e6 (too slow).
    # Tier-XL n=12 remains intractable — skip there.
    _MAX_ENUM_N = 8

    def _iter_scored_perms(self, kind: str = "C"):
        """Yield (score, kind, perm) for every permutation of the kind vocab."""
        vocab_full = VOCABS[kind]
        if self.n_actions > self._MAX_ENUM_N:
            return
        vocab = list(vocab_full[: self.n_actions])
        if len(vocab) < self.n_actions:
            return
        for perm in itertools.permutations(vocab):
            yield self._replay_score(kind, perm), kind, perm

    def _best_and_ties(self, kind: str = "C", score_tol: float = 1e-6):
        """Return (best_score, best_perm, tie_perms) for a vocabulary kind.

        tie_perms includes every perm within score_tol of the best score
        (the under-identification set H* when |tie_perms|>1).
        """
        best_score = None
        best_perm = None
        ties: list[tuple[str, ...]] = []
        for score, _, perm in self._iter_scored_perms(kind):
            if best_score is None or score > best_score + score_tol:
                best_score = score
                best_perm = perm
                ties = [perm]
            elif abs(score - best_score) <= score_tol:
                ties.append(perm)
        return best_score, best_perm, ties

    def _induce_forward_model(self) -> ForwardModel:
        best = None  # (score, kind, perm)
        for kind in VOCABS:
            if kind == "C" and self.n_actions > self._MAX_ENUM_N:
                continue  # 12! (and 10!) intractable; Tier-XL point
            for score, k, perm in self._iter_scored_perms(kind):
                if best is None or score > best[0]:
                    best = (score, k, perm)

        model = ForwardModel()
        if best is None:
            return model
        _, kind, perm = best
        self._last_induce_meta = {
            "kind": kind,
            "score": best[0],
            "n_actions": self.n_actions,
        }
        # Attach tie size for relational (diagnostic for under-identification)
        if kind == "C":
            _, _, ties = self._best_and_ties("C")
            self._last_induce_meta["n_ties"] = len(ties)
        for a, effect in enumerate(perm):
            if kind == "C":
                model.actions[a] = ActionSchema(
                    EffectType.RELATIONAL,
                    {"relation": RELATIONAL_SHORT[effect]})
            elif kind == "E":
                model.actions[a] = ActionSchema(
                    EffectType.COMPOSITE,
                    {"primitives": _COMPOSITE_PRIMS[effect]})
            else:
                model.actions[a] = ActionSchema(
                    EffectType.TEMPORAL, dict(_TEMPORAL_PARAMS[effect]))
        return model

    def get_induction_log(self) -> list[dict]:
        meta = getattr(self, "_last_induce_meta", None)
        return [meta] if meta else []
