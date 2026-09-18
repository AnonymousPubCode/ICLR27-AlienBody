"""AFMB Stage 0: Active Forward-Model Babbling (no LLM).

Extends Enumerate+Verify with tie-aware experiment design:

  1. Systematic warm-start (test each action a few times).
  2. Enumerate type-C permutations; collect the max-score tie set H*.
  3. While |H*| > 1 and Phase-1 budget remains:
       - Prefer an action that *splits* H* at the current position
         (different hyps predict different next cells).
       - Else pick an action that (under a tied hyp) moves to a
         least-visited cell — cheap prep that diversifies observations.
  4. Phase 2: BFS over the best-scoring induced ForwardModel.

Gate G1 (TODO): on F4 n=8, AFMB must significantly beat Enumerate+Verify.
"""
from __future__ import annotations

import copy
from collections import Counter

from alienbody.agents import DONE_EXPLORING
from alienbody.agents.enumerate_agent import (
    EnumerateVerifyAgent, APPLIERS, RELATIONAL_SHORT, VOCABS,
)
from alienbody.env.grid import GridState, Position
from alienbody.schema import ActionSchema, EffectType, ForwardModel


class AFMBAgent(EnumerateVerifyAgent):
    """Active enum: break observational ties with discriminating probes.

    Modes (env AFMB_MODE or constructor):
      active   — full EIG probe + prep (default; Gate G1)
      passive  — systematic sweeps for full Phase-1 budget, no tie-breaking
                 (A2: same enum inducer, no active experiments)
      warm_only — stop after warm sweeps; induce immediately (fewer obs)
      no_eig   — after warm-start, continue systematic (ignore split/prep)
                 but still early-stop when |H*|=1 (A7: no discriminating choice)
    """

    def __init__(self, config, mode: str | None = None):
        super().__init__(config)
        import os
        self._mode = (mode or os.environ.get("AFMB_MODE", "active")).lower()
        self._warm_sweeps = int(os.environ.get("AFMB_WARM_SWEEPS", "2"))
        self._max_tie_log = 32

    @property
    def name(self) -> str:
        if self._mode == "active":
            return "AFMBAgent"
        return f"AFMBAgent[{self._mode}]"

    def reset(self):
        super().reset()
        self._p1_actions = 0
        self._visited_cells: Counter = Counter()
        self._tie_log: list[dict] = []
        self._last_n_ties = None
        self._cached_ties: list[tuple[str, ...]] | None = None
        self._cached_best_score: float | None = None
        self._ties_dirty = True
        self._max_relational_rounds = max(self._max_relational_rounds, 8)

    def record_step(self, action, prev_pos, new_pos, prev_dir=0, new_dir=0,
                    prev_color=0, new_color=0):
        super().record_step(action, prev_pos, new_pos, prev_dir, new_dir,
                            prev_color, new_color)
        self._visited_cells[tuple(new_pos)] += 1
        self._ties_dirty = True

    def _refresh_ties(self):
        """Full enum if no cache; else re-score cached ties and shrink H*."""
        if not self._ties_dirty and self._cached_ties is not None:
            return self._cached_best_score, (
                self._cached_ties[0] if self._cached_ties else None
            ), self._cached_ties

        if self._cached_ties is None or len(self._cached_ties) > 256:
            score, best, ties = self._best_and_ties("C")
        else:
            # Incremental: only rescore previous H* (new obs prune ties)
            scored = [(self._replay_score("C", perm), perm)
                      for perm in self._cached_ties]
            if not scored:
                score, best, ties = self._best_and_ties("C")
            else:
                score = max(s for s, _ in scored)
                ties = [p for s, p in scored if abs(s - score) <= 1e-6]
                best = ties[0] if ties else None
                # If everything pruned (model class wrong), full re-enum
                if not ties:
                    score, best, ties = self._best_and_ties("C")

        self._cached_best_score = score
        self._cached_ties = ties
        self._ties_dirty = False
        return score, best, ties

    def _act_phase1(self, obs: dict, info: dict | None) -> int:
        self._phase1_step += 1
        self._p1_actions += 1
        budget = getattr(self.config, "phase1_budget", 40)
        warm_steps = self.n_actions * self._warm_sweeps

        if self._p1_actions <= warm_steps:
            return (self._p1_actions - 1) % self.n_actions

        # A2 warm_only: induce on warm-start observations only
        if self._mode == "warm_only":
            return DONE_EXPLORING

        # A2 passive: burn full budget on systematic sweeps (no EIG)
        if self._mode == "passive":
            if self._p1_actions >= budget:
                return DONE_EXPLORING
            return (self._p1_actions - 1) % self.n_actions

        if not info:
            if self._p1_actions >= budget:
                return DONE_EXPLORING
            return (self._p1_actions - 1) % self.n_actions

        cur = tuple(info.get("agent_pos", (0, 0)))
        cur_dir = int(info.get("agent_dir", 0))
        score, best_perm, ties = self._refresh_ties()

        if best_perm is None:
            if self._p1_actions >= budget:
                return DONE_EXPLORING
            return (self._p1_actions - 1) % self.n_actions

        n_ties = len(ties)
        self._last_n_ties = n_ties
        if len(self._tie_log) < self._max_tie_log:
            self._tie_log.append({
                "step": self._p1_actions,
                "n_ties": n_ties,
                "best_score": score,
                "pos": cur,
                "mode": self._mode,
            })

        if n_ties <= 1 or self._p1_actions >= budget:
            return DONE_EXPLORING

        # A7 no_eig: keep sweeping systematically; do not pick split/prep
        if self._mode == "no_eig":
            return (self._p1_actions - 1) % self.n_actions

        return self._select_discriminating_action(cur, cur_dir, ties)

    # ── Experiment selection ─────────────────────────────────────

    def _predict_next(self, perm: tuple[str, ...], pos: tuple[int, int],
                      action: int, agent_dir: int = 0) -> tuple[int, int]:
        """Predict next (r,c) under a candidate mapping (cached, no deepcopy)."""
        # Reuse relational effect cache; seed with this probe position if needed.
        self._ensure_relational_cache()
        effect = perm[action]
        key = (pos[0], pos[1], effect)
        if key not in self._rel_effect_cache:
            from alienbody.env.actions import apply_type_c
            temp_cfg = copy.deepcopy(self.config)
            object.__setattr__(
                temp_cfg, "action_mapping", tuple([effect] * self.n_actions)
            )
            state = GridState(
                agent_pos=Position(pos[0], pos[1]), agent_dir=0, agent_color=0,
                phase=0, step_count=0, phase1_steps=0, phase2_steps=0,
                prev_action=-1,
            )
            new_state = apply_type_c(copy.deepcopy(state), temp_cfg, 0)
            self._rel_effect_cache[key] = (
                new_state.agent_pos.row, new_state.agent_pos.col
            )
        return self._rel_effect_cache[key]

    def _select_discriminating_action(
        self,
        cur: tuple[int, int],
        cur_dir: int,
        ties: list[tuple[str, ...]],
    ) -> int:
        """Pick action that maximizes predicted-outcome diversity under H*."""
        # Cap for speed; prefer first 48 ties (already max-score)
        if len(ties) > 48:
            ties = ties[:48]

        # Ensure cache covers current cell for all effects in H*
        self._ensure_relational_cache()
        effects_needed = {perm[a] for perm in ties for a in range(self.n_actions)}
        for effect in effects_needed:
            key = (cur[0], cur[1], effect)
            if key not in self._rel_effect_cache:
                self._predict_next(
                    tuple([effect] * self.n_actions), cur, 0, cur_dir
                )

        best_a, best_div, best_move_div = 0, -1, -1
        for a in range(self.n_actions):
            preds = [self._predict_next(perm, cur, a, cur_dir) for perm in ties]
            diversity = len(set(preds))
            moves = sum(1 for p in preds if p != cur)
            if (diversity > best_div
                    or (diversity == best_div and moves > best_move_div)):
                best_div, best_move_div, best_a = diversity, moves, a

        if best_div > 1:
            return best_a

        return self._select_prep_action(cur, cur_dir, ties[0])

    def _select_prep_action(
        self,
        cur: tuple[int, int],
        cur_dir: int,
        perm: tuple[str, ...],
    ) -> int:
        """Choose an action that reaches the least-visited predicted cell."""
        best_a, best_key = 0, None
        for a in range(self.n_actions):
            nxt = self._predict_next(perm, cur, a, cur_dir)
            # Prefer unvisited / rarely visited; slight preference for moving
            visit = self._visited_cells.get(nxt, 0)
            moved = 0 if nxt == cur else 1
            key = (-visit, moved)
            if best_key is None or key > best_key:
                best_key, best_a = key, a
        return best_a

    def get_induction_log(self) -> list[dict]:
        base = super().get_induction_log()
        return base + [{"tie_log": self._tie_log, "final_n_ties": self._last_n_ties}]


def apply_type_c_safe(state, config, action_idx: int):
    """Thin wrapper so AFMB does not import apply_type_c at module top only."""
    return APPLIERS["C"](copy.deepcopy(state), config, action_idx)
