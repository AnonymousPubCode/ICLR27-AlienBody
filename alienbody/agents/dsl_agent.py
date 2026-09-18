"""DSL blind-search and VLM-proposer agents (Method v3 decision experiment).

Both induce a list[Prog] mapping, then BFS with the DSL interpreter.

  - DSLBlindAgent: budgeted enumeration over atomic (+ optional depth-2) programs
  - DSLProposeAgent: VLM proposes programs; verifier scores against Phase-1 traj;
    AFMB-style disagreement probes when |H*|>1 (Stage-0: EIG over prog ties)
"""
from __future__ import annotations

import copy
import itertools
import json
import re
from collections import Counter, deque

from alienbody.agents import DONE_EXPLORING, Agent
from alienbody.agents.llm_agent import ModelClient
from alienbody.effect_dsl import (
    Prog, apply_prog_mapping, dsl_prompt_spec, interpret,
    iter_atomic_programs, iter_depth2_programs, normalize_prog_json,
    program_space_stats,
)
from alienbody.env.grid import EnvConfig, GridState, Phase, Position


def _traj_score(mapping: list[Prog], trajectory: list[dict], config: EnvConfig) -> float:
    if not trajectory:
        return -1.0
    score = 0.0
    for rec in trajectory:
        state = GridState(
            agent_pos=Position(*rec["prev_pos"]),
            agent_dir=rec.get("prev_dir", 0),
            agent_color=rec.get("prev_color", 0),
            phase=Phase.CALIBRATION, step_count=0, phase1_steps=0, phase2_steps=0,
            prev_action=rec.get("prev_action", -1),
            last_move_delta=tuple(rec["last_move_delta"]) if rec.get("last_move_delta") else None,
        )
        a = rec["action"]
        s2 = apply_prog_mapping(state, config, mapping, a)
        got = (s2.agent_pos.row, s2.agent_pos.col, s2.agent_dir)
        want = (rec["new_pos"][0], rec["new_pos"][1], rec.get("new_dir", got[2]))
        if (got[0], got[1]) == (want[0], want[1]):
            score += 1.0
            if got[2] == want[2]:
                score += 0.2
        elif abs(got[0] - want[0]) + abs(got[1] - want[1]) <= 1:
            score += 0.3
    return score


class _DSLPlanMixin:
    """Shared Phase-2 BFS over an induced Prog mapping."""

    config: EnvConfig
    _mapping: list[Prog] | None
    _plan: list[int]
    _trajectory: list[dict]
    _prev_act: int
    _last_lmd: tuple[int, int] | None

    def _reset_common(self):
        self._mapping = None
        self._plan = []
        self._trajectory = []
        self._prev_act = -1
        self._last_lmd = None
        self._p1 = 0
        self._visited = Counter()

    def record_step(self, action, prev_pos, new_pos, prev_dir=0, new_dir=0,
                    prev_color=0, new_color=0):
        if action is None or action < 0:
            return
        self._trajectory.append({
            "action": action,
            "prev_pos": tuple(prev_pos), "new_pos": tuple(new_pos),
            "prev_dir": prev_dir, "new_dir": new_dir,
            "prev_color": prev_color, "new_color": new_color,
            "prev_action": self._prev_act,
            "last_move_delta": self._last_lmd,
        })
        self._last_lmd = (
            new_pos[0] - prev_pos[0], new_pos[1] - prev_pos[1]
        )
        self._prev_act = action
        self._visited[tuple(new_pos)] += 1

    def _bfs(self, info: dict) -> list[int]:
        if not self._mapping:
            return []
        cur = info.get("agent_pos", self.config.agent_start)
        cur_r, cur_c = int(cur[0]), int(cur[1])
        cur_dir = int(info.get("agent_dir", 0))
        cur_color = int(info.get("agent_color", 0))
        pa = int(info.get("prev_action", -1))
        lmd = info.get("last_move_delta")
        if lmd is not None:
            lmd = tuple(lmd)
        target = Position(*self.config.target_pos)
        # Relational / large-n: position-only BFS keys (effects ignore history)
        simple = (
            self.config.n_actions >= 8
            or getattr(self.config, "action_type", "") == "C"
        )
        max_depth = 60 if simple else 40
        if simple:
            start = (cur_r, cur_c)
            q = deque([(start, cur_dir, cur_color, pa, lmd, [])])
        else:
            start = (cur_r, cur_c, cur_dir, cur_color, pa, lmd)
            q = deque([(start, [])])
        visited = {start}
        while q:
            if simple:
                (r, c), d, col, pa_, lmd_, path = q.popleft()
            else:
                (r, c, d, col, pa_, lmd_), path = q.popleft()
            if len(path) >= max_depth:
                continue
            for a in range(self.config.n_actions):
                st = GridState(
                    agent_pos=Position(r, c), agent_dir=d, agent_color=col,
                    phase=Phase.EXECUTION, step_count=0, phase1_steps=0, phase2_steps=0,
                    prev_action=pa_, last_move_delta=lmd_,
                )
                st2 = apply_prog_mapping(st, self.config, self._mapping, a)
                if st2.agent_pos == target:
                    return path + [a]
                if simple:
                    nk = (st2.agent_pos.row, st2.agent_pos.col)
                    if nk not in visited:
                        visited.add(nk)
                        q.append((nk, st2.agent_dir, st2.agent_color,
                                  st2.prev_action, st2.last_move_delta, path + [a]))
                else:
                    nk = (st2.agent_pos.row, st2.agent_pos.col, st2.agent_dir,
                          st2.agent_color, st2.prev_action, st2.last_move_delta)
                    if nk not in visited:
                        visited.add(nk)
                        q.append((nk, path + [a]))
        return []


class DSLBlindAgent(_DSLPlanMixin, Agent):
    """Budgeted blind search over the open DSL (no LLM)."""

    def __init__(self, config: EnvConfig, max_candidates: int = 2000,
                 allow_depth2: bool = False, phase1_sweeps: int = 3,
                 relational_first: bool = True):
        self.config = config
        self.max_candidates = max_candidates
        self.allow_depth2 = allow_depth2
        self.phase1_sweeps = phase1_sweeps
        self.relational_first = relational_first
        # XL / large-n: shrink pool so induction finishes
        if config.n_actions >= 8 and max_candidates > 80:
            self.max_candidates = 80
        self._reset_common()
        self._meta: dict = {}

    @property
    def name(self) -> str:
        return "DSLBlindAgent"

    def reset(self):
        self._reset_common()
        self._meta = {}

    def get_induction_log(self) -> list[dict]:
        return [self._meta] if self._meta else []

    def _candidate_pool(self) -> list[Prog]:
        pool = list(iter_atomic_programs())
        if self.relational_first:
            # Prioritize teleport / step_* (Relational) so small budgets still hit
            def _pri(p: Prog) -> int:
                op = p.to_json().get("op")
                return 0 if op in ("teleport", "step_toward", "step_away") else 1
            pool.sort(key=_pri)
        if self.allow_depth2:
            for p in iter_depth2_programs(limit=max(0, self.max_candidates - len(pool))):
                pool.append(p)
        return pool[: self.max_candidates]

    def _induce(self) -> list[Prog]:
        pool = self._candidate_pool()
        n = self.config.n_actions
        # Per-action: pick best program independently (greedy product).
        # Full joint search is |pool|^n — intractable; this is the honest
        # "blind local search" baseline a reviewer would accept.
        best_per_action: list[Prog] = []
        scores = []
        for a in range(n):
            best_p, best_s = pool[0], -1.0
            sub = [r for r in self._trajectory if r["action"] == a]
            for cand in pool:
                mapping = [Prog.from_json({"op": "noop"})] * n
                mapping[a] = cand
                s = _traj_score(mapping, sub, self.config) if sub else -1.0
                if s > best_s:
                    best_s, best_p = s, cand
            best_per_action.append(best_p)
            scores.append(best_s)
        joint = list(best_per_action)
        joint_score = _traj_score(joint, self._trajectory, self.config)
        # Joint refine only for small n (O(n·|pool|·|traj|) too slow on XL)
        if n <= 6:
            top = pool[: min(40, len(pool))]
            for a in range(n):
                for cand in top:
                    trial = list(joint)
                    trial[a] = cand
                    s = _traj_score(trial, self._trajectory, self.config)
                    if s > joint_score:
                        joint, joint_score = trial, s
        self._meta = {
            "kind": "dsl_blind",
            "pool": len(pool),
            "per_action_scores": scores,
            "joint_score": joint_score,
            "n_traj": len(self._trajectory),
            "space_stats": program_space_stats(),
        }
        return joint

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)
        if phase == 1:
            self._p1 += 1
            budget = self.config.n_actions * self.phase1_sweeps
            if self._p1 <= budget:
                return (self._p1 - 1) % self.config.n_actions
            if self._mapping is None:
                self._mapping = self._induce()
            return DONE_EXPLORING
        if self._mapping is None:
            self._mapping = self._induce()
        if not self._plan:
            self._plan = self._bfs(info or {})
        if self._plan:
            return self._plan.pop(0)
        return 0


_JSON_RE = re.compile(r"\{[\s\S]*\}")


class DSLProposeAgent(_DSLPlanMixin, Agent):
    """VLM proposes DSL programs; local verifier scores; optional active probes."""

    def __init__(
        self,
        config: EnvConfig,
        client: ModelClient,
        induce_rounds: int = 3,
        active: bool = True,
        warm_sweeps: int = 2,
        response_max_tokens: int = 1500,
    ):
        self.config = config
        self.client = client
        self.induce_rounds = induce_rounds
        self.active = active
        self.warm_sweeps = warm_sweeps
        self.response_max_tokens = response_max_tokens
        self._reset_common()
        self._log: list[dict] = []
        self._candidates: list[list[Prog]] = []

    @property
    def name(self) -> str:
        return f"DSLProposeAgent({self.client.name})"

    def reset(self):
        self._reset_common()
        self._log = []
        self._candidates = []

    def get_induction_log(self) -> list[dict]:
        return list(self._log)

    def _obs_table(self) -> str:
        lines = []
        for i, r in enumerate(self._trajectory[-40:]):
            lines.append(
                f"{i}: a={r['action']} ({r['prev_pos']})->({r['new_pos']}) "
                f"dir {r.get('prev_dir')}->{r.get('new_dir')}"
            )
        return "\n".join(lines) if lines else "(no observations yet)"

    def _parse_mapping(self, text: str) -> list[Prog] | None:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        progs = []
        for i in range(self.config.n_actions):
            key = str(i) if str(i) in obj else i
            if key not in obj:
                return None
            try:
                node = normalize_prog_json(obj[key])
                progs.append(Prog.from_json(node))
                # dry-run interpret on a dummy state to catch bad ops early
                st = GridState(
                    agent_pos=Position(*self.config.agent_start),
                    agent_dir=0, agent_color=0, phase=Phase.CALIBRATION,
                    step_count=0, phase1_steps=0, phase2_steps=0,
                )
                interpret(progs[-1], copy.deepcopy(st), self.config)
            except Exception:
                return None
        return progs

    def _propose_once(self, feedback: str | None = None) -> list[Prog] | None:
        examples = (
            "Examples of good Relational programs:\n"
            '  {"op":"teleport","sel":{"metric":"manhattan","agg":"min","pred":"diff","scope":"grid"}}\n'
            '  {"op":"teleport","sel":{"metric":"manhattan","agg":"min","pred":"same","scope":"grid"}}\n'
            '  {"op":"step_toward","sel":{"metric":"color","agg":"max","pred":"any","scope":"neighbors"}}\n'
            '  {"op":"step_away","sel":{"metric":"manhattan","agg":"min","pred":"same","scope":"grid"}}\n'
        )
        prompt = (
            f"{dsl_prompt_spec()}\n\n{examples}\n"
            f"n_actions={self.config.n_actions}, grid={self.config.grid_size}.\n"
            f"Observations (Phase-1):\n{self._obs_table()}\n\n"
        )
        if feedback:
            prompt += f"Verifier feedback:\n{feedback}\n\n"
        prompt += "Return ONLY the JSON object."
        messages = [
            {"role": "system", "content": "You synthesize executable action programs."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.complete(messages, max_tokens=self.response_max_tokens)
        mapping = self._parse_mapping(resp or "")
        self._log.append({
            "round": len(self._log),
            "raw_preview": (resp or "")[:400],
            "parsed": mapping is not None,
            "score": _traj_score(mapping, self._trajectory, self.config) if mapping else None,
        })
        return mapping

    def _induce_from_llm(self) -> list[Prog]:
        best, best_s = None, -1.0
        feedback = None
        for _ in range(self.induce_rounds):
            m = self._propose_once(feedback)
            if m is None:
                feedback = "Could not parse JSON / invalid op. Resubmit valid DSL."
                continue
            s = _traj_score(m, self._trajectory, self.config)
            self._candidates.append(m)
            if s > best_s:
                best, best_s = m, s
            # Build mismatch feedback
            mismatches = []
            for rec in self._trajectory[:12]:
                st = GridState(
                    agent_pos=Position(*rec["prev_pos"]),
                    agent_dir=rec.get("prev_dir", 0),
                    agent_color=rec.get("prev_color", 0),
                    phase=Phase.CALIBRATION, step_count=0, phase1_steps=0, phase2_steps=0,
                    prev_action=rec.get("prev_action", -1),
                )
                s2 = apply_prog_mapping(st, self.config, m, rec["action"])
                got = (s2.agent_pos.row, s2.agent_pos.col)
                want = tuple(rec["new_pos"])
                if got != want:
                    mismatches.append(
                        f"a={rec['action']} from {rec['prev_pos']}: pred {got} != obs {want}"
                    )
            if not mismatches:
                break
            feedback = "\n".join(mismatches[:8])
        if best is None:
            # fallback: all noop
            best = [Prog.from_json({"op": "noop"}) for _ in range(self.config.n_actions)]
        self._log.append({"final_score": best_s, "n_candidates": len(self._candidates)})
        return best

    def _select_probe(self, cur: tuple[int, int], ties: list[list[Prog]]) -> int:
        best_a, best_div = 0, -1
        for a in range(self.config.n_actions):
            preds = set()
            for mapping in ties[:24]:
                st = GridState(
                    agent_pos=Position(*cur), agent_dir=0, agent_color=0,
                    phase=Phase.CALIBRATION, step_count=0, phase1_steps=0, phase2_steps=0,
                )
                s2 = apply_prog_mapping(st, self.config, mapping, a)
                preds.add((s2.agent_pos.row, s2.agent_pos.col))
            if len(preds) > best_div:
                best_div, best_a = len(preds), a
        return best_a

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)
        budget = getattr(self.config, "phase1_budget", 40)
        warm = self.config.n_actions * self.warm_sweeps

        if phase == 1:
            self._p1 += 1
            if self._p1 <= warm:
                return (self._p1 - 1) % self.config.n_actions

            # After warm-start, induce once; then optionally probe
            if self._mapping is None:
                self._mapping = self._induce_from_llm()

            if not self.active or self._p1 >= budget:
                return DONE_EXPLORING

            # Collect near-ties among candidates
            scored = [
                (_traj_score(m, self._trajectory, self.config), m)
                for m in self._candidates
            ]
            if not scored:
                return DONE_EXPLORING
            top = max(s for s, _ in scored)
            ties = [m for s, m in scored if abs(s - top) < 1e-6]
            if len(ties) <= 1:
                return DONE_EXPLORING
            cur = tuple((info or {}).get("agent_pos", self.config.agent_start))
            return self._select_probe(cur, ties)

        if self._mapping is None:
            self._mapping = self._induce_from_llm()
        # Re-score candidates after probes
        if self._candidates:
            best = max(
                self._candidates,
                key=lambda m: _traj_score(m, self._trajectory, self.config),
            )
            self._mapping = best
        if not self._plan:
            self._plan = self._bfs(info or {})
        if self._plan:
            return self._plan.pop(0)
        return 0


class DSLCatalogProposeAgent(_DSLPlanMixin, Agent):
    """VLM picks per-action programs from a numbered atomic catalog."""

    def __init__(self, config: EnvConfig, client: ModelClient, induce_rounds: int = 3,
                 warm_sweeps: int = 2, response_max_tokens: int = 800):
        self.config = config
        self.client = client
        self.induce_rounds = induce_rounds
        self.warm_sweeps = warm_sweeps
        self.response_max_tokens = response_max_tokens
        self._catalog = list(iter_atomic_programs())
        self._reset_common()
        self._log: list[dict] = []

    @property
    def name(self) -> str:
        return f"DSLCatalogProposeAgent({self.client.name})"

    def reset(self):
        self._reset_common()
        self._log = []

    def get_induction_log(self) -> list[dict]:
        return list(self._log)

    def _catalog_text(self) -> str:
        return "\n".join(
            f"  [{i}] {json.dumps(p.to_json(), separators=(',', ':'))}"
            for i, p in enumerate(self._catalog)
        )

    def _obs_table(self) -> str:
        lines = [
            f"{i}: a={r['action']} ({r['prev_pos']})->({r['new_pos']})"
            for i, r in enumerate(self._trajectory[-40:])
        ]
        return "\n".join(lines) if lines else "(none)"

    def _parse_indices(self, text: str) -> list[Prog] | None:
        m = _JSON_RE.search(text or "")
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        idxs = []
        if "actions" in obj and isinstance(obj["actions"], list):
            idxs = [int(x) for x in obj["actions"]]
        else:
            for i in range(self.config.n_actions):
                key = str(i) if str(i) in obj else i
                if key not in obj:
                    return None
                idxs.append(int(obj[key]))
        if len(idxs) != self.config.n_actions:
            return None
        if any(i < 0 or i >= len(self._catalog) for i in idxs):
            return None
        return [self._catalog[i] for i in idxs]

    def _propose(self, feedback: str | None = None) -> list[Prog] | None:
        prompt = (
            "Pick ONE catalog index per action that best explains the observations.\n"
            "Relational effects usually need teleport/step_* with color predicates, "
            "NOT plain translate.\n\n"
            f"Catalog:\n{self._catalog_text()}\n\n"
            f"n_actions={self.config.n_actions}\n"
            f"Observations:\n{self._obs_table()}\n\n"
        )
        if feedback:
            prompt += f"Feedback:\n{feedback}\n\n"
        prompt += 'Return ONLY JSON like {"0": 12, "1": 3, "2": 15, "3": 8}.'
        messages = [
            {"role": "system", "content": "You select programs from a fixed catalog."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.complete(messages, max_tokens=self.response_max_tokens)
        mapping = self._parse_indices(resp or "")
        score = _traj_score(mapping, self._trajectory, self.config) if mapping else None
        self._log.append({
            "round": len(self._log), "raw_preview": (resp or "")[:300],
            "parsed": mapping is not None, "score": score,
        })
        return mapping

    def _induce(self) -> list[Prog]:
        best, best_s = None, -1.0
        feedback = None
        for _ in range(self.induce_rounds):
            m = self._propose(feedback)
            if m is None:
                feedback = "Invalid JSON / out-of-range index. Resubmit."
                continue
            s = _traj_score(m, self._trajectory, self.config)
            if s > best_s:
                best, best_s = m, s
            mismatches = []
            for rec in self._trajectory[:10]:
                st = GridState(
                    agent_pos=Position(*rec["prev_pos"]),
                    agent_dir=rec.get("prev_dir", 0),
                    agent_color=rec.get("prev_color", 0),
                    phase=Phase.CALIBRATION, step_count=0, phase1_steps=0, phase2_steps=0,
                    prev_action=rec.get("prev_action", -1),
                )
                s2 = apply_prog_mapping(st, self.config, m, rec["action"])
                got = (s2.agent_pos.row, s2.agent_pos.col)
                want = tuple(rec["new_pos"])
                if got != want:
                    mismatches.append(
                        f"a={rec['action']} from {rec['prev_pos']}: pred {got} != obs {want}"
                    )
            if not mismatches:
                break
            feedback = "\n".join(mismatches[:8])
        if best is None:
            best = [Prog.from_json({"op": "noop"}) for _ in range(self.config.n_actions)]
        self._log.append({"final_score": best_s})
        return best

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)
        warm = self.config.n_actions * self.warm_sweeps
        if phase == 1:
            self._p1 += 1
            if self._p1 <= warm:
                return (self._p1 - 1) % self.config.n_actions
            if self._mapping is None:
                self._mapping = self._induce()
            return DONE_EXPLORING
        if self._mapping is None:
            self._mapping = self._induce()
        if not self._plan:
            self._plan = self._bfs(info or {})
        if self._plan:
            return self._plan.pop(0)
        return 0
