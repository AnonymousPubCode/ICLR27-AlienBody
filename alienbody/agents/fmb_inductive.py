"""Inductive FMB Agent: LLM-proposed schemas + simulator verification loop.

The trained FMB variants (LoRA/FullFT/CoT) fail to induce relational schemas
from babbling observations (induction frontier: 2%, below chance). The
heuristic FMBAgent succeeds on F4 by enumerating the 4! permutations of known
relation types and verifying each against a forward-model simulator (88%),
but fails on F5/F6 where the candidate space does not factorize the same way.

This agent is the *hybrid* point between the two extremes (cf. HYSYNTH:
pure-neural fails in unfamiliar DSLs, pure-symbolic does not scale; the fix
is LLM-guided search). The LLM proposes a complete action->effect mapping
from the Phase-1 observations; the mapping is verified against the true
transition function; mismatches are fed back to the LLM for revision
(induce_rounds iterations). A fully-verified mapping is handed to the
inherited BFS planner. If no round fully verifies and fallback is enabled,
we fall back to the heuristic permutation enumeration (guaranteeing F4
performance); otherwise the best-scoring proposal is used.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Optional

from alienbody.agents.fmb_agent import FMBAgent
from alienbody.agents.llm_agent import ModelClient
from alienbody.env.actions import apply_action, TYPE_C_EFFECTS
from alienbody.env.grid import EnvConfig, GridState, Position, Phase
from alienbody.schema import ActionSchema, EffectType, ForwardModel

# Relation vocabulary (F4): long names (LLM interface) → short names (schema)
RELATION_TYPES = {
    "move_to_nearest_diff_color": ("nearest_diff",
        "teleport to the NEAREST cell whose color is DIFFERENT from the cell you stand on"),
    "move_to_nearest_same_color": ("nearest_same",
        "teleport to the NEAREST cell whose color is the SAME as the cell you stand on"),
    "move_toward_brightest": ("toward_brightest",
        "move 1 step toward the brightest (highest color value) of your 4 neighbors"),
    "flee_same_color": ("flee_same",
        "move 1 step AWAY from the nearest cell whose color is the SAME as the cell you stand on"),
    # Tier-L extensions
    "move_to_farthest_same_color": ("farthest_same",
        "teleport to the FARTHEST cell whose color is the SAME as the cell you stand on"),
    "move_away_from_brightest": ("away_brightest",
        "move 1 step AWAY from the brightest (highest color value) of your 4 neighbors"),
    # Tier-XL extensions
    "move_to_farthest_diff_color": ("farthest_diff",
        "teleport to the FARTHEST cell whose color is DIFFERENT from the cell you stand on"),
    "flee_nearest_diff_color": ("flee_nearest_diff",
        "move 1 step AWAY from the nearest cell whose color is DIFFERENT from the cell you stand on"),
    "move_toward_darkest": ("toward_darkest",
        "move 1 step toward the darkest (lowest color value) of your 4 neighbors"),
    "move_away_from_darkest": ("away_darkest",
        "move 1 step AWAY from the darkest (lowest color value) of your 4 neighbors"),
    "move_to_nearest_brighter": ("nearest_brighter",
        "teleport to the NEAREST cell whose color is HIGHER than the cell you stand on"),
    "move_to_nearest_darker": ("nearest_darker",
        "teleport to the NEAREST cell whose color is LOWER than the cell you stand on"),
}

# Composite vocabulary (F5): combo name → (short, description, schema primitives)
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
COMPOSITE_TYPES = {
    "move_and_rotate_cw": ("move_and_rotate_cw",
        "move FORWARD one step (in the facing direction), then rotate CLOCKWISE 90 degrees",
        _COMPOSITE_PRIMS["move_and_rotate_cw"]),
    "rotate_and_strafe": ("rotate_and_strafe",
        "rotate COUNTER-CLOCKWISE 90 degrees, then move FORWARD one step (in the NEW facing direction)",
        _COMPOSITE_PRIMS["rotate_and_strafe"]),
    "double_forward": ("double_forward",
        "move FORWARD two steps (in the facing direction)",
        _COMPOSITE_PRIMS["double_forward"]),
    "retreat_and_spin": ("retreat_and_spin",
        "move BACKWARD one step, then rotate CLOCKWISE 180 degrees (two CW rotations)",
        _COMPOSITE_PRIMS["retreat_and_spin"]),
}

# Temporal vocabulary (F6): type name → (short, description, schema params)
TEMPORAL_TYPES = {
    "temporal_0": ("temporal_0",
        "if SAME action as the previous step: move UP; if DIFFERENT from previous step: move RIGHT",
        {"same_dir": 0, "diff_dir": 1}),
    "temporal_1": ("temporal_1",
        "if SAME action as the previous step: move DOWN; if DIFFERENT from previous step: move LEFT",
        {"same_dir": 2, "diff_dir": 3}),
    "temporal_2": ("temporal_2",
        "REPEAT the movement of the previous step (same direction and distance)",
        {"repeat": True}),
    "temporal_3": ("temporal_3",
        "INVERT the movement of the previous step (opposite direction)",
        {"inverse": True}),
}

_PRIMITIVE_PARAMS = {"forward": {"direction": "forward"}, "backward": {"direction": "backward"}}

_SYSTEM_INDUCTION = """You are identifying what {n_actions} anonymous actions do in a grid world.
Each action is exactly one of the following types:

{vocab}

Below are the grid's cell colors (numbers 0-9; 0 = black) and the transitions
observed while testing the actions. Propose the complete mapping.

Grid colors (row, col -> color):
{grid}

Observed transitions (position -> action -> new position, with the color of the
cell the agent stood on):
{observations}

Output ONLY a JSON object mapping action index to type name, e.g.
{{"0": "move_to_nearest_same_color", "1": "move_toward_brightest", ...}}
Your answer:"""

_FEEDBACK_INDUCTION = """Your mapping predicted {score}/{total} transitions correctly. Here are the mismatches:

{mismatches}

Revise the mapping. Output ONLY a JSON object mapping action index to type name.
Your answer:"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# Fallback for fenced/truncated responses: pull "k": "name" pairs directly
_KV_RE = re.compile(r'"\s*(\d+)\s*"\s*:\s*"([a-z_]+)"')


class InductiveFMBAgent(FMBAgent):
    """LLM proposes action schemas; a simulator verifies; mismatch feedback
    drives iterative revision. Inherits Phase-1 exploration and Phase-2 BFS
    planning from FMBAgent."""

    def __init__(self, config: EnvConfig, client: ModelClient,
                 induce_rounds: int = 4, fallback: bool = True):
        super().__init__(config)
        self._client = client
        self.induce_rounds = induce_rounds
        self.fallback = fallback
        self._induction_log: list[dict] = []  # per-episode trace for analysis
        # Sequential observation trace (preserves global order + prev_action
        # for temporal verification)
        self._obs_sequence: list[dict] = []
        self._last_action: int = -1

    @property
    def name(self) -> str:
        return f"InductiveFMBAgent(r={self.induce_rounds},fb={self.fallback})"

    def reset(self):
        super().reset()
        self._induction_log = []
        self._obs_sequence = []
        self._last_action = -1

    def record_step(self, action: int, prev_pos: tuple, new_pos: tuple, **kw):
        super().record_step(action, prev_pos, new_pos, **kw)
        self._obs_sequence.append({
            "action": action, "prev_pos": prev_pos, "new_pos": new_pos,
            "prev_action": self._last_action,
            "prev_dir": kw.get("prev_dir", 0),
        })
        self._last_action = action

    def _act_phase1(self, obs: dict, info: dict | None) -> int:
        """Uniform round-robin exploration to the budget (fair across
        families: provides prev-action diversity for F6 and position
        diversity for F4/F5)."""
        self._phase1_step += 1
        if self._phase1_step <= self.config.phase1_budget:
            return (self._phase1_step - 1) % self.n_actions
        return DONE_EXPLORING

    # ── Vocabulary selection ──────────────────────────────────────

    def _vocab(self) -> dict:
        at = self.config.action_type
        if at == "C":
            # Tier-M (n=4) environments were generated from the first 4
            # relation types; only expose those (exposing the Tier-L
            # extensions biases proposals toward impossible effects).
            return dict(list(RELATION_TYPES.items())[:self.n_actions])
        if at == "E":
            return COMPOSITE_TYPES
        if at == "F":
            return TEMPORAL_TYPES
        return RELATION_TYPES

    # ── Induction override ─────────────────────────────────────────

    def _induce_forward_model(self) -> ForwardModel:
        if self.induce_rounds <= 0:
            return self._llm_propose_once()

        vocab = self._vocab()
        system_prompt = _SYSTEM_INDUCTION.format(
            n_actions=self.n_actions,
            vocab="\n".join(f"  {name}: {info[1]}" for name, info in vocab.items()),
            grid=self._render_color_grid(),
            observations=self._format_observations(),
        )
        # The proposal prompt is sent as a user message: the Leihuo gateway's
        # Gemini upstream rejects system-only conversations ("at least one
        # contents field is required"), and user-message instructions are
        # equivalent for all three backends.
        messages = [{"role": "user", "content": system_prompt}]

        best_mapping: Optional[dict[int, str]] = None
        best_score = -1

        for round_idx in range(self.induce_rounds):
            response = self._client.complete(messages, max_tokens=4096)
            mapping = self._parse_mapping(response or "")
            score, mismatches = self._score_mapping(mapping)
            verified = bool(mapping) and score == self._n_observations()
            self._induction_log.append({
                "round": round_idx, "response": (response or "")[:200],
                "mapping": mapping, "score": score, "verified": verified,
            })
            if mapping and score > best_score:
                best_mapping, best_score = mapping, score
            if verified:
                # Fully verified: build the model directly
                return self._mapping_to_model(mapping)
            if round_idx < self.induce_rounds - 1:
                messages.append({"role": "assistant", "content": response or ""})
                messages.append({"role": "user", "content": _FEEDBACK_INDUCTION.format(
                    score=score, total=self._n_observations(),
                    mismatches=self._format_mismatches(mismatches),
                )})

        if self.fallback:
            # Keep the LLM proposal when it is good enough (>= 80% of
            # observations); otherwise fall back to the heuristic induction
            # (permutation enumeration for F4, pattern matching elsewhere).
            n_obs = self._n_observations()
            if best_mapping is not None and n_obs > 0 and best_score >= 0.8 * n_obs:
                self._induction_log.append({"round": -1, "fallback": False,
                                            "kept_llm": True,
                                            "best_score": best_score})
                return self._mapping_to_model(best_mapping)
            self._induction_log.append({"round": -1, "fallback": True,
                                        "best_score": best_score})
            return super()._induce_forward_model()
        return self._mapping_to_model(best_mapping) if best_mapping else ForwardModel()

    def _llm_propose_once(self) -> ForwardModel:
        """Zero-shot proposal with no verification feedback (matrix baseline)."""
        vocab = self._vocab()
        system_prompt = _SYSTEM_INDUCTION.format(
            n_actions=self.n_actions,
            vocab="\n".join(f"  {name}: {info[1]}" for name, info in vocab.items()),
            grid=self._render_color_grid(),
            observations=self._format_observations(),
        )
        response = self._client.complete(
            [{"role": "user", "content": system_prompt}], max_tokens=4096)
        mapping = self._parse_mapping(response or "")
        score, _ = self._score_mapping(mapping)
        self._induction_log.append({"round": 0, "response": (response or "")[:200],
                                    "mapping": mapping, "score": score})
        return self._mapping_to_model(mapping) if mapping else ForwardModel()

    # ── Observation utilities ─────────────────────────────────────

    def _observations_flat(self) -> list[tuple[int, int, int, int, int, int]]:
        """[(action, prev_r, prev_c, new_r, new_c, cell_color_at_prev)]."""
        out = []
        for a in range(self.n_actions):
            for rec in self._observations.get(a, []):
                pr, pc = rec.get("prev_pos", (0, 0))
                nr, nc = rec.get("new_pos", (0, 0))
                gs = self.config.grid_size
                if 0 <= pr < gs and 0 <= pc < gs:
                    out.append((a, pr, pc, nr, nc,
                                self.config.cell_colors[pr][pc]))
        return out

    def _n_observations(self) -> int:
        return len(self._obs_sequence)

    def _format_observations(self) -> str:
        lines = []
        for obs in self._obs_sequence:
            pr, pc = obs["prev_pos"]
            nr, nc = obs["new_pos"]
            gs = self.config.grid_size
            col = self.config.cell_colors[pr][pc] if 0 <= pr < gs and 0 <= pc < gs else "?"
            prev = obs.get("prev_action", -1)
            prev_str = f"prev={prev}, " if prev >= 0 else ""
            facing = obs.get("prev_dir", 0)
            lines.append(
                f"  at ({pr},{pc}) color={col} facing={facing}, {prev_str}"
                f"Action {obs['action']} -> ({nr},{nc})")
        return "\n".join(lines) if lines else "  (no observations)"

    def _render_color_grid(self) -> str:
        """Compact color matrix (row col -> value)."""
        gs = self.config.grid_size
        lines = []
        for r in range(gs):
            row = " ".join(str(self.config.cell_colors[r][c]) for c in range(gs))
            lines.append(f"  row {r:2d}: {row}")
        return "\n".join(lines)

    # ── Proposal parsing / scoring ─────────────────────────────────

    def _parse_mapping(self, response: str) -> Optional[dict[int, str]]:
        """Extract a JSON mapping {action_idx: type_name}, with a
        key-value-pair fallback for fenced or truncated responses."""
        vocab = self._vocab()

        def _finish(raw: dict) -> Optional[dict[int, str]]:
            mapping = {}
            for k, v in raw.items():
                try:
                    a = int(str(k).strip())
                except ValueError:
                    continue
                if a < 0 or a >= self.n_actions:
                    continue
                name = str(v).strip()
                if name in vocab:
                    mapping[a] = name
            return mapping if len(mapping) == self.n_actions else None

        m = _JSON_RE.search(response)
        if m:
            try:
                parsed = _finish(json.loads(m.group(0)))
                if parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        # KV fallback (handles ```json fences and truncation)
        pairs = {}
        for k, v in _KV_RE.findall(response):
            a = int(k)
            if 0 <= a < self.n_actions and v in vocab:
                pairs[a] = v
        return _finish(pairs)

    def _score_mapping(self, mapping: Optional[dict[int, str]]) -> tuple[int, list]:
        """Verify the proposal against the true transition function, replaying
        the observation sequence in order (prev_action matters for F6).

        Returns (n_correct, mismatches) where each mismatch is a dict.
        """
        if not mapping:
            return 0, []
        correct = 0
        mismatches = []
        for obs in self._obs_sequence:
            a = obs["action"]
            pr, pc = obs["prev_pos"]
            nr, nc = obs["new_pos"]
            state = GridState(agent_pos=Position(pr, pc), agent_color=0,
                              agent_dir=obs.get("prev_dir", 0),
                              phase=Phase.EXECUTION,
                              step_count=0, phase1_steps=0, phase2_steps=0,
                              prev_action=obs.get("prev_action", -1))
            temp_cfg = copy.deepcopy(self.config)
            object.__setattr__(temp_cfg, 'action_mapping',
                               tuple(mapping.get(i, "x")
                                     for i in range(self.n_actions)))
            new_state = apply_action(copy.deepcopy(state), temp_cfg, a)
            pr_r, pr_c = new_state.agent_pos.row, new_state.agent_pos.col
            if (pr_r, pr_c) == (nr, nc):
                correct += 1
            else:
                mismatches.append({
                    "action": a, "at": (pr, pc), "actual": (nr, nc),
                    "predicted": (pr_r, pr_c),
                })
        return correct, mismatches

    def _format_mismatches(self, mismatches: list) -> str:
        if not mismatches:
            return "  (none)"
        return "\n".join(
            f"  at {m['at']}, Action {m['action']}: predicted {m['predicted']}, "
            f"actual {m['actual']}" for m in mismatches[:15]
        )

    def _mapping_to_model(self, mapping: dict[int, str]) -> ForwardModel:
        model = ForwardModel()
        at = self.config.action_type
        vocab = self._vocab()
        for a in range(self.n_actions):
            name = mapping.get(a)
            if name not in vocab:
                model.actions[a] = ActionSchema(EffectType.NOOP)
                continue
            if at == "C":
                model.actions[a] = ActionSchema(
                    EffectType.RELATIONAL, {"relation": vocab[name][0]})
            elif at == "E":
                model.actions[a] = ActionSchema(
                    EffectType.COMPOSITE, {"primitives": vocab[name][2]})
            elif at == "F":
                model.actions[a] = ActionSchema(
                    EffectType.TEMPORAL, dict(vocab[name][2]))
        return model

    def get_induction_log(self) -> list[dict]:
        return list(self._induction_log)
