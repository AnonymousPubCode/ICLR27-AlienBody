"""VLM-powered FMB: Zero-Shot Schema Induction.

Replaces the heuristic pattern-matching in FMBAgent with VLM inference.
The VLM receives observation history → proposes structured ForwardModel (T-hat).
No training needed — pure prompt engineering + existing API/VLLM clients.

Two variants:
  VLMFMBAgent — full VLM-powered: exploration + induction + planning
  VLMInductionWrapper — wraps an existing agent, adds VLM induction at Phase 1→2
"""
from __future__ import annotations

import json
import re
from typing import Optional

from alienbody.agents import Agent, DONE_EXPLORING
from alienbody.agents.fmb_agent import FMBAgent
from alienbody.agents.llm_agent import make_agent, ModelClient
from alienbody.schema import ForwardModel, ActionSchema, EffectType, ForwardModelSimulator


# ── Schema Induction Prompt ──────────────────────────────────────

def _pos_to_dir(r1: int, c1: int, r2: int, c2: int) -> str:
    """Convert a position delta to a compass direction label."""
    dr, dc = r2 - r1, c2 - c1
    if abs(dr) >= abs(dc):
        return "N" if dr < 0 else "S"
    else:
        return "W" if dc < 0 else "E"

_SCHEMA_INDUCTION_SYSTEM = """You are an expert at reverse-engineering action systems from observations.
Given a history of action→effect observations from an unknown action space,
your task is to infer the underlying action schema.

The environment has {n_actions} anonymous actions (Action 0 through {max_action}).
Each action belongs to one of these effect types:

1. "translate" — moves the agent in a direction
   params: {{"direction": "up"|"down"|"left"|"right"|"forward"|"backward"}}
   "forward"/"backward" are relative to the agent's facing direction (red arrow).

2. "rotate" — changes the agent's facing direction
   params: {{"rotation": "cw"|"ccw"}}

3. "state_change" — modifies the agent's internal state (color)
   params: {{"change": "inc"|"dec"|"color_move"|"color_interact"}}
   "color_move" = move in a direction determined by agent's color (0=up,1=right,2=down,3=left)
   "color_interact" = if agent color matches cell color → move forward, else → rotate cw

4. "relational" — moves relative to colored cells
   params: {{"relation": "nearest_diff"|"nearest_same"|"toward_brightest"|"flee_same"}}

5. "composite" — chains two primitives
   params: {{"primitives": [{{"type": "...", "params": {{...}}}}, ...]}}

6. "temporal" — effect depends on previous action
   params: {{"same_dir": int, "diff_dir": int}} or {{"repeat": true}} or {{"inverse": true}}

7. "noop" — unknown / no visible effect

Output ONLY a valid JSON object mapping action indices to schemas:
{{"0": {{"effect_type": "translate", "params": {{"direction": "up"}}}}, ...}}

Think step by step about what you observe, then output the JSON."""


def build_induction_prompt(history: list[dict], n_actions: int) -> str:
    """Build prompt for VLM schema induction from observation history."""
    lines = ["Here are the observations from Phase 1 exploration:\n"]

    for action in range(n_actions):
        records = [h for h in history if h.get("action") == action]
        if not records:
            lines.append(f"Action {action}: NOT TESTED")
            continue

        lines.append(f"Action {action}: tested {len(records)} time(s)")
        for rec in records:
            parts = []
            if rec.get("position_changed"):
                pr, pc = rec.get("prev_pos", (0,0))
                nr, nc = rec.get("new_pos", (0,0))
                dr, dc = nr - pr, nc - pc
                delta_to_dir = {(-1,0):"up",(1,0):"down",(0,-1):"left",(0,1):"right"}
                direction = delta_to_dir.get((dr,dc), f"({dr:+d},{dc:+d})")
                dist = abs(dr) + abs(dc)
                if dist > 2:
                    parts.append(f"TELEPORTED to ({nr},{nc}) via {direction}")
                else:
                    parts.append(f"moved {direction}")
            if rec.get("direction_changed"):
                dd = (rec.get("new_dir",0) - rec.get("prev_dir",0)) % 4
                rot = "cw" if dd == 1 else "ccw"
                parts.append(f"rotated {rot}")
            if rec.get("color_changed"):
                parts.append(f"color {rec['prev_color']}->{rec['new_color']}")
            if not parts:
                parts.append("no visible effect")

            conditions = []
            if rec.get("prev_color", 0) != 0:
                conditions.append(f"color={rec['prev_color']}")

            # F4 relational context: cell-color-aware hints
            ctx = rec.get("_f4_context", {})
            if ctx:
                cond_parts = []
                if ctx.get("agent_cell_color", 0) != 0:
                    cond_parts.append(f"cell={ctx['agent_cell_color']}")
                if ctx.get("nearest_diff_dir"):
                    cond_parts.append(f"nearest_diff={ctx['nearest_diff_dir']}")
                if ctx.get("nearest_same_dir"):
                    cond_parts.append(f"nearest_same={ctx['nearest_same_dir']}")
                if ctx.get("brightest_dir"):
                    cond_parts.append(f"brightest_nbr={ctx['brightest_dir']}")
                if ctx.get("flee_dir"):
                    cond_parts.append(f"flee_dir={ctx['flee_dir']}")
                if cond_parts:
                    conditions.append("grid: " + ", ".join(cond_parts))

            cond_str = f"[{'; '.join(conditions)}] " if conditions else ""
            lines.append(f"  {cond_str}{'; '.join(parts)}")
        lines.append("")

    lines.append(f"Based on these observations, infer the schema for all {n_actions} actions.")
    lines.append("Output ONLY the JSON object.")
    return "\n".join(lines)


def parse_schema_response(response: str, n_actions: int) -> ForwardModel:
    """Parse VLM response into ForwardModel. Robust to markdown code blocks."""
    # Extract JSON from response (may be wrapped in ```json ... ```)
    json_match = re.search(r'\{[^{}]*"effect_type"[^{}]*\}', response, re.DOTALL)
    if not json_match:
        # Try to find any JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

    if json_match:
        try:
            raw = json.loads(json_match.group(0))
            # Handle nested object vs flat
            if any(isinstance(v, dict) and "effect_type" in v for v in raw.values()):
                # Already in correct format: {"0": {...}, "1": {...}}
                pass
            else:
                # Single action schema — wrap it
                raw = {"0": raw}
        except json.JSONDecodeError:
            raw = {}
    else:
        raw = {}

    model = ForwardModel()
    for key, value in raw.items():
        try:
            action_id = int(key)
            effect_type = EffectType(value.get("effect_type", "noop"))
            params = value.get("params", {})
            model.actions[action_id] = ActionSchema(effect_type=effect_type, params=params)
        except (ValueError, KeyError):
            continue

    return model


# ── VLM FMB Agent ────────────────────────────────────────────────

class VLMFMBAgent(FMBAgent):
    """FMB with VLM-powered schema induction (zero-shot, no training).

    Phase 1: Uses a separate LLM agent for exploration (systematic or VLM-driven).
    Phase 1→2: Calls VLM to induce ForwardModel from observation history.
    Phase 2: BFS plan over VLM-induced T-hat.

    The VLM induction can use any ModelClient (GPT-4o via the gateway, Qwen via vLLM, etc.).
    """

    def __init__(self, config, induction_client: ModelClient,
                 explore_agent_name: str = "systematic",
                 n_phase1_rounds: int = 1):
        """
        Args:
            config: EnvConfig for the environment
            induction_client: ModelClient for schema induction (e.g. GatewayClient, VLLMClient)
            explore_agent_name: agent to use for Phase 1 exploration
                                "systematic" = test each action once
                                "double" = test each action twice (for direction-dependent)
            n_phase1_rounds: number of exploration rounds (1=basic, 2=double-test)
        """
        super().__init__(config)
        self._induction_client = induction_client
        self._explore_agent_name = explore_agent_name
        self._n_phase1_rounds = n_phase1_rounds

        # Phase 1 sub-agent
        self._explore_agent: Optional[Agent] = None
        self._induction_done = False

    @property
    def name(self) -> str:
        return f"VLMFMB({self._induction_client.name})"

    def reset(self):
        super().reset()
        self._explore_agent = None
        self._induction_done = False

    def _act_phase1(self, obs: dict, info: dict | None) -> int:
        """Phase 1 with VLM-directed or systematic exploration."""
        self._phase1_step += 1

        if self._explore_agent is None:
            # Initialize exploration sub-agent
            if self._explore_agent_name == "systematic":
                self._explore_agent = _SystematicSubAgent(self.n_actions)
            elif self._explore_agent_name == "double":
                self._explore_agent = _DoubleTestSubAgent(self.n_actions)
            else:
                self._explore_agent = _SystematicSubAgent(self.n_actions)

        # Let the sub-agent choose the action
        action = self._explore_agent.act(obs, info)
        if action == DONE_EXPLORING:
            # Trigger VLM induction
            if not self._induction_done:
                self._forward_model = self._vlm_induce()
                self._simulator = ForwardModelSimulator(self._forward_model, self.config)
                self._induction_done = True
            return DONE_EXPLORING

        return action

    def _vlm_induce(self) -> ForwardModel:
        """Call VLM to induce ForwardModel from observation history."""
        # Build history from observations, with F4 color context
        history = []
        for action in range(self.n_actions):
            for rec in self._observations.get(action, []):
                entry = {
                    "action": action,
                    "position_changed": rec.get("dr", 0) != 0 or rec.get("dc", 0) != 0,
                    "direction_changed": rec.get("ddir", 0) != 0,
                    "color_changed": rec.get("dcolor", 0) != 0,
                    "prev_pos": rec.get("prev_pos", (0, 0)),
                    "new_pos": rec.get("new_pos", (0, 0)),
                    "prev_dir": rec.get("prev_dir", 0),
                    "new_dir": rec.get("new_dir", 0),
                    "prev_color": rec.get("prev_color", 0),
                    "new_color": rec.get("new_color", 0),
                }

                # Compute F4 relational context: cell-color-aware hints
                prev_pos = entry["prev_pos"]
                pr, pc = prev_pos
                gs = self.config.grid_size
                if 0 <= pr < gs and 0 <= pc < gs:
                    try:
                        from alienbody.env.actions import _find_nearest_by_color
                        from alienbody.env.grid import DIRECTION_DELTAS

                        agent_cell_color = self.config.cell_colors[pr][pc]
                        f4_ctx = {"agent_cell_color": agent_cell_color}

                        # nearest different-color cell direction
                        nd = _find_nearest_by_color(
                            type('Pos', (), {'row': pr, 'col': pc})(),
                            self.config, same=False, ref_color=agent_cell_color)
                        if nd:
                            f4_ctx["nearest_diff_dir"] = _pos_to_dir(pr, pc, nd.row, nd.col)

                        # nearest same-color cell direction
                        ns = _find_nearest_by_color(
                            type('Pos', (), {'row': pr, 'col': pc})(),
                            self.config, same=True, ref_color=agent_cell_color)
                        if ns:
                            f4_ctx["nearest_same_dir"] = _pos_to_dir(pr, pc, ns.row, ns.col)

                        # brightest 4-neighbor direction
                        obstacles = set()
                        if hasattr(self.config, 'obstacles') and self.config.obstacles:
                            obstacles = {(r, c) for r, c in self.config.obstacles}
                        best_color = -1
                        best_d = -1
                        for d_idx in range(4):
                            delta = DIRECTION_DELTAS[d_idx]
                            nb_r, nb_c = pr + delta[0], pc + delta[1]
                            if 0 <= nb_r < gs and 0 <= nb_c < gs and (nb_r, nb_c) not in obstacles:
                                nc_val = self.config.cell_colors[nb_r][nb_c]
                                if nc_val > best_color:
                                    best_color = nc_val
                                    best_d = d_idx
                        if best_d >= 0:
                            f4_ctx["brightest_dir"] = {0: "N", 1: "E", 2: "S", 3: "W"}[best_d]

                        # flee direction: away from nearest same-color cell
                        if ns:
                            flee_dr = pr - ns.row
                            flee_dc = pc - ns.col
                            if abs(flee_dr) >= abs(flee_dc):
                                step = (1 if flee_dr > 0 else -1, 0)
                            else:
                                step = (0, 1 if flee_dc > 0 else -1)
                            f4_ctx["flee_dir"] = _pos_to_dir(pr, pc, pr + step[0], pc + step[1])

                        entry["_f4_context"] = f4_ctx
                    except Exception:
                        pass  # config lacks cell_colors or env not F4

                history.append(entry)

        prompt = build_induction_prompt(history, self.n_actions)
        system = _SCHEMA_INDUCTION_SYSTEM.format(
            n_actions=self.n_actions,
            max_action=self.n_actions - 1,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._induction_client.complete(messages, max_tokens=512)
            model = parse_schema_response(response, self.n_actions)
            return model
        except Exception:
            # Fallback to heuristic induction
            return self._induce_forward_model()


# ── Phase 1 Sub-Agents ───────────────────────────────────────────

class _SystematicSubAgent:
    """Test each action once, then signal done."""
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._count = 0

    def act(self, obs: dict, info: dict | None) -> int:
        if self._count < self.n_actions:
            action = self._count
            self._count += 1
            return action
        return DONE_EXPLORING

    def reset(self):
        self._count = 0


class _DoubleTestSubAgent:
    """Test each action twice, then signal done."""
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._count = 0

    def act(self, obs: dict, info: dict | None) -> int:
        if self._count < self.n_actions * 2:
            action = self._count % self.n_actions
            self._count += 1
            return action
        return DONE_EXPLORING

    def reset(self):
        self._count = 0
