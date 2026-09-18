"""Active Babbling: VLM-driven interactive system identification.

The core novelty of FMB: VLM is NOT a policy that outputs actions.
It is a SYSTEM IDENTIFIER that actively designs experiments, proposes
hypotheses, and refines them from observations.

Flow:
  Phase 1 — Active System Identification:
    Round 0: systematic test (each action once)
    Round 1+: VLM analyzes observations → proposes T-hat hypotheses →
             identifies uncertain actions → selects experiment →
             observes result → updates beliefs → repeats

    The VLM sees: observation history + current hypotheses + uncertainty
    The VLM outputs: {refined_hypotheses, next_experiment_to_run}

  Phase 2 — Plan-over-Model:
    Best T-hat → ForwardModelSimulator → BFS → execute

This is the method described in METHOD_DESIGN_FMB.md §2.1 (Component A).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from alienbody.agents import Agent, DONE_EXPLORING
from alienbody.agents.fmb_agent import FMBAgent
from alienbody.agents.llm_agent import ModelClient
from alienbody.schema import ForwardModel, ActionSchema, EffectType, ForwardModelSimulator


# ── Active Babbling Prompt ───────────────────────────────────────

_ACTIVE_BABBLING_SYSTEM = """You are a scientist trying to understand an unknown action system through experimentation.

You have {n_actions} anonymous actions (Action 0 through {max_action}).
You CANNOT see the effects directly — you only get observation reports.
Your job is to DESIGN EXPERIMENTS to figure out what each action does.

The actions belong to these possible types:
- translate(direction): moves agent. direction = up/down/left/right/forward/backward
  "forward"/"backward" are relative to the agent's facing direction (red arrow)
- rotate(rotation): changes facing. rotation = cw/ccw
- state_change(change): modifies internal state (color). change = inc/dec/color_move/color_interact
  color_move = direction depends on agent's color (0=up,1=right,2=down,3=left)
- relational(relation): moves relative to colored cells on the grid
- composite(primitives): chains two sub-effects
- temporal: effect depends on previous action

You have {budget} exploration steps remaining.
For each round, I will show you what you've observed so far.
You must respond with a JSON object:

{{
  "hypotheses": {{
    "0": {{"effect_type": "translate", "params": {{"direction": "?"}}, "confidence": 0.5}},
    ...
  }},
  "most_uncertain": <action_id>,
  "next_experiment": <action_id>,
  "reasoning": "<why this experiment?>"
}}

Rules:
- "most_uncertain": the action you're LEAST sure about (lowest confidence or ambiguous effects)
- "next_experiment": which action to test next (0-{max_action})
- "confidence": 0.0 (pure guess) to 1.0 (certain)
- Use "?" for unknown params
- If all confidences > 0.8, set "next_experiment" to -1 (done exploring)
- Before testing, consider: should you change the agent's STATE first?
  (e.g., rotate to a different facing, or change color) to see if effects are state-dependent.
  If so, suggest the state-change action as "next_experiment" before re-testing.

Output ONLY the JSON object, no other text."""


def build_active_babbling_prompt(history: list[dict], n_actions: int,
                                  budget: int, round_num: int) -> str:
    """Build prompt for round N of active babbling."""
    lines = [f"=== Round {round_num} ==="]
    lines.append(f"Exploration budget remaining: {budget} steps\n")
    lines.append("Observations so far:")

    for action in range(n_actions):
        recs = [h for h in history if h.get("action") == action]
        if not recs:
            lines.append(f"  Action {action}: NOT YET TESTED")
            continue

        lines.append(f"  Action {action}: tested {len(recs)} time(s)")
        for rec in recs:
            parts = []
            if rec.get("position_changed"):
                pr, pc = rec.get("prev_pos", (0,0))
                nr, nc = rec.get("new_pos", (0,0))
                dr, dc = nr - pr, nc - pc
                delta_to_dir = {(-1,0):"up",(1,0):"down",(0,-1):"left",(0,1):"right"}
                direction = delta_to_dir.get((dr,dc), f"({dr:+d},{dc:+d})")
                parts.append(f"moved {direction}")
            if rec.get("direction_changed"):
                dd = (rec.get("new_dir",0) - rec.get("prev_dir",0)) % 4
                rot = "cw" if dd == 1 else "ccw"
                parts.append(f"rotated {rot}")
            if rec.get("color_changed"):
                parts.append(f"color {rec['prev_color']}->{rec['new_color']}")
            if not parts:
                parts.append("no visible effect")

            ctx = []
            if rec.get("prev_dir", 0) != 0:
                dir_names = {0:"N",1:"E",2:"S",3:"W"}
                ctx.append(f"facing={dir_names.get(rec['prev_dir'],'?')}")
            if rec.get("prev_color", 0) != 0:
                ctx.append(f"color={rec['prev_color']}")
            ctx_str = f"[{', '.join(ctx)}] " if ctx else ""
            lines.append(f"    {ctx_str}{'; '.join(parts)}")
        lines.append("")

    lines.append("Based on these observations, propose your hypotheses and next experiment.")
    lines.append("Output ONLY the JSON.")
    return "\n".join(lines)


def parse_active_response(response: str, n_actions: int) -> dict:
    """Parse VLM's active babbling response into structured data."""
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        return {"hypotheses": {}, "most_uncertain": 0, "next_experiment": 0, "reasoning": "parse error"}

    try:
        data = json.loads(json_match.group(0))
        return {
            "hypotheses": data.get("hypotheses", {}),
            "most_uncertain": data.get("most_uncertain", 0),
            "next_experiment": data.get("next_experiment", 0),
            "reasoning": data.get("reasoning", ""),
        }
    except json.JSONDecodeError:
        return {"hypotheses": {}, "most_uncertain": 0, "next_experiment": 0, "reasoning": "json error"}


# ── Active Babbling Agent ────────────────────────────────────────

class ActiveBabblingAgent(FMBAgent):
    """FMB with VLM-driven Active Babbling (interactive system identification).

    Phase 1: VLM iteratively proposes hypotheses, selects experiments, refines.
    Phase 2: BFS over best inferred T-hat.

    This agent demonstrates the core FMB paradigm:
    "VLM as system identifier, not policy."
    """

    def __init__(self, config, vlm_client: ModelClient,
                 max_active_rounds: int = 3, steps_per_round: int = 4):
        super().__init__(config)
        self._vlm = vlm_client
        self._max_rounds = max_active_rounds
        self._steps_per_round = steps_per_round
        self._active_round = 0
        self._active_step = 0
        self._active_done = False
        self._vlm_reasoning: list[str] = []  # record for analysis

    @property
    def name(self) -> str:
        return f"ActiveBabbling({self._vlm.name})"

    def reset(self):
        super().reset()
        self._active_round = 0
        self._active_step = 0
        self._active_done = False
        self._vlm_reasoning = []

    def _act_phase1(self, obs: dict, info: dict | None) -> int:
        """Active Babbling: VLM designs experiments."""
        self._phase1_step += 1

        # Round 0: systematic (test each action once)
        if self._active_round == 0:
            if self._phase1_step <= self.n_actions:
                return self._phase1_step - 1
            self._active_round = 1
            self._active_step = 0

        # Rounds 1+: VLM-driven active exploration
        if not self._active_done and self._active_round <= self._max_rounds:
            # Ask VLM what to do next
            if self._active_step == 0 or self._active_step >= self._steps_per_round:
                decision = self._ask_vlm()
                self._next_action = decision["next_experiment"]
                self._vlm_reasoning.append(
                    f"Round {self._active_round}: {decision.get('reasoning', '')[:200]}")

                if self._next_action < 0:
                    self._active_done = True
                    return DONE_EXPLORING

                self._active_step = 0
                self._active_round += 1

            self._active_step += 1
            return self._next_action

        return DONE_EXPLORING

    def _ask_vlm(self) -> dict:
        """Ask VLM to analyze observations and propose next experiment."""
        budget = self.config.phase1_budget - self._phase1_step

        history = []
        for action in range(self.n_actions):
            for rec in self._observations.get(action, []):
                history.append({
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
                })

        prompt = build_active_babbling_prompt(
            history, self.n_actions, budget, self._active_round)

        system = _ACTIVE_BABBLING_SYSTEM.format(
            n_actions=self.n_actions,
            max_action=self.n_actions - 1,
            budget=budget,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._vlm.complete(messages, max_tokens=512)
            return parse_active_response(response, self.n_actions)
        except Exception:
            return {"hypotheses": {}, "most_uncertain": 0,
                    "next_experiment": -1, "reasoning": "API error"}

    def get_reasoning_log(self) -> list[str]:
        """Get VLM reasoning trace for analysis."""
        return self._vlm_reasoning


# ── Plan-over-Model Experiment ───────────────────────────────────

def run_plan_over_model_experiment(config, vlm_client, n_envs: int = 5):
    """Headline experiment: compare BFS(T-hat) vs VLM-in-context(T-hat).

    For the SAME inferred forward model T-hat:
      (A) BFS planner over ForwardModelSimulator(T-hat) → plan → execute
      (B) VLM reads T-hat as text → asked to navigate → executes step by step

    Gap between (A) and (B) = the "planning wall":
    VLM knows what actions do but cannot use that knowledge to plan.

    Returns: dict with comparison results.
    """
    from alienbody.env.core import AlienBodyEnv
    from alienbody.env.grid import EnvConfig
    from alienbody.env.generator import generate_env
    from alienbody.agents import run_episode
    from alienbody.agents.fmb_vlm import VLMFMBAgent
    from alienbody.prompts import build_system_prompt, PromptVariant

    results = {"bfs_plan": [], "vlm_context_plan": []}

    for idx in range(n_envs):
        env_config = generate_env(family=config.family, idx=idx,
                                   split='test', seed=config.seed + idx)

        # Step 1: Use heuristic FMB to induce T-hat
        env = AlienBodyEnv(env_config, render_mode="text")
        fmb_agent = FMBAgent(env_config)
        traj = run_episode(env, fmb_agent)
        t_hat = fmb_agent._forward_model
        if t_hat is None:
            continue

        t_hat_text = json.dumps(t_hat.to_dict(), indent=2)

        # (A) BFS over T-hat
        env_a = AlienBodyEnv(env_config, render_mode="text")
        obs_a, info_a = env_a.reset()
        # Skip to Phase 2
        env_a.start_phase2()
        info_a = env_a._get_info()

        from alienbody.schema import ForwardModelSimulator
        sim = ForwardModelSimulator(t_hat, env_config)
        plan = _bfs_with_simulator(sim, info_a, env_config)

        # Execute BFS plan
        for action in plan:
            if env_a.done:
                break
            obs_a, _, _, _, info_a = env_a.step(action)
        results["bfs_plan"].append(env_a.state.success)

        # (B) VLM-in-context planning
        env_b = AlienBodyEnv(env_config, render_mode="image")
        obs_b, info_b = env_b.reset()
        env_b.start_phase2()
        info_b = env_b._get_info()

        # Build VLM prompt with T-hat in context
        nav_prompt = f"""You are navigating a grid world. You know EXACTLY what each action does:

{t_hat_text}

Your current position: {info_b['agent_pos']}
Your facing direction: {info_b['agent_dir']}
Target position: {env_config.target_pos}

Navigate to the target efficiently. Output only the action number (0-3) for each step."""

        for step in range(50):
            if env_b.done:
                break
            try:
                response = vlm_client.complete([
                    {"role": "user", "content": nav_prompt}
                ], max_tokens=10)
                action = int(re.search(r'\d', response).group(0))
                action = max(0, min(action, env_config.n_actions - 1))
            except:
                action = 0
            obs_b, _, _, _, info_b = env_b.step(action)
            nav_prompt = f"""Position: {info_b['agent_pos']}, Facing: {info_b['agent_dir']}
Target: {env_config.target_pos}
Output next action (0-3):"""

        results["vlm_context_plan"].append(env_b.state.success)

    return results


def _bfs_with_simulator(sim: ForwardModelSimulator, info: dict, config) -> list[int]:
    """BFS plan using ForwardModelSimulator."""
    from collections import deque
    import copy
    from alienbody.env.grid import GridState, Position, Phase

    cur_pos = info.get("agent_pos", (0, 0))
    cur_r, cur_c = cur_pos if isinstance(cur_pos, tuple) else (0, 0)
    cur_dir = info.get("agent_dir", 0)
    cur_color = info.get("agent_color", 0)
    target = info.get("target_pos", config.target_pos)
    tr, tc = target if isinstance(target, tuple) else (0, 0)

    start = GridState(
        agent_pos=Position(cur_r, cur_c), agent_color=cur_color,
        agent_dir=cur_dir, phase=Phase.EXECUTION,
        step_count=0, phase1_steps=0, phase2_steps=0, prev_action=-1)

    queue = deque([(start, [])])
    visited = {(cur_r, cur_c, cur_dir, cur_color, -1)}

    while queue:
        state, path = queue.popleft()
        if len(path) >= 40:
            continue
        for a in range(config.n_actions):
            ns = copy.deepcopy(state)
            ns = sim.predict(ns, a)
            ns.prev_action = a
            nk = (ns.agent_pos.row, ns.agent_pos.col,
                  ns.agent_dir, ns.agent_color, ns.prev_action)
            if ns.agent_pos.row == tr and ns.agent_pos.col == tc:
                return path + [a]
            if nk not in visited:
                visited.add(nk)
                queue.append((ns, path + [a]))
    return []
