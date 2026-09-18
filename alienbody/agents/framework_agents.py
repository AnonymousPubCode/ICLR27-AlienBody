"""Agent framework wrappers: ReAct, Reflexion, InnerMonologue, PlanAndExecute.

Each wrapper augments an LLMAgent with a specific reasoning strategy:
  ReAct:          Forces Thought → Action interleaved format
  Reflexion:      Multi-attempt with self-reflection between failures
  InnerMonologue: Maintains explicit working memory of action semantics
  PlanAndExecute: Generates structured plans before acting
"""
from __future__ import annotations

import re
from typing import Optional

from alienbody.agents import Agent, DONE_EXPLORING
from alienbody.agents.llm_agent import LLMAgent
from alienbody.prompts import parse_action_response


# ── Base Framework Wrapper ────────────────────────────────────────

class _FrameworkWrapper(Agent):
    """Base class for framework wrappers that augment LLMAgent system prompts.

    Handles the system prompt addon lifecycle safely:
    - Stores the addon text separately from the base agent
    - Re-injects after base.reset() without string growth
    """

    def __init__(self, base_agent: LLMAgent, system_addon: str = ""):
        self.base = base_agent
        self._addon = system_addon
        # Store original system prompt BEFORE modification
        self._original_system_prompt = base_agent._system_prompt
        self.base._system_prompt = self._original_system_prompt + system_addon

    def reset(self):
        self.base.reset()
        # Restore addon cleanly (no accumulation)
        self.base._system_prompt = self._original_system_prompt + self._addon
        self.base._messages[0]["content"] = self.base._system_prompt

    def get_reasoning_log(self) -> list[str]:
        return self.base.get_reasoning_log()


# ── ReAct Agent ────────────────────────────────────────────────────

REACT_SYSTEM_ADDON = """

You MUST follow this format for EVERY response:

THOUGHT: [your reasoning about what you've learned and what to try next]
ACTION: [a single number 0-3, or DONE]

Always include both THOUGHT and ACTION. The ACTION must be on its own line."""


class ReActAgent(_FrameworkWrapper):
    """ReAct wrapper: forces Thought → Action interleaved reasoning.

    Wraps any LLMAgent and modifies prompts to require explicit reasoning
    before each action. Parses THOUGHT and ACTION from response.
    """

    def __init__(self, base_agent: LLMAgent):
        super().__init__(base_agent, REACT_SYSTEM_ADDON)

    @property
    def name(self) -> str:
        return f"ReAct({self.base.name})"

    def act(self, observation: dict, info: dict | None = None) -> int:
        action = self.base.act(observation, info)

        # Parse the last response for THOUGHT/ACTION structure
        if self.base._reasoning_log:
            parsed = _parse_react_response(
                self.base._reasoning_log[-1], self.base.n_actions
            )
            if parsed is not None:
                return parsed

        return action


def _parse_react_response(response: str, n_actions: int) -> int | None:
    """Parse ReAct format: THOUGHT: ... ACTION: ..."""
    action_match = re.search(r"ACTION\s*:\s*(.+)", response, re.IGNORECASE)
    if action_match:
        action_text = action_match.group(1).strip()
        result = parse_action_response(action_text, n_actions)
        if result == "done":
            return DONE_EXPLORING
        return result
    return None


# ── Reflexion Agent ────────────────────────────────────────────────

REFLECTION_PROMPT = """You just failed this task. Here is your trajectory:

{trajectory_summary}

Reflect on what went wrong:
1. Did you explore all actions systematically?
2. Did you correctly identify what each action does?
3. What would you do differently next time?

Write a brief reflection (2-3 sentences):"""


class ReflexionAgent(_FrameworkWrapper):
    """Reflexion wrapper: retries with self-reflection on failure.

    After a failed episode, generates a reflection and injects it as
    context for the next attempt at the same environment.
    """

    def __init__(self, base_agent: LLMAgent, max_retries: int = 3):
        super().__init__(base_agent, "")
        self.max_retries = max_retries
        self._reflections: list[str] = []
        self._current_reflection: str | None = None

    @property
    def name(self) -> str:
        return f"Reflexion({self.base.name},retries={self.max_retries})"

    def reset(self):
        # Build reflection addon for this attempt
        if self._current_reflection:
            self._addon = (
                f"\n\nIMPORTANT — Reflection from previous attempt:\n"
                f"{self._current_reflection}\n"
                f"Use this insight to do better this time."
            )
        else:
            self._addon = ""
        super().reset()

    def act(self, observation: dict, info: dict | None = None) -> int:
        return self.base.act(observation, info)

    def add_reflection(self, reflection: str):
        """Add a reflection from a failed attempt."""
        self._reflections.append(reflection)
        self._current_reflection = reflection

    def clear_reflections(self):
        """Clear all reflections (for new environment)."""
        self._reflections = []
        self._current_reflection = None
        self._addon = ""


def run_episode_with_reflexion(env, agent: ReflexionAgent, verbose: bool = False) -> dict:
    """Run an episode with Reflexion retry loop.

    If the agent fails, generates reflection and retries up to max_retries times.
    Returns the trajectory from the successful or last attempt.
    """
    from alienbody.agents import run_episode

    agent.clear_reflections()
    best_traj = None

    for attempt in range(agent.max_retries):
        if verbose:
            print(f"  Reflexion attempt {attempt + 1}/{agent.max_retries}")

        traj = run_episode(env, agent, verbose=verbose)

        if traj["success"]:
            return traj

        best_traj = traj

        # Generate reflection on failure
        summary = _summarize_trajectory(traj)
        reflection_prompt = REFLECTION_PROMPT.format(trajectory_summary=summary)

        # Use the base LLM client to generate reflection
        reflection_messages = [
            {"role": "system", "content": "You are reflecting on a failed navigation task."},
            {"role": "user", "content": reflection_prompt},
        ]
        reflection = agent.base.client.complete(reflection_messages, max_tokens=200)
        agent.add_reflection(reflection)

        if verbose:
            print(f"  Reflection: {reflection[:100]}...")

    return best_traj


def _summarize_trajectory(traj: dict) -> str:
    """Summarize a trajectory for reflection prompt."""
    lines = []
    lines.append(f"Total steps: {traj['total_steps']} (P1={traj['phase1_steps']}, P2={traj['phase2_steps']})")
    lines.append(f"Result: {'SUCCESS' if traj['success'] else 'FAILED'}")

    for record in traj["history"][:10]:
        phase = "P1" if record["phase"] == 1 else "P2"
        pos_info = f"pos ({record['prev_pos'][0]},{record['prev_pos'][1]})"
        if record["position_changed"]:
            pos_info += f"→({record['new_pos'][0]},{record['new_pos'][1]})"
        else:
            pos_info += " (unchanged)"
        lines.append(f"  [{phase}] Action {record['action']}: {pos_info}")

    if len(traj["history"]) > 10:
        lines.append(f"  ... ({len(traj['history']) - 10} more steps)")

    return "\n".join(lines)


# ── InnerMonologue Agent ──────────────────────────────────────────

INNER_MONOLOGUE_ADDON = """

CRITICAL: You maintain a WORKING MEMORY of what you've learned about each action.
After EVERY action, update your working memory by summarizing the observed effect.

Format your response as:

MEMORY UPDATE:
- Action 0: [what you know about it, or "unknown"]
- Action 1: [what you know about it, or "unknown"]
- Action 2: [what you know about it, or "unknown"]
- Action 3: [what you know about it, or "unknown"]

REASONING: [your plan for next step]
ACTION: [number 0-3, or DONE]"""


class InnerMonologueAgent(_FrameworkWrapper):
    """Maintains an explicit working memory of learned action semantics.

    After each step, the agent updates a structured memory of what it has
    learned about each action. This memory is injected into every prompt,
    helping the LLM maintain coherent beliefs across the episode.

    This tests whether explicit memory scaffolding improves calibration.
    """

    def __init__(self, base_agent: LLMAgent):
        super().__init__(base_agent, INNER_MONOLOGUE_ADDON)
        self._working_memory: dict[int, str] = {}

    @property
    def name(self) -> str:
        return f"InnerMonologue({self.base.name})"

    def reset(self):
        super().reset()
        self._working_memory = {
            i: "unknown" for i in range(self.base.n_actions)
        }

    def act(self, observation: dict, info: dict | None = None) -> int:
        # Inject current working memory into observation text
        memory_str = "\n".join(
            f"- Action {k}: {v}" for k, v in sorted(self._working_memory.items())
        )
        obs = dict(observation)  # shallow copy to avoid mutating caller's dict
        if "text" in obs:
            obs["text"] = (
                f"YOUR CURRENT WORKING MEMORY:\n{memory_str}\n\n"
                + obs["text"]
            )

        action = self.base.act(obs, info)

        # Parse memory updates from response
        if self.base._reasoning_log:
            self._parse_memory_update(self.base._reasoning_log[-1])
            parsed = _parse_react_response(
                self.base._reasoning_log[-1], self.base.n_actions
            )
            if parsed is not None:
                return parsed

        return action

    def _parse_memory_update(self, response: str):
        """Extract memory updates from model response."""
        for line in response.split("\n"):
            match = re.match(
                r"-?\s*Action\s*(\d)\s*:\s*(.+)", line.strip(), re.IGNORECASE
            )
            if match:
                idx = int(match.group(1))
                desc = match.group(2).strip()
                if idx < self.base.n_actions and desc.lower() != "unknown":
                    self._working_memory[idx] = desc


# ── PlanAndExecute Agent ──────────────────────────────────────────

PLAN_PHASE1_PROMPT = """Before you start exploring, create a PLAN for Phase 1.
You have {n_actions} unknown actions and a budget of {budget} exploration steps.

Write your exploration plan:
PLAN: [describe your strategy for systematically testing all actions]

Then execute the first step:
ACTION: [number]"""

PLAN_PHASE2_PROMPT = """Phase 2 has started. The target is now visible.

Based on what you learned in Phase 1, create a NAVIGATION PLAN:
- What does each action do?
- What sequence of actions will take you to the target?

PLAN: [your navigation strategy]
ACTION: [number]"""


class PlanAndExecuteAgent(_FrameworkWrapper):
    """Two-stage planning agent: generates explicit plans before acting.

    At the start of each phase, asks the LLM to generate a structured plan.
    Then executes the plan step-by-step.

    This tests whether upfront planning improves calibration efficiency.
    """

    def __init__(self, base_agent: LLMAgent):
        super().__init__(base_agent, "")
        self._phase1_planned = False
        self._phase2_planned = False

    @property
    def name(self) -> str:
        return f"PlanAndExecute({self.base.name})"

    def reset(self):
        super().reset()
        self._phase1_planned = False
        self._phase2_planned = False

    def act(self, observation: dict, info: dict | None = None) -> int:
        phase = observation.get("phase", 1)

        # Inject planning prompt at phase transitions
        if phase == 1 and not self._phase1_planned:
            self._phase1_planned = True
            return self._plan_and_act(observation, info, PLAN_PHASE1_PROMPT.format(
                n_actions=self.base.n_actions,
                budget=20,
            ))

        if phase == 2 and not self._phase2_planned:
            self._phase2_planned = True
            return self._plan_and_act(observation, info, PLAN_PHASE2_PROMPT)

        return self.base.act(observation, info)

    def _plan_and_act(self, observation: dict, info: dict | None,
                      plan_prompt: str) -> int:
        """Inject planning prompt, get plan + first action."""
        obs = dict(observation)
        if "text" in obs:
            obs["text"] = plan_prompt + "\n\n" + obs["text"]
        else:
            obs["feedback"] = plan_prompt

        action = self.base.act(obs, info)

        # Try to extract action from structured response
        if self.base._reasoning_log:
            parsed = _parse_react_response(
                self.base._reasoning_log[-1], self.base.n_actions
            )
            if parsed is not None:
                return parsed

        return action
