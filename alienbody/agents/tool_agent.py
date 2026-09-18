"""ToolAgent — L3 mapping + one-step transition oracle (A1 clean planning).

The A1 experiment isolates the *search* component of the planning wall.
Everything else is removed by construction:

  - symbolic text state (no perception),
  - exact action->effect mapping, identical to the L3 setting (no interpretation),
  - a `next_state(row, col, action)` simulator query executed locally against
    the true transition function (no mental simulation).

The only thing the agent must still do is assemble a search over the
transition function. Protocol in Phase 2: each turn the model may output
either (a) a single action number -> executed in the real environment
(consumes 1 of the 30 step budget), or (b) `CALL next_state(r, c, a)` ->
executed in a cloned state, result appended (consumes no env step).
Tool queries are capped per episode (default 120, i.e. 4x the step budget).
"""
from __future__ import annotations

import re

from alienbody.agents import DONE_EXPLORING
from alienbody.agents.llm_agent import LLMAgent, ModelClient
from alienbody.env.actions import apply_action
from alienbody.env.grid import EnvConfig, GridState, Phase, Position
from alienbody.prompts import PromptVariant, build_turn_prompt, parse_action_response

_SYSTEM_L3_TOOL = """\
You are in a grid world. You can see your position and the environment.

You have {n_actions} actions with the following effects:
{mapping_description}

A target location is marked on the grid. Navigate to it as efficiently as possible.

You have access to a simulator. To find out what would happen if you took an
action from a position, output:
  CALL next_state(<row>, <col>, <action>)
The simulator will reply with the position you would end up at. You may query
it as many times as you need to plan your path.

Each turn, output ONLY one of:
  - a single number ({action_range}) to take an action in the real environment
  - CALL next_state(<row>, <col>, <action>) to query the simulator"""

_TOOL_RE = re.compile(
    r"next_state\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", re.IGNORECASE
)


class ToolAgent(LLMAgent):
    """L3-oracle agent with a one-step transition simulator query.

    Subclasses LLMAgent to reuse client handling and history management;
    overrides the system prompt and the Phase-2 decision loop so that
    `CALL next_state(r, c, a)` outputs are executed against the true
    environment transition function instead of being treated as actions.

    Simulator semantics: for the relational family (Type C) the transition
    depends only on (position, action, static cell colors), so a position-only
    query is exact. Queries outside the grid return the input unchanged.
    """

    def __init__(
        self,
        config: EnvConfig,
        client: ModelClient,
        modality: str = "text",
        prompt_variant: str | PromptVariant = "minimal",
        familiar: bool = False,
        max_tool_calls: int = 120,
        max_turns_per_step: int = 12,
        max_history_turns: int = 25,
    ):
        self.config = config
        self.max_tool_calls = max_tool_calls
        self.max_turns_per_step = max_turns_per_step
        self._tool_log: list[dict] = []
        self._tool_calls_this_episode = 0

        mapping_desc = "\n".join(
            f"  Action {i} = {effect}" for i, effect in enumerate(config.action_mapping)
        )
        tool_system_prompt = _SYSTEM_L3_TOOL.format(
            n_actions=config.n_actions,
            mapping_description=mapping_desc,
            action_range=f"0-{config.n_actions - 1}",
        )

        variant = prompt_variant if isinstance(prompt_variant, PromptVariant) \
            else PromptVariant(prompt_variant)

        super().__init__(
            client=client, modality=modality, prompt_variant=variant,
            familiar=familiar, n_actions=config.n_actions,
            max_history_turns=max_history_turns,
            assist_level=3, action_mapping=list(config.action_mapping),
        )
        # Override the L3 system prompt with the tool-enabled version
        self._system_prompt = tool_system_prompt

    def reset(self):
        super().reset()
        self._tool_log = []
        self._tool_calls_this_episode = 0

    def act(self, observation: dict, info: dict | None = None) -> int:
        # Phase 1: mapping given — no exploration needed (same as L3)
        if observation.get("phase", 1) == 1:
            return DONE_EXPLORING

        for _ in range(self.max_turns_per_step):
            self._step += 1
            turn_text = build_turn_prompt(
                self.prompt_variant, observation, 2, self._step,
                observation.get("feedback"),
            )
            self._messages.append({"role": "user", "content": turn_text})
            self._trim_history()

            response = self.client.complete(self._messages, max_tokens=256)
            if not response or not response.strip():
                response = "0"
            self._reasoning_log.append(response)
            self._messages.append({"role": "assistant", "content": response})

            m = _TOOL_RE.search(response)
            if m and self._tool_calls_this_episode < self.max_tool_calls:
                r, c, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
                a = a % self.config.n_actions  # guard out-of-range action ids
                nr, nc = self._simulate(r, c, a)
                self._tool_calls_this_episode += 1
                self._tool_log.append({
                    "env_step": self._step,
                    "query": [r, c, a],
                    "result": [nr, nc],
                })
                self._messages.append({
                    "role": "user",
                    "content": f"Simulator: next_state({r}, {c}, {a}) -> ({nr}, {nc})",
                })
                continue

            result = parse_action_response(response, self.n_actions)
            if result == "done":
                return DONE_EXPLORING
            return result

        # Turn budget exhausted without a real action: fall back to parsing
        # the last response, else action 0 (mirrors LLMAgent behavior).
        last = self._reasoning_log[-1] if self._reasoning_log else "0"
        result = parse_action_response(last, self.n_actions)
        return result if result != "done" else DONE_EXPLORING

    def _simulate(self, r: int, c: int, a: int) -> tuple[int, int]:
        """Apply the true transition function to a hypothetical position."""
        if not (0 <= r < self.config.grid_size and 0 <= c < self.config.grid_size):
            return (r, c)  # invalid query: treat as no-op
        state = GridState(
            agent_pos=Position(r, c), agent_color=0, agent_dir=0,
            phase=Phase.EXECUTION, step_count=0, phase1_steps=0, phase2_steps=0,
        )
        state = apply_action(state, self.config, a)
        return (state.agent_pos.row, state.agent_pos.col)

    def get_tool_log(self) -> list[dict]:
        """Return all simulator queries made this episode (for analysis)."""
        return list(self._tool_log)
