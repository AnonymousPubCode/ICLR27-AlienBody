"""CodeToolAgent — L3 mapping + Python REPL over next_state (wall stress test).

Closes the reviewer hole: "why not just write BFS?"

Compared to ToolAgent (conversational one-shot next_state queries), this agent
lets the model write a Python program that may call ``next_state`` in a loop,
run BFS / Dijkstra, and return ``PLAN = [a0, a1, ...]``.

Same information as L3 + ToolAgent; the only change is that search can be
*programmed* rather than assembled turn-by-turn in natural language.
"""
from __future__ import annotations

import ast
import re
import signal
import sys
import textwrap
import time
import traceback
from collections import deque
from typing import Any

from alienbody.agents import DONE_EXPLORING, Agent
from alienbody.agents.llm_agent import ModelClient
from alienbody.env.actions import apply_action
from alienbody.env.grid import EnvConfig, GridState, Phase, Position
from alienbody.prompts import PromptVariant

# Operational gloss for L3 effect names (Type C + common others).
# Names alone are under-specified (tie-breaks); docs make reimplementation fair.
_EFFECT_DOCS = {
    "move_to_nearest_diff_color": (
        "Teleport to the nearest cell whose color differs from the current "
        "cell. Ties broken by row-major scan (smallest row, then col)."
    ),
    "move_to_nearest_same_color": (
        "Teleport to the nearest other cell with the same color as the "
        "current cell. Ties: row-major."
    ),
    "move_toward_brightest": (
        "Move 1 step to the 4-neighbor with the highest color value "
        "(must be > 0). Ties: first among N,E,S,W in that order."
    ),
    "flee_same_color": (
        "Move 1 step away from the nearest same-color cell "
        "(axis-aligned unit step)."
    ),
    "move_to_farthest_same_color": (
        "Teleport to the farthest same-color cell (Manhattan). Ties: last "
        "in row-major overwrite."
    ),
    "move_away_from_brightest": (
        "Identify brightest 4-neighbor, then step one cell in the opposite "
        "direction if in bounds."
    ),
    "move_to_farthest_diff_color": (
        "Teleport to farthest different-color cell (Manhattan)."
    ),
    "flee_nearest_diff_color": (
        "Move 1 step away from nearest different-color cell."
    ),
    "move_toward_darkest": (
        "Move 1 step to the 4-neighbor with the lowest color value."
    ),
    "move_away_from_darkest": (
        "Step opposite the darkest 4-neighbor."
    ),
    "move_to_nearest_brighter": (
        "Teleport to nearest cell with strictly higher color."
    ),
    "move_to_nearest_darker": (
        "Teleport to nearest cell with strictly lower color."
    ),
}

_SYSTEM_L3_CODE = """\
You are in a discrete grid world. Exact action effects are given below.
Your job is to reach TARGET by writing Python that searches over a simulator.

Available actions (0-{action_hi}):
{mapping_description}

Effect semantics (operational):
{effect_docs}

You may execute Python in a restricted sandbox. The sandbox preloads:
  START: (row, col)          # current agent position
  TARGET: (row, col)
  GRID_SIZE: int
  COLORS: list[list[int]]    # COLORS[r][c] cell color
  OBSTACLES: set[(r,c)]
  N_ACTIONS: int
  MAPPING: list[str]         # effect name per action index
  next_state(r, c, a) -> (nr, nc)
      # true one-step transition from (r,c) taking action a
      # (uses the real environment transition; call budget {sim_budget})

Write a program that computes a path. End by setting ONE of:
  PLAN = [a0, a1, ...]   # preferred: full action sequence from START to TARGET
  ACTION = <int>         # only the next action (weaker)

Output format — a single fenced block, nothing else:
```python
# your code
PLAN = [...]
```

If your previous program errored, you will see the traceback; fix and resubmit.
Do not narrate. Do not call the real environment — only set PLAN/ACTION.
"""

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

# AST safety: reject these node types / names
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
    ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith, ast.Await,
    ast.With,  # no context managers that could touch files
)
_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input",
    "breakpoint", "memoryview", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr",
    "classmethod", "staticmethod", "property",
    "exit", "quit", "help", "copyright", "credits", "license",
}


class _SandboxTimeout(Exception):
    pass


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, ast.Call):
            # block getattr(obj, "__class__") style via literal attr strings
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden call: {node.func.id}")


def _run_sandbox(
    code: str,
    env: dict[str, Any],
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Execute user code against a restricted builtins set."""
    tree = ast.parse(code, mode="exec")
    _validate_ast(tree)
    compiled = compile(tree, filename="<agent_code>", mode="exec")

    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "len": len,
        "list": list, "max": max, "min": min, "pow": pow, "print": print,
        "range": range, "reversed": reversed, "round": round, "set": set,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "True": True, "False": False, "None": None,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "KeyError": KeyError, "IndexError": IndexError, "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
    }
    # Limited stdlib helpers (no IO)
    import collections
    import heapq
    import itertools
    import math

    namespace = {
        "__builtins__": safe_builtins,
        "deque": collections.deque,
        "Counter": collections.Counter,
        "heapq": heapq,
        "itertools": itertools,
        "math": math,
        **env,
    }

    # D34: the timeout has to hold on Windows, where there is no SIGALRM.  A
    # legal ``while True:`` in a submitted program otherwise spins the runner at
    # 100% CPU forever and the cell can never finish (observed 2026-09-15 on the
    # preflight2 F4 dsv4 J4 cell: 35+ min at 100% CPU, nothing written).  The
    # alarm stays on POSIX as a backstop; the deadline trace hook is what bounds
    # ``exec`` everywhere -- it fires on line events in the submitted program's
    # own frames, so a Python-level infinite loop raises instead of spinning.
    # A pure C-level call that never returns to bytecode (``pow(3, 10**9)``) is
    # out of its reach and is bounded only by memory; none was observed, and
    # next_state loops are bounded by the query budget regardless.
    use_alarm = hasattr(signal, "SIGALRM")
    deadline = time.monotonic() + float(timeout_s)
    ticks = 0

    def _trace(frame, event, arg):  # noqa: ARG001
        nonlocal ticks
        if frame.f_code.co_filename != "<agent_code>":
            return None               # never instrument the adapter's frames
        ticks += 1
        if ticks % 4096 == 0 and time.monotonic() > deadline:
            raise _SandboxTimeout("sandbox timeout")
        return _trace

    def _handler(signum, frame):  # noqa: ARG001
        raise _SandboxTimeout("sandbox timeout")

    old_alarm = None
    if use_alarm:
        old_alarm = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
    old_trace = sys.gettrace()
    sys.settrace(_trace)
    try:
        exec(compiled, namespace, namespace)  # noqa: S102 — intentional sandbox
    finally:
        sys.settrace(old_trace)
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_alarm is not None:
                signal.signal(signal.SIGALRM, old_alarm)

    return {
        "PLAN": namespace.get("PLAN"),
        "ACTION": namespace.get("ACTION"),
        "stdout_note": None,
    }


class CodeToolAgent(Agent):
    """L3 + programmable next_state search."""

    def __init__(
        self,
        config: EnvConfig,
        client: ModelClient,
        modality: str = "text",
        prompt_variant: str | PromptVariant = "minimal",
        max_code_rounds: int = 4,
        sim_budget: int = 50000,
        max_plan_len: int = 80,
        response_max_tokens: int = 2048,
    ):
        self.config = config
        self.client = client
        self.modality = modality
        self.max_code_rounds = max_code_rounds
        self.sim_budget = sim_budget
        self.max_plan_len = max_plan_len
        self.response_max_tokens = response_max_tokens
        self._messages: list[dict] = []
        self._plan: list[int] = []
        self._tool_log: list[dict] = []
        self._reasoning_log: list[str] = []
        self._sim_calls = 0
        self._planned = False
        self._system_prompt = self._build_system_prompt()

    @property
    def name(self) -> str:
        return f"CodeToolAgent({self.client.name})"

    def reset(self):
        self._messages = []
        self._plan = []
        self._tool_log = []
        self._reasoning_log = []
        self._sim_calls = 0
        self._planned = False

    def get_tool_log(self) -> list[dict]:
        return list(self._tool_log)

    def get_reasoning_log(self) -> list[str]:
        return list(self._reasoning_log)

    def _build_system_prompt(self) -> str:
        mapping = list(self.config.action_mapping)
        mapping_desc = "\n".join(
            f"  Action {i} = {effect}" for i, effect in enumerate(mapping)
        )
        docs = []
        for i, effect in enumerate(mapping):
            gloss = _EFFECT_DOCS.get(effect, "(see effect name; implement via next_state)")
            docs.append(f"  Action {i} ({effect}): {gloss}")
        return _SYSTEM_L3_CODE.format(
            action_hi=self.config.n_actions - 1,
            mapping_description=mapping_desc,
            effect_docs="\n".join(docs),
            sim_budget=self.sim_budget,
        )

    def _next_state_fn(self, r: int, c: int, a: int) -> tuple[int, int]:
        self._sim_calls += 1
        if self._sim_calls > self.sim_budget:
            raise RuntimeError(f"next_state budget exceeded ({self.sim_budget})")
        if not (0 <= r < self.config.grid_size and 0 <= c < self.config.grid_size):
            return (r, c)
        a = int(a) % self.config.n_actions
        state = GridState(
            agent_pos=Position(r, c), agent_color=0, agent_dir=0,
            phase=Phase.EXECUTION, step_count=0, phase1_steps=0, phase2_steps=0,
        )
        state = apply_action(state, self.config, a)
        return (state.agent_pos.row, state.agent_pos.col)

    def _sandbox_env(self, start: tuple[int, int]) -> dict[str, Any]:
        obstacles = set()
        if getattr(self.config, "obstacles", None):
            obstacles = {(int(r), int(c)) for r, c in self.config.obstacles}
        return {
            "START": tuple(start),
            "TARGET": tuple(self.config.target_pos),
            "GRID_SIZE": int(self.config.grid_size),
            "COLORS": [list(row) for row in self.config.cell_colors],
            "OBSTACLES": obstacles,
            "N_ACTIONS": int(self.config.n_actions),
            "MAPPING": list(self.config.action_mapping),
            "next_state": self._next_state_fn,
        }

    def _extract_code(self, response: str) -> str | None:
        m = _CODE_FENCE_RE.search(response)
        if m:
            return textwrap.dedent(m.group(1)).strip()
        # Fallback: entire response if it looks like code
        if "PLAN" in response or "ACTION" in response:
            return textwrap.dedent(response).strip()
        return None

    def _coerce_plan(self, result: dict[str, Any]) -> list[int] | None:
        plan = result.get("PLAN")
        action = result.get("ACTION")
        if plan is not None:
            if not isinstance(plan, (list, tuple)):
                raise ValueError(f"PLAN must be a list, got {type(plan)}")
            out = []
            for a in plan:
                ai = int(a)
                if not (0 <= ai < self.config.n_actions):
                    raise ValueError(f"PLAN action out of range: {ai}")
                out.append(ai)
            if len(out) > self.max_plan_len:
                out = out[: self.max_plan_len]
            return out
        if action is not None:
            ai = int(action)
            if not (0 <= ai < self.config.n_actions):
                raise ValueError(f"ACTION out of range: {ai}")
            return [ai]
        return None

    def _user_state_message(self, info: dict) -> str:
        pos = tuple(info.get("agent_pos", self.config.agent_start))
        return (
            f"START = {pos}\n"
            f"TARGET = {tuple(self.config.target_pos)}\n"
            f"GRID_SIZE = {self.config.grid_size}\n"
            f"N_ACTIONS = {self.config.n_actions}\n"
            f"Write Python now to set PLAN (or ACTION)."
        )

    def _induce_plan(self, info: dict) -> list[int]:
        """Multi-round code induction; returns a plan (possibly empty)."""
        start = tuple(info.get("agent_pos", self.config.agent_start))
        if isinstance(start, list):
            start = tuple(start)

        self._messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": self._user_state_message(info)},
        ]

        last_err = None
        for round_i in range(self.max_code_rounds):
            response = self.client.complete(
                self._messages, max_tokens=self.response_max_tokens
            )
            if not response or not response.strip():
                response = "# empty\nPLAN = []"
            self._reasoning_log.append(response)
            self._messages.append({"role": "assistant", "content": response})

            code = self._extract_code(response)
            if not code:
                last_err = "No ```python``` block found. Resubmit with a fenced block."
                self._messages.append({"role": "user", "content": last_err})
                self._tool_log.append({
                    "round": round_i, "ok": False, "error": "no_code_block",
                    "sim_calls": self._sim_calls,
                })
                continue

            sim_before = self._sim_calls
            try:
                result = _run_sandbox(code, self._sandbox_env(start), timeout_s=5.0)
                plan = self._coerce_plan(result)
                if plan is None:
                    raise ValueError("Code ran but neither PLAN nor ACTION was set")
                self._tool_log.append({
                    "round": round_i, "ok": True, "plan_len": len(plan),
                    "sim_calls_delta": self._sim_calls - sim_before,
                    "sim_calls": self._sim_calls,
                    "code_preview": code[:500],
                })
                return plan
            except Exception as e:
                last_err = (
                    f"Execution error (round {round_i}):\n"
                    f"{type(e).__name__}: {e}\n"
                    f"{traceback.format_exc(limit=3)}\n"
                    "Fix the program and resubmit a ```python``` block."
                )
                self._messages.append({"role": "user", "content": last_err})
                self._tool_log.append({
                    "round": round_i, "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "sim_calls": self._sim_calls,
                    "code_preview": code[:300],
                })

        return []

    def act(self, observation: dict, info: dict | None = None) -> int:
        if observation.get("phase", 1) == 1:
            return DONE_EXPLORING

        info = info or {}
        if not self._planned:
            self._plan = self._induce_plan(info)
            self._planned = True

        if self._plan:
            return self._plan.pop(0)
        return 0


def reference_bfs_plan(config: EnvConfig, start: tuple[int, int] | None = None,
                       max_depth: int = 40) -> list[int]:
    """Oracle BFS using the same next_state semantics (for unit tests)."""
    if start is None:
        start = tuple(config.agent_start)
    target = tuple(config.target_pos)
    q = deque([(start, [])])
    visited = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == target:
            return path
        if len(path) >= max_depth:
            continue
        for a in range(config.n_actions):
            state = GridState(
                agent_pos=Position(r, c), agent_color=0, agent_dir=0,
                phase=Phase.EXECUTION, step_count=0, phase1_steps=0, phase2_steps=0,
            )
            state = apply_action(state, config, a)
            nxt = (state.agent_pos.row, state.agent_pos.col)
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [a]))
    return []
