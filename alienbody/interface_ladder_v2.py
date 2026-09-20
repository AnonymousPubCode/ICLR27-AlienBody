"""Small, auditable primitives used by the Interface Ladder v2 runner.

These deliberately do not subclass the legacy agents: their failure semantics
are part of the preregistered v2 contract, while legacy agents silently map
some malformed outputs to action 0.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field

from alienbody.env.actions import apply_action
from alienbody.env.grid import EnvConfig, GridState, Phase, Position
from alienbody.state_v2 import state_v2_payload, render_state_v2


_ACTION_ONLY = re.compile(r"^\s*([0-9]+)\s*$")


def parse_action_strict(response: str | None, n_actions: int) -> int | None:
    """Parse an action only when the complete response is one valid integer.

    ``None`` represents a protocol-visible parse failure.  The runner records
    it and consumes an environment action; it must never substitute action 0.
    """
    if not response:
        return None
    match = _ACTION_ONLY.fullmatch(response)
    if not match:
        return None
    action = int(match.group(1))
    return action if 0 <= action < n_actions else None


def parse_plan_strict(response: str | None, n_actions: int, max_actions: int = 30) -> list[int] | None:
    """Accept only a JSON list of legal action integers for I2."""
    if not response:
        return None
    try:
        plan = json.loads(response)
    except json.JSONDecodeError:
        return None
    if not isinstance(plan, list) or not plan or len(plan) > max_actions:
        return None
    if any(not isinstance(action, int) or not 0 <= action < n_actions for action in plan):
        return None
    return plan


def parse_scratchpad_action_strict(response: str | None, n_actions: int, max_chars: int = 2000) -> tuple[str, int] | None:
    """Accept I1's sole response format: JSON object with scratchpad/action."""
    if not response:
        return None
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return None
    if set(value) != {"scratchpad", "action"} or not isinstance(value["scratchpad"], str):
        return None
    action = value["action"]
    if len(value["scratchpad"]) > max_chars or not isinstance(action, int) or not 0 <= action < n_actions:
        return None
    return value["scratchpad"], action


@dataclass(frozen=True)
class OracleReply:
    status: str
    position: tuple[int, int]
    query_index: int


@dataclass
class CounterfactualOracle:
    """F4's one-state oracle with v2-invalid-input and budget semantics."""
    config: EnvConfig
    max_queries: int = 120
    query_log: list[dict] = field(default_factory=list)

    def query(self, row: int, col: int, action: int) -> OracleReply:
        if len(self.query_log) >= self.max_queries:
            return OracleReply("budget_exhausted", (row, col), len(self.query_log))

        index = len(self.query_log) + 1
        if not (0 <= row < self.config.grid_size and 0 <= col < self.config.grid_size):
            reply = OracleReply("invalid_coordinate", (row, col), index)
        elif not (0 <= action < self.config.n_actions):
            # Invalid action ids are failures, never modulo-wrapped.
            reply = OracleReply("invalid_action", (row, col), index)
        else:
            state = GridState(
                agent_pos=Position(row, col),
                agent_color=self.config.agent_start_color,
                agent_dir=self.config.agent_start_dir,
                phase=Phase.EXECUTION,
                step_count=0,
                phase1_steps=0,
                phase2_steps=0,
            )
            successor = apply_action(state, self.config, action)
            reply = OracleReply(
                "ok", (successor.agent_pos.row, successor.agent_pos.col), index
            )
        self.query_log.append({
            "index": index,
            "input": [row, col, action],
            "status": reply.status,
            "position": list(reply.position),
        })
        return reply


# ──────────────────────────────────────────────────────────────────────────
# I4 (restricted code oracle) and I5 (BFS reference) — permission aligned.
#
# Permission contract (PROTOCOL_INTERFACE_LADDER.md v2.1 + 2026-09-09
# budget amendment; see that file for the full text):
#   * the model receives exactly the same STATE_V2 block as every other arm;
#   * the sandbox exposes only `STATE` (the parsed canonical observation, i.e.
#     a parse of the exact bytes shown in the transcript — nothing private)
#     and `next_state(r,c,a)` with the shared oracle (no modulo wrapping, no
#     50k simulator budget, no injected COLORS/OBSTACLES/MAPPING globals, no
#     live-environment access);
#   * code is limited to code_tool_agent.py's safe builtins: no import, I/O,
#     object inspection, or random.
#
# Oracle budget calibration (measured 2026-09-09, all 100 frozen F4 envs):
# an uncapped BFS through the same CounterfactualOracle needs 72-964 calls
# (median 460, p90 824).  The v2-era 120-call budget silently capped the
# optimal searcher at 1/50 on the replication primary; 1000 covers all 100.
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_I4 = """You receive a STATE_V2 JSON block for a grid-navigation task. The action mapping and target are exact. You must write ONE Python program that searches with a simulator and returns a complete action plan.

The STATE_V2 block is your ONLY source of grid layout, colors, obstacles, and the action mapping. Inside the sandbox the identical block is available already parsed as the dict STATE (same content as the block above; it adds nothing).

Sandbox provides:
  STATE            dict, parsed STATE_V2 payload (keys match the JSON keys:
                   grid_size, cell_colors_row_major, obstacles, agent,
                   target, n_actions, action_mapping, actions_remaining)
  next_state(r,c,a) -> (nr,nc)
                   true one-step successor of action a from cell (r,c);
                   at most 1000 calls total across all your programs;
                   out-of-range (r,c) returns (r,c) unchanged and counts;
                   an illegal action id raises an error; exhausting the
                   budget raises an error
  deque, heapq, itertools, math, and safe builtins only

No import statements, no file or network access, no access to the real environment.

Your program must end by setting exactly one of:
  PLAN = [a0, a1, ...]   # full action sequence from START to TARGET (1..30 actions)
  ACTION = <int>         # only the next action (weaker)

Output format: a single fenced ```python``` block and nothing else. If your program errors you receive the error and may resubmit a fixed program; at most 4 repair rounds total. Do not narrate."""

I4_MAX_ROUNDS = 4
I4_ORACLE_CALLS = 1000
I4_PLAN_CAP = 30
I4_PHASE2_ACTIONS = 30


def make_sandbox_next_state(config: EnvConfig, oracle: CounterfactualOracle):
    """next_state bound to the shared v2 oracle (counts, strict action ids)."""
    def next_state(row: int, col: int, action: int) -> tuple[int, int]:
        if len(oracle.query_log) >= oracle.max_queries:
            raise RuntimeError(f"next_state budget exhausted ({oracle.max_queries})")
        if isinstance(action, bool) or not isinstance(action, int):
            raise RuntimeError(f"next_state action must be an int, got {action!r}")
        row, col = int(row), int(col)
        reply = oracle.query(row, col, action)
        if reply.status == "budget_exhausted":
            raise RuntimeError(f"next_state budget exhausted ({oracle.max_queries})")
        if reply.status == "invalid_action":
            raise RuntimeError(f"next_state invalid action id: {action}")
        return (reply.position[0], reply.position[1])
    return next_state


def _coerce_plan_v2(result: dict, config: EnvConfig, plan_cap: int = I4_PLAN_CAP):
    """Strict PLAN/ACTION coercion; invalid values raise for the repair loop."""
    plan = result.get("PLAN")
    action = result.get("ACTION")
    if plan is not None:
        if not isinstance(plan, (list, tuple)):
            raise ValueError(f"PLAN must be a list, got {type(plan).__name__}")
        out: list[int] = []
        for entry in plan:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise ValueError(f"PLAN entries must be ints, got {entry!r}")
            if not (0 <= entry < config.n_actions):
                raise ValueError(f"PLAN action out of range: {entry}")
            out.append(entry)
        if not out:
            raise ValueError("PLAN must contain at least one action")
        if len(out) > plan_cap:
            raise ValueError(f"PLAN longer than {plan_cap} actions; shorten it")
        return out
    if action is not None:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError(f"ACTION must be an int, got {action!r}")
        if not (0 <= action < config.n_actions):
            raise ValueError(f"ACTION out of range: {action}")
        return [action]
    return None


def run_i4_episode(config: EnvConfig, env, complete,
                   model: str, max_rounds: int = I4_MAX_ROUNDS,
                   oracle_calls: int = I4_ORACLE_CALLS,
                   response_max_tokens: int = 2048,
                   timeout_s: float = 5.0) -> dict:
    """Execute one I4 episode.

    ``complete(messages, max_tokens) -> str`` abstracts the model client so
    the arm is testable without any API. Returns the row written to the
    non-append result JSONL.
    """
    import textwrap

    from alienbody.agents.code_tool_agent import _CODE_FENCE_RE, _run_sandbox

    oracle = CounterfactualOracle(config, max_queries=oracle_calls)
    payload = state_v2_payload(config, env.state)
    state_text = render_state_v2(config, env.state)
    messages = [
        {"role": "system", "content": SYSTEM_I4},
        {"role": "user", "content": state_text},
    ]
    transcript: list[dict] = []
    parse_failures = 0
    code_errors = 0
    plan: list[int] | None = None
    oracle_calls_at_plan = 0

    for round_i in range(max_rounds):
        response = complete(messages, response_max_tokens)
        messages.append({"role": "assistant", "content": response})
        record: dict = {"round": round_i, "response": response}
        transcript.append(record)

        fence = _CODE_FENCE_RE.search(response or "")
        if not fence:
            parse_failures += 1
            message = (
                "No ```python``` block found. Resubmit with exactly one fenced "
                "```python``` block and nothing else."
            )
            record.update({"ok": False, "error": "no_code_block"})
        else:
            code = textwrap.dedent(fence.group(1)).strip()
            record["code"] = code[:800]
            calls_before = len(oracle.query_log)
            try:
                result = _run_sandbox(
                    code,
                    {
                        "STATE": payload,
                        "next_state": make_sandbox_next_state(config, oracle),
                    },
                    timeout_s=timeout_s,
                )
                plan = _coerce_plan_v2(result, config)
                oracle_calls_at_plan = len(oracle.query_log)
                record.update({
                    "ok": True,
                    "plan_len": len(plan),
                    "oracle_calls": oracle_calls_at_plan,
                    "calls_this_round": oracle_calls_at_plan - calls_before,
                })
                break
            except Exception as exc:  # sandbox / AST / coercion errors -> repair
                code_errors += 1
                message = (
                    f"Execution error (round {round_i}): {type(exc).__name__}: {exc}\n"
                    "Fix the program and resubmit one fenced ```python``` block.\n"
                    "Remember: import statements are forbidden (deque, heapq, "
                    "itertools, and math are preloaded); map/filter are "
                    "unavailable --- use list comprehensions; do not reference "
                    "the same unmodified program twice."
                )
                record.update({
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "oracle_calls": len(oracle.query_log),
                    "calls_this_round": len(oracle.query_log) - calls_before,
                })
        messages.append({"role": "user", "content": message})

    if plan is not None:
        for action in plan:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    else:
        for _ in range(I4_PHASE2_ACTIONS):
            _, _, terminated, truncated, _ = env.consume_invalid_action("no_plan")
            if terminated or truncated:
                break
    trajectory = env.get_trajectory()
    return {
        "env_id": config.env_id,
        "family": config.family,
        "arm": "I4",
        "model": model,
        "success": trajectory["success"],
        "parse_failures": parse_failures,
        "code_errors": code_errors,
        "code_rounds": len(transcript),
        "query_log": oracle.query_log,
        "trajectory": trajectory,
        "transcript": transcript,
    }


def run_i5_episode(config: EnvConfig, env,
                   oracle_calls: int = I4_ORACLE_CALLS,
                   plan_cap: int = I4_PLAN_CAP) -> dict:
    """Deterministic BFS reference through the SAME counted oracle (I5)."""
    from collections import deque

    oracle = CounterfactualOracle(config, max_queries=oracle_calls)
    start = (env.state.agent_pos.row, env.state.agent_pos.col)
    target = tuple(config.target_pos)
    queue = deque([(start, [])])
    seen = {start}
    best: list[int] = []
    exhausted = False
    while queue:
        position, path = queue.popleft()
        if position == target:
            best = path
            break
        if len(path) >= plan_cap:
            continue
        if len(oracle.query_log) >= oracle.max_queries:
            exhausted = True
            break
        for action in range(config.n_actions):
            if len(oracle.query_log) >= oracle.max_queries:
                exhausted = True
                break
            reply = oracle.query(position[0], position[1], action)
            if reply.status != "ok":
                continue
            nxt = (reply.position[0], reply.position[1])
            if nxt == position or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [action]))
        if exhausted:
            break
    for action in best:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    trajectory = env.get_trajectory()
    return {
        "env_id": config.env_id,
        "family": config.family,
        "arm": "I5",
        "model": "bfs-over-counterfactual-oracle",
        "success": trajectory["success"],
        "parse_failures": 0,
        "plan_len": len(best),
        "budget_exhausted": exhausted,
        "query_log": oracle.query_log,
        "trajectory": trajectory,
        "transcript": [],
    }
