"""Task adapters and model callers for the NEXTSTEP J-protocol.

The adapter is the *only* place a task's semantics enter the J-arms: the arms
see ``initial_state`` / ``render_public_state`` / ``apply_real_action`` /
``oracle_transition`` / ``is_success`` and nothing else (§8.3).  Swapping the
F4 grid for the AlienWorkflow flagship must not require touching an arm.

Two task adapters ship here:

* :class:`F4Adapter` — wraps the frozen environment through ``ladder_config``
  and reuses the frozen ``CounterfactualOracle`` / strict parsers of
  ``interface_ladder_v2`` (DECISIONS D2).  The query space is the oracle's:
  ``(row, col, action) -> (row', col')`` for the canonical fresh state at a
  cell, exactly as the published I4 arm defined it.
* :class:`WorkflowAdapter` — wraps ``alienbody.workflow`` (S2), whose query
  space is ``(bit_state, action_id) -> bit_state``.

Callers: :class:`StubCaller` (scripted, no network — the regression harness of
§8.3) and :class:`GatewayCaller` (usage-aware wrapper that reuses the frozen
client's credential handling and request shaping, DECISIONS D3).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Protocol

from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig, GridState
from alienbody.interface_ladder_v2 import CounterfactualOracle, make_sandbox_next_state
from alienbody.state_v2 import render_state_v2, state_v2_payload

# ── model responses ────────────────────────────────────────────────────────

@dataclass
class ModelResponse:
    """One model request (after its retry chain).

    ``text is None`` marks a transport/context failure; every token field is
    ``None`` when the provider did not report it — never 0 (DECISIONS D12).
    """
    text: Optional[str]
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    raw: Optional[dict] = None
    attempts: int = 1
    transport_error: Optional[str] = None
    failure_kind: Optional[str] = None      # "transport" | "context"
    wall_seconds: float = 0.0

    @property
    def failed(self) -> bool:
        return self.text is None


class Caller(Protocol):
    provider: str
    model_id: str

    def call(self, system: str, user: str, max_tokens: int) -> ModelResponse: ...


class StubCaller:
    """Scripted caller for the offline regressions.  Never touches a network.

    A script is a list of entries consumed in order.  An entry is either a
    string (the response text), a dict carrying an explicit failure
    (``{"transport_error": str}`` / ``{"context_error": str}``), or any other
    JSON-shaped value (a plan, a J3L action object), which is serialized and
    served as the response text.  When the script is exhausted the caller
    returns a transport error, so a runaway arm fails visibly instead of
    looping.
    """
    provider = "stub"

    def __init__(self, model_id: str = "stub-model", script: Optional[list] = None):
        self.model_id = model_id
        self.script = list(script or [])
        self.prompts: list[tuple[str, str, int]] = []

    def call(self, system: str, user: str, max_tokens: int) -> ModelResponse:
        self.prompts.append((system, user, max_tokens))
        if not self.script:
            return ModelResponse(text=None, attempts=1, failure_kind="transport",
                                 transport_error="stub script exhausted")
        entry = self.script.pop(0)
        if isinstance(entry, str):
            return ModelResponse(text=entry, finish_reason="stop")
        if isinstance(entry, dict):
            if "transport_error" in entry:
                return ModelResponse(text=None, attempts=2, failure_kind="transport",
                                     transport_error=entry["transport_error"])
            if "context_error" in entry:
                return ModelResponse(text=None, attempts=1, failure_kind="context",
                                     transport_error=entry["context_error"])
            if "text" in entry:
                return ModelResponse(text=entry["text"],
                                     finish_reason=entry.get("finish_reason", "stop"))
        return ModelResponse(text=json.dumps(entry), finish_reason="stop")


class NullCaller:
    """The J5 reference arm makes no model calls; any call is a bug."""
    provider = "none"
    model_id = "none"

    def call(self, system: str, user: str, max_tokens: int) -> ModelResponse:
        raise RuntimeError("NullCaller: the J5 arm must not call a model")


class GatewayCaller:
    """Usage-aware wrapper around the frozen ``GatewayClient``.

    The frozen client's ``complete()`` returns text only, dropping usage and
    ``finish_reason``; the J-protocol records them, so this caller builds the
    request with the client's own helpers (credentials + body shaping) and
    reads the raw JSON itself.  Transport policy (§8.1): at most one retry of
    an **identical** payload; two failures end the episode as a transport
    failure.  Nothing here ever steers the model with an error message.

    ``retry_backoff_s`` delays the single retry.  The retry already existed;
    the delay only changes *when* the identical payload is re-sent, never what
    the model sees, and it is what makes the retry able to recover from a
    gateway hiccup: an immediate retry lands inside the same 502/429 window
    (measured 2026-09-15: the gateway rate-limits above ~3 calls/s per
    model and drops occasional 502s, so a ~750-call F4 J3L episode would survive
    only about half the time without it).  The default is 0 so unit tests do
    not sleep; every real run gets the value from the frozen config.
    """

    def __init__(self, client, model_id: str, provider: str = "gateway",
                 timeout_s: float = 180.0, post=None, retry_backoff_s: float = 0.0,
                 sleep=time.sleep):
        self._client = client
        self.model_id = model_id
        self.provider = provider
        self.timeout_s = timeout_s
        self._post = post          # injectable for tests; defaults to requests.post
        self.retry_backoff_s = float(retry_backoff_s)
        self._sleep = sleep        # injectable so tests do not wait

    def call(self, system: str, user: str, max_tokens: int) -> ModelResponse:
        import requests

        post = self._post or requests.post
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.time()
        last_err: Optional[str] = None
        for attempt in (1, 2):
            try:
                headers = self._client._build_headers()
                body = self._client._build_body(messages, max_tokens)
                resp = post(
                    self._client._api["end_point"], headers=headers, json=body,
                    timeout=self.timeout_s,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                text, _ = self._client._parse_response(data)
                usage = extract_usage(data)
                return ModelResponse(
                    text=text,
                    finish_reason=extract_finish_reason(data),
                    attempts=attempt,
                    wall_seconds=time.time() - started,
                    raw=data,
                    **usage,
                )
            except Exception as exc:  # noqa: BLE001 — recorded, never raised
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt == 1 and self.retry_backoff_s > 0:
                    self._sleep(self.retry_backoff_s)
        return ModelResponse(text=None, attempts=2, failure_kind=classify_failure(last_err),
                             transport_error=last_err, wall_seconds=time.time() - started)


_CONTEXT_HINTS = ("context length", "context_length", "maximum context",
                  "too many tokens", "context window", "input is too long")


def classify_failure(message: Optional[str]) -> str:
    """Context overflow is its own failure kind (§8.1), not a transport error."""
    lowered = (message or "").lower()
    return "context" if any(h in lowered for h in _CONTEXT_HINTS) else "transport"


def extract_usage(data: dict) -> dict:
    """Pull token fields out of any of the providers' response shapes."""
    usage = (data or {}).get("usage") or {}
    detail = (data or {}).get("detail") or {}
    if not usage and isinstance(detail, dict):
        usage = detail.get("usage") or {}
    if not usage and (data or {}).get("candidates"):
        usage = data.get("usageMetadata") or {}

    def _get(*names):
        for n in names:
            if n in usage and usage[n] is not None:
                return int(usage[n])
        return None

    input_tokens = _get("prompt_tokens", "input_tokens", "promptTokenCount")
    output_tokens = _get("completion_tokens", "output_tokens", "candidatesTokenCount")
    total = _get("total_tokens", "totalTokens", "totalTokenCount")
    if input_tokens is None and total is not None and output_tokens is not None:
        input_tokens = total - output_tokens
    details = usage.get("completion_tokens_details") or {}
    reasoning = _get("reasoning_tokens") or details.get("reasoning_tokens")
    cached = _get("cached_tokens", "cache_read_input_tokens") \
        or (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "cached_tokens": cached,
    }


def extract_finish_reason(data: dict) -> Optional[str]:
    for path in (("choices", 0, "finish_reason"),):
        node: Any = data
        for step in path:
            if isinstance(node, dict) and step in node:
                node = node[step]
            elif isinstance(node, list) and isinstance(step, int) and len(node) > step:
                node = node[step]
            else:
                node = None
                break
        if node:
            return str(node)
    detail = (data or {}).get("detail") or {}
    choices = detail.get("choices") if isinstance(detail, dict) else None
    if choices:
        return choices[0].get("finishReason") or choices[0].get("finish_reason")
    cands = (data or {}).get("candidates") or []
    if cands:
        return cands[0].get("finishReason")
    return None


# ── task adapters ──────────────────────────────────────────────────────────

class _BudgetExhausted(Exception):
    """Raised by the J5 reference when the shared oracle runs out mid-search."""


class OracleOutcome:
    """Result of one oracle interaction.

    ``served`` False means the attempt did not consume budget — currently only
    a query refused after exhaustion; either way the arm counts one oracle
    *attempt* (DECISIONS D7/D15).
    """
    __slots__ = ("answer", "status", "served")

    def __init__(self, answer: Optional[tuple], status: str, served: bool):
        self.answer = answer
        self.status = status
        self.served = served


class TaskAdapter(Protocol):
    """The whole surface a J-arm may use for one episode."""

    name: str
    n_actions: int
    oracle_cap: int
    action_cap: int

    def initial_state(self) -> None: ...
    def render_public_state(self) -> str: ...
    def state_handle(self) -> dict: ...
    def apply_real_action(self, action: int) -> str: ...
    def oracle_transition(self, query: tuple) -> OracleOutcome: ...
    def is_success(self) -> bool: ...
    def is_done(self) -> bool: ...
    def parse_query(self, text: str) -> Optional[tuple]: ...
    def render_query(self, query: tuple) -> str: ...
    def render_query_answer(self, query: tuple, answer: Optional[tuple], status: str) -> str: ...


@dataclass
class F4Adapter:
    """F4 grid task, scoped to the ladder episode (30 real actions)."""
    config: EnvConfig
    oracle_cap: int = 1000
    action_cap: int = 30
    name: str = "f4"
    #: how a J3L response spells the query origin: [row, col] for the grid,
    #: an integer bit vector for the workflow
    query_kind: str = "grid"
    _env: Optional[AlienBodyEnv] = None
    _oracle: Optional[CounterfactualOracle] = None

    def __post_init__(self) -> None:
        self._oracle = CounterfactualOracle(self.config, max_queries=self.oracle_cap)

    @property
    def n_actions(self) -> int:
        return self.config.n_actions

    @classmethod
    def from_suite(cls, env_file: str, action_cap: int = 30,
                   oracle_cap: int = 1000) -> "F4Adapter":
        from alienbody.interface_ladder_v3 import ladder_config  # read-only reuse

        raw = EnvConfig.from_file(env_file)
        return cls(config=ladder_config(raw, action_cap), action_cap=action_cap,
                   oracle_cap=oracle_cap)

    # -- episode lifecycle -------------------------------------------------
    def initial_state(self) -> None:
        from alienbody.interface_ladder_v3 import assert_state_budget

        self._env = AlienBodyEnv(self.config, render_mode="text")
        self._env.reset()
        self._env.start_phase2()
        assert_state_budget(self.config, self._env.state, self.action_cap)

    @property
    def env(self) -> AlienBodyEnv:
        if self._env is None:
            raise RuntimeError("initial_state() must be called first")
        return self._env

    @property
    def oracle(self) -> CounterfactualOracle:
        assert self._oracle is not None
        return self._oracle

    @property
    def oracle_calls_served(self) -> int:
        return len(self.oracle.query_log)

    # -- what every arm sees ----------------------------------------------
    def render_public_state(self) -> str:
        return render_state_v2(self.config, self.env.state)

    def state_handle(self) -> dict:
        """Compact public handle for the event table (static parts omitted)."""
        agent = state_v2_payload(self.config, self.env.state)["agent"]
        return {"agent": agent}

    def render_event_row_state(self, handle: dict) -> str:
        a = handle["agent"]
        row, col = a["position"]
        return f"(r{row},c{col}) d={a['direction']} color={a['color']}"

    # -- real actions ------------------------------------------------------
    def apply_real_action(self, action: int) -> str:
        if not isinstance(action, int) or isinstance(action, bool) \
                or not 0 <= action < self.n_actions:
            return "invalid"
        self.env.step(action)          # the environment itself, never a substitute
        return "ok"

    def consume_invalid_action(self, reason: str) -> None:
        """Spend one real step without applying an action (§8.3, D15)."""
        self.env.consume_invalid_action(reason)

    def is_success(self) -> bool:
        return bool(self.env.state.success)

    def is_done(self) -> bool:
        return bool(self.env.state.done)

    # -- oracle ------------------------------------------------------------
    def oracle_transition(self, query: tuple) -> OracleOutcome:
        if len(query) != 3:
            return OracleOutcome(None, "malformed_query", False)
        row, col, action = query
        reply = self.oracle.query(int(row), int(col), int(action))
        if reply.status == "ok":
            return OracleOutcome(tuple(reply.position), "ok", True)
        # The frozen oracle spends a budget slot on every query except the ones
        # it refuses once the budget is gone; an invalid query therefore costs
        # an attempt (8.3) and is not free.
        return OracleOutcome(None, reply.status, reply.status != "budget_exhausted")

    def parse_query(self, text: str) -> Optional[tuple]:
        import re
        m = re.fullmatch(r"\s*CALL\s+next_state\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*", text or "", re.I)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    def render_query(self, query: tuple) -> str:
        return f"next_state({query[0]},{query[1]},{query[2]})"

    def render_query_answer(self, query: tuple, answer: Optional[tuple], status: str) -> str:
        if answer is None:
            return f"next_state({query[0]},{query[1]},{query[2]}) -> {status}"
        return (f"next_state({query[0]},{query[1]},{query[2]}) -> "
                f"(r{answer[0]},c{answer[1]})")

    def sandbox_next_state(self) -> Callable[..., tuple]:
        """The same callable the J4 sandbox exposes (frozen v2 implementation)."""
        return make_sandbox_next_state(self.config, self.oracle)

    def sandbox_state(self) -> dict:
        """The parsed public payload the J4 sandbox exposes as ``STATE``."""
        return state_v2_payload(self.config, self.env.state)

    # -- J5 -----------------------------------------------------------------
    def bfs_plan(self, cap: int = 30) -> Optional[list]:
        """Deterministic reference plan through the shared oracle."""
        from alienbody.env.actions import apply_action
        from alienbody.env.grid import Phase, Position

        def successor(row, col, action):
            reply = self.oracle.query(row, col, action)
            if reply.status == "budget_exhausted":
                raise _BudgetExhausted()
            return reply.position if reply.status == "ok" else (row, col)

        target = tuple(self.config.target_pos)
        start = (self.env.state.agent_pos.row, self.env.state.agent_pos.col)
        if start == target:
            return []
        from collections import deque
        queue = deque([(start, [])])
        seen = {start}
        try:
            while queue:
                pos, plan = queue.popleft()
                if len(plan) >= cap:
                    continue
                for a in range(self.n_actions):
                    nxt = successor(pos[0], pos[1], a)
                    if nxt in seen:
                        continue
                    new_plan = plan + [a]
                    if nxt == target:
                        return new_plan
                    seen.add(nxt)
                    queue.append((nxt, new_plan))
        except _BudgetExhausted:
            # The reference is budget-limited too; a capped BFS is reported as
            # a failure, never silently answered from a wrong graph (1.4).
            return None
        return None


@dataclass
class WorkflowAdapter:
    """AlienWorkflow (S2, W-K: the tool schemas are public) on the same surface.

    The workflow's public state is the flag vector, the goal's partial
    assignment and the exact four masks of every tool (TODO_NEXTSTEP 2.2); the
    grading-only labels (instance id, difficulty, shortest plan, fingerprint)
    never leave the instance.  The oracle is the frozen
    :func:`alienbody.workflow.env.oracle_next_state` under the same budget rules
    as F4.

    Note for the analysis (2.4): because the masks are public, a code arm may
    reproduce the transition itself.  That is not an information violation, and
    ``oracle_served == 0`` must never be read as "did not simulate".
    """
    instance: Any
    action_cap: int = 24
    oracle_cap: int = 2048
    contract: str = "STATE_WORKFLOW"
    name: str = "workflow"
    query_kind: str = "bitvec"
    _env: Any = None
    _oracle_calls: int = 0

    def __post_init__(self) -> None:
        from alienbody.workflow.env import WorkflowEnv

        self._env = WorkflowEnv(self.instance, step_budget=self.action_cap)

    @classmethod
    def from_payload(cls, payload: dict, action_cap: int = 24,
                     oracle_cap: int = 2048) -> "WorkflowAdapter":
        from alienbody.workflow.env import WorkflowInstance

        return cls(instance=WorkflowInstance.from_dict(payload),
                   action_cap=action_cap, oracle_cap=oracle_cap)

    @property
    def n_actions(self) -> int:
        return int(self.instance.m)

    @property
    def env(self):
        if self._env is None:
            raise RuntimeError("initial_state() must be called first")
        return self._env

    @property
    def bit_state(self) -> int:
        return int(self.env.state)

    # -- episode lifecycle -------------------------------------------------
    def initial_state(self) -> None:
        from alienbody.workflow.env import WorkflowEnv

        self._env = WorkflowEnv(self.instance, step_budget=self.action_cap)
        self._env.reset()
        self._oracle_calls = 0

    # -- what every arm sees ----------------------------------------------
    def payload(self) -> dict:
        """The public STATE payload: no grading-only field may appear."""
        from alienbody.workflow.observation import assert_public, format_action

        k = int(self.instance.k)
        state = self.bit_state
        goal_on = int(self.instance.goal_on)
        goal_off = int(self.instance.goal_off)
        payload = {
            "contract": self.contract,
            "k": k,
            "m": self.n_actions,
            "state_bits": format(state, "0%db" % k),
            "state": state,
            "goal_on": goal_on,
            "goal_off": goal_off,
            "goal_literals": [("f%d=%d" % (i, (goal_on >> i) & 1))
                              for i in range(k - 1, -1, -1)
                              if ((goal_on | goal_off) >> i) & 1],
            "actions_remaining": max(0, self.action_cap - int(self.env.steps_used)),
            "tools": [{"id": i, "pre_on": int(a[0]), "pre_off": int(a[1]),
                       "set_on": int(a[2]), "set_off": int(a[3]),
                       "schema": format_action(a, k)}
                      for i, a in enumerate(self.instance.actions)],
        }
        return assert_public(payload)

    def render_public_state(self) -> str:
        return self.contract + "\n" + json.dumps(self.payload(), sort_keys=True,
                                                  separators=(",", ":"))

    def state_handle(self) -> dict:
        return {"state": self.bit_state}

    def render_event_row_state(self, handle: dict) -> str:
        return format(int(handle["state"]), "0%db" % int(self.instance.k))

    # -- real actions ------------------------------------------------------
    def apply_real_action(self, action: int) -> str:
        if not isinstance(action, int) or isinstance(action, bool) \
                or not 0 <= action < self.n_actions:
            return "invalid"
        self.env.step(action)          # the environment itself, never a substitute
        return "ok"

    def consume_invalid_action(self, reason: str) -> None:
        """Spend one real step: an out-of-range id leaves the state unchanged."""
        self.env.step(-1)

    def is_success(self) -> bool:
        from alienbody.workflow.env import is_success as _ok
        return bool(_ok(self.instance, self.env.state))

    def is_done(self) -> bool:
        return self.env.steps_left <= 0 or self.is_success()

    # -- oracle ------------------------------------------------------------
    @property
    def oracle_calls_served(self) -> int:
        return self._oracle_calls

    def oracle_transition(self, query: tuple) -> OracleOutcome:
        from alienbody.workflow.env import oracle_next_state

        if len(query) != 2:
            return OracleOutcome(None, "malformed_query", False)
        bits, action = int(query[0]), int(query[1])
        if self._oracle_calls >= self.oracle_cap:
            return OracleOutcome(None, "budget_exhausted", False)
        # as in F4: any query that is not already refused consumes budget
        self._oracle_calls += 1
        if not 0 <= bits < (1 << int(self.instance.k)):
            return OracleOutcome(None, "invalid_state", True)
        if not 0 <= action < self.n_actions:
            return OracleOutcome(None, "invalid_action", True)
        return OracleOutcome((int(oracle_next_state(self.instance, bits, action)),),
                             "ok", True)

    def parse_query(self, text: str) -> Optional[tuple]:
        import re
        m = re.fullmatch(r"\s*CALL\s+next_state\((\d+)\s*,\s*(\d+)\)\s*",
                         text or "", re.I)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)))

    def render_query(self, query: tuple) -> str:
        return "next_state(%d,%d)" % (query[0], query[1])

    def render_query_answer(self, query: tuple, answer: Optional[tuple], status: str) -> str:
        k = int(self.instance.k)
        bits = format(int(query[0]), "0%db" % k)
        if answer is None:
            return "next_state(%s,%d) -> %s" % (bits, query[1], status)
        return "next_state(%s,%d) -> %s" % (bits, query[1],
                                            format(int(answer[0]), "0%db" % k))

    # -- J4 sandbox surface (the same oracle callable as J3L) --------------
    def sandbox_state(self) -> dict:
        return self.payload()

    def sandbox_next_state(self) -> Callable[..., int]:
        def next_state(state: int, action: int) -> int:
            outcome = self.oracle_transition((int(state), int(action)))
            if outcome.status == "budget_exhausted":
                raise RuntimeError("next_state budget exhausted (%d)" % self.oracle_cap)
            if outcome.status != "ok":
                raise RuntimeError("next_state rejected: %s" % outcome.status)
            return int(outcome.answer[0])
        return next_state

    # -- J5 -----------------------------------------------------------------
    def bfs_plan(self, cap: int = 24) -> Optional[list]:
        """Deterministic reference plan through the shared oracle."""
        from alienbody.workflow.baselines import shortest_plan

        def transition(instance, state, action_id):
            outcome = self.oracle_transition((int(state), int(action_id)))
            if outcome.status == "budget_exhausted":
                raise _BudgetExhausted()
            if outcome.status != "ok":
                return int(state)
            return int(outcome.answer[0])

        try:
            plan = shortest_plan(self.instance, transition=transition, max_len=cap)
        except _BudgetExhausted:
            return None
        return None if plan is None else list(plan)
