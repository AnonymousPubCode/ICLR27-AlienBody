"""The J-arm interfaces (TODO_NEXTSTEP.md §1.2, §8.3) — one episode loop.

Every arm sees the same explicit STATE, the same exact mapping, the same goal
and the same remaining-action count.  What differs is the *channel*:

===========  =========================  ==============  ============
arm          memory across calls         oracle          program
===========  =========================  ==============  ============
J0           event table                no              no
J1           event table + draft        no              no
J2           initial STATE only         no              no
J3L          event table + ledger + draft (fresh context each round)  yes  no
J4           event table + ledger, exported facts; program memory per round  yes  yes
J5           deterministic BFS          same oracle     reference
===========  =========================  ==============  ============

The arms are a **new frozen protocol**, not renamed I-arms: J0's shared event
table, J3L's fresh-context ledger and J1's draft rules do not exist in v3/v4
and their numbers must never be pooled with the old ones (§1.2).
"""
from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from alienbody.nextstep.adapters import Caller, ModelResponse, OracleOutcome, TaskAdapter
from alienbody.nextstep.records import (
    EpisodeRecord, Event, ModelCall, Query, sha256_json_canonical, sha256_text,
    utc_now,
)

# ── budgets (TODO_NEXTSTEP.md §8.1) ────────────────────────────────────────

ARM_RESPONSE_TOKENS = {
    # D33 (08.5.2's budget rule): an arm's cap is raised to the tier above the
    # largest output its models actually produced, because reasoning tokens are
    # counted inside `output_tokens` — a cap that ignores them starves the
    # visible answer.  16000 is the smaller of the two providers' documented
    # maxima (dsv4-lh 16000, gemini 65536) and both accept it (probe artifacts
    # `results/nextstep/probes/maxtok_*.json`).
    #
    # D33b (raised again 2026-09-15, same rule, applied to the re-measurement):
    # at 1024 gemini's J0 thinking sits *at* the cap (median 981, and the two
    # calls that reported `length` had crossed it at 1069/1090), and at 16000
    # gemini's J2 one-shot plan still lost one reply of eleven; both move to the
    # next tier.  The cost of a raise is bounded by the calls that were
    # previously cut — gemini's per-call thinking has a natural length (~980 on
    # J0, ~15 450 on J2) and does not grow with the budget, measured over 682
    # J0 calls at 128 and 11 J2 calls at 16000.
    "J0": 2048,
    "J1": 16000,    # must hold a 2000-character draft plus the JSON wrapper
    "J2": 32768,    # gemini's one-shot plan needs > 16000 on the tail
    "J3L": 16000,   # draft + structured object
    "J4": 16000,
    "J5": 0,
}
DRAFT_MAX_CHARS = 2000
CODE_SUBMISSIONS_MAX = 4
J2_PLAN_CAP = 30

#: J3L's model-call cap: the oracle budget plus two calls per real action plus
#: slack; every CALL/ACTION/format failure counts (§8.1).
def j3l_model_call_cap(oracle_cap: int, action_cap: int) -> int:
    return oracle_cap + 2 * action_cap + 4


ARMS = ("J0", "J1", "J2", "J3L", "J4", "J5")

# ── shared prompt text ─────────────────────────────────────────────────────

_TABLE_HELP = """Each round you receive:
  STATE          — the complete public state (JSON), identical for every interface.
  EVENT TABLE    — every real action so far, in order: the public state before it, the action id, and the state after it. A row marked INVALID spent one step without applying an action.
  ACTIONS REMAINING — how many real actions are left. The environment stops after that many attempts, valid or not; there is no further signal at the end."""

SYSTEM_J0 = f"""You control an agent in a discrete task. The action mapping and the target in STATE are exact.

{_TABLE_HELP}

Output ONLY one integer action id, e.g. 2. Do not explain."""

SYSTEM_J1 = f"""You control an agent in a discrete task. The action mapping and the target in STATE are exact.

{_TABLE_HELP}

You keep a persistent draft of at most {DRAFT_MAX_CHARS} characters across rounds: it is your own note, it is never executed, and it is the only memory you carry besides the table.

Output ONLY a JSON object with exactly two keys:
{{"draft": "your updated notes", "action": N}}
where N is one legal action id. Nothing outside the JSON."""

SYSTEM_J2 = f"""You control an agent in a discrete task. The action mapping and the target in STATE are exact.

You receive the initial STATE only: you cannot revise your plan and you cannot query a simulator.

Output ONLY a JSON list of 1 to {J2_PLAN_CAP} action ids, e.g. [2,1,2]. The whole list is checked first and then executed in order without revision. A list that does not parse, is empty or is longer than {J2_PLAN_CAP} actions ends the episode as a failure. Do not explain."""

def _j3l_prompt(state_form: str, query_help: str) -> str:
    """J3L's prompt; only the query's state syntax is task-specific."""
    return f"""You control an agent in a discrete task. The action mapping and the target in STATE are exact.

{_TABLE_HELP}
  QUERY LEDGER   — every distinct simulator query you have made, with its answer and how many times you have asked it. Asking again is allowed and answered again, but it costs the same as a new query.

You may either take one real action or ask the simulator one question about a single step. Each round starts a fresh context: only the STATE, the EVENT TABLE, the QUERY LEDGER and your draft survive. Queries cost nothing in real steps but a fixed query budget; once it is gone the simulator refuses and you must act.

You keep a persistent draft of at most {DRAFT_MAX_CHARS} characters across rounds.

Output ONLY one JSON object:
  {{"kind": "action", "action": N, "draft": "..."}}
  {{"kind": "query", "state": {state_form}, "action": N, "draft": "..."}}
The query form asks: {query_help} Nothing outside the JSON."""


SYSTEM_J3L = _j3l_prompt("[row, col]", "from cell (row, col), what does action N do?")
SYSTEM_J3L_WF = _j3l_prompt("S", "from the integer flag-state S, what does tool N do?")


def _j4_prompt(callable_lines: str, plan_help: str) -> str:
    """J4's prompt; only the oracle callable's signature is task-specific."""
    return f"""You control an agent in a discrete task. The action mapping and the target in STATE are exact.

{_TABLE_HELP}
  QUERY LEDGER   — every distinct simulator query made so far with its answer. Every call your program makes costs the same budget, including repeats.

You must write ONE Python program that finds and returns a complete action plan.

Inside the sandbox:
  STATE            dict, the parsed STATE payload (keys exactly as in the JSON above)
{callable_lines}
  deque, heapq, itertools, math and safe builtins only. No imports, no files,
  no network, no access to the real environment.

End your program by setting exactly one of:
  PLAN = [a0, a1, ...]   # {plan_help}
  ACTION = <int>         # only the next action

Output ONLY one fenced ```python``` block. If your program errors or returns no plan you receive the error and may submit a corrected program, at most {CODE_SUBMISSIONS_MAX} submissions in total; all submissions share the one query budget. Do not narrate."""


SYSTEM_J4 = _j4_prompt(
    """  next_state(r,c,a) -> (nr,nc)
                   true one-step successor of action a from cell (r,c); every
                   call spends the shared query budget; out-of-range (r,c)
                   returns (r,c) unchanged and counts; an illegal action id or an
                   exhausted budget raises an error""",
    "1..30 actions from START to TARGET",
)
SYSTEM_J4_WF = _j4_prompt(
    """  next_state(S,t) -> S2
                   true one-step successor of tool t from the integer flag-state
                   S (the integer STATE shows); every call spends the shared
                   query budget; an out-of-range state or tool id raises an
                   error, as does an exhausted budget""",
    "1..30 tools from START to TARGET",
)

SYSTEM_PROMPTS = {
    "J0": SYSTEM_J0, "J1": SYSTEM_J1, "J2": SYSTEM_J2,
    "J3L": SYSTEM_J3L, "J4": SYSTEM_J4, "J5": None,
}

#: Only the query/callable syntax differs between tasks; every other arm's
#: prompt is byte-identical (the table help is task-neutral by design).
SYSTEM_PROMPTS_BY_TASK = {
    "f4": dict(SYSTEM_PROMPTS),
    "workflow": {**SYSTEM_PROMPTS, "J3L": SYSTEM_J3L_WF, "J4": SYSTEM_J4_WF},
}


def system_prompt(arm: str, task: str = "f4") -> Optional[str]:
    return SYSTEM_PROMPTS_BY_TASK.get(task, SYSTEM_PROMPTS)[arm]



def prompt_sha256(arm: str, task: str = "f4") -> Optional[str]:
    text = system_prompt(arm, task)
    return sha256_text(text) if text else None


# What each arm's parse outcome is decided by: the entry parser plus the
# helpers it shares with its siblings.  Hashed into config.frozen.json so the
# acceptance of a formal batch can be checked against the code that produced it.
_PARSER_SOURCES = {
    "J0": ("parse_action_j0", "_classify_unparsed", "_ACTION_ONLY"),
    "J1": ("parse_draft_action_j1", "_classify_unparsed", "_ACTION_ONLY"),
    "J2": ("parse_plan_j2", "_classify_unparsed"),
    "J3L": ("parse_j3l", "_classify_unparsed", "_ACTION_ONLY"),
    "J4": ("extract_code", "_FENCE_RE"),
    "J5": (),
}


def parser_sha256(arm: str) -> Optional[str]:
    """sha256 over the arm's parser source (empty tuple -> None for J5)."""
    names = _PARSER_SOURCES.get(arm)
    if not names:
        return None
    chunks = []
    for name in names:
        obj = globals().get(name)
        if obj is None:
            return None
        text = obj.pattern if hasattr(obj, "pattern") else inspect.getsource(obj)
        chunks.append(f"# {name}\n{text}")
    return sha256_text("\n".join(chunks))


def interface_hashes(task: str = "f4") -> dict:
    """Everything about the interfaces that a frozen config must pin."""
    return {
        arm: {"prompt_sha256": prompt_sha256(arm, task),
              "parser_sha256": parser_sha256(arm),
              "response_max_tokens": ARM_RESPONSE_TOKENS[arm]}
        for arm in ARMS
    }


# ── the shared renderers (identical bytes across J0/J1/J3L) ────────────────

def render_event_table(adapter: TaskAdapter, events: list[Event]) -> str:
    """The one event table every stepwise arm re-renders, byte for byte.

    Rows carry the *public* state handle per step, so the table is an exact
    replay of the public history and nothing more.
    """
    lines = ["EVENT TABLE (oldest first):"]
    if not events:
        lines.append("  (empty: no real action has been taken yet)")
        return "\n".join(lines)
    for e in events:
        before = e.state_before
        if e.status == "ok":
            lines.append(f"  [{e.index}] {before} -- action {e.action} --> {e.state_after}")
        else:
            lines.append(f"  [{e.index}] {before} -- INVALID ({e.note}) --> {e.state_after} (step spent)")
    return "\n".join(lines)


def render_query_ledger(adapter: TaskAdapter, queries: list[Query]) -> str:
    """Ledger view: distinct queries in first-seen order, with repeat counts."""
    if not queries:
        return "QUERY LEDGER: (empty)"
    lines = ["QUERY LEDGER (first-seen order; count = times asked):"]
    for q in queries:
        answer = adapter.render_query_answer(tuple(q.key), tuple(q.answer) if q.answer else None,
                                             q.status)
        count = f" x{q.count}" if q.count > 1 else ""
        lines.append(f"  [{q.index}] {answer}{count}")
    return "\n".join(lines)


def build_state_block(adapter: TaskAdapter, events: list[Event],
                      queries: Optional[list[Query]] = None,
                      draft: Optional[str] = None,
                      extra: str = "") -> str:
    """The user message: STATE + table (+ ledger) (+ draft) (+ extra)."""
    parts = [
        "STATE:",
        adapter.render_public_state(),
        "",
        render_event_table(adapter, events),
        "",
        f"ACTIONS REMAINING: {max(0, adapter.action_cap - len(events))}",
    ]
    if queries is not None:
        parts += ["", render_query_ledger(adapter, queries)]
    if draft is not None:
        parts += ["", f"DRAFT ({len(draft)}/{DRAFT_MAX_CHARS} chars):", draft or "(empty)"]
    if extra:
        parts += ["", extra]
    return "\n".join(parts)


# ── parse verdicts ─────────────────────────────────────────────────────────

@dataclass
class Verdict:
    """What one model response turned out to be."""
    kind: str                      # "action" | "query" | "plan" | "none"
    action: Optional[int] = None
    query: Optional[tuple] = None
    plan: Optional[list] = None
    draft: Optional[str] = None
    failure: Optional[str] = None  # a label from records.INVALID_ATTEMPT_KINDS
    code: Optional[str] = None


_ACTION_ONLY = re.compile(r"\s*([0-9]+)\s*")


def _classify_unparsed(response: Optional[str]) -> str:
    if not response or not response.strip():
        return "other"
    if re.search(r"```[A-Za-z0-9_+-]*\s*$", response.rstrip()):
        return "fenced_json"
    stripped = re.sub(r'"(?:\\.|[^"\\])*"', '""', response)
    if stripped.count("{") > stripped.count("}") or stripped.count("[") > stripped.count("]"):
        return "unterminated_json"
    return "not_json"


def parse_action_j0(response: Optional[str], n_actions: int) -> Verdict:
    m = _ACTION_ONLY.fullmatch(response or "")
    if not m:
        return Verdict("none", failure=_classify_unparsed(response))
    action = int(m.group(1))
    if not 0 <= action < n_actions:
        return Verdict("none", failure="action_out_of_range")
    return Verdict("action", action=action)


def parse_draft_action_j1(response: Optional[str], n_actions: int,
                          max_chars: int = DRAFT_MAX_CHARS) -> Verdict:
    if not response:
        return Verdict("none", failure="other")
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return Verdict("none", failure=_classify_unparsed(response))
    if not isinstance(value, dict) or set(value) != {"draft", "action"} \
            or not isinstance(value["draft"], str):
        return Verdict("none", failure="not_json")
    draft = value["draft"]
    action = value["action"]
    if not isinstance(action, int) or isinstance(action, bool):
        return Verdict("none", failure="not_json")
    if not 0 <= action < n_actions:
        return Verdict("none", failure="action_out_of_range")
    if len(draft) > max_chars:
        # The cap is enforced by rejection, never by trimming (§8.1).
        return Verdict("none", failure="draft_over_cap", draft=draft)
    return Verdict("action", action=action, draft=draft)


def parse_j3l(response: Optional[str], adapter: TaskAdapter,
              max_chars: int = DRAFT_MAX_CHARS) -> Verdict:
    """J3L's one structured object: kind=query/action, state, action, scratchpad.

    The frozen grammar also accepts ``scratchpad`` as the draft key (the plan
    names it that way); the two spellings are the *only* accepted variants and
    are documented in the protocol before any formal run.
    """
    if not response:
        return Verdict("none", failure="other")
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        return Verdict("none", failure=_classify_unparsed(response))
    if not isinstance(value, dict) or value.get("kind") not in ("query", "action"):
        return Verdict("none", failure="not_json")
    draft = value.get("draft", value.get("scratchpad", ""))
    if not isinstance(draft, str):
        return Verdict("none", failure="not_json")
    if len(draft) > max_chars:
        return Verdict("none", failure="draft_over_cap", draft=draft)
    action = value.get("action")
    if not isinstance(action, int) or isinstance(action, bool):
        return Verdict("none", failure="action_out_of_range")
    if value["kind"] == "action":
        if not 0 <= action < adapter.n_actions:
            return Verdict("none", failure="action_out_of_range")
        return Verdict("action", action=action, draft=draft)
    # kind == "query": the state is task-specific (F4: [row, col]; workflow: bits)
    state = value.get("state")
    if getattr(adapter, "query_kind", "grid") == "grid":
        if not isinstance(state, list) or len(state) != 2 \
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in state):
            return Verdict("none", failure="query_out_of_range")
        query = (int(state[0]), int(state[1]), int(action))
    else:
        if not isinstance(state, int) or isinstance(state, bool):
            return Verdict("none", failure="query_out_of_range")
        query = (int(state), int(action))
    return Verdict("query", query=query, draft=draft)


def parse_plan_j2(response: Optional[str], n_actions: int,
                  cap: int = J2_PLAN_CAP) -> Verdict:
    if not response:
        return Verdict("none", failure="other")
    try:
        plan = json.loads(response)
    except json.JSONDecodeError:
        return Verdict("none", failure=_classify_unparsed(response))
    if not isinstance(plan, list):
        return Verdict("none", failure="plan_not_list")
    if not plan:
        return Verdict("none", failure="plan_empty")
    if len(plan) > cap:
        return Verdict("none", failure="plan_over_cap")
    for entry in plan:
        if not isinstance(entry, int) or isinstance(entry, bool) \
                or not 0 <= entry < n_actions:
            return Verdict("none", failure="plan_action_out_of_range")
    return Verdict("plan", plan=list(plan))


_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_code(response: Optional[str]) -> Optional[str]:
    if not response:
        return None
    m = _FENCE_RE.search(response)
    return m.group(1) if m else None


# ── the episode loop ───────────────────────────────────────────────────────

@dataclass
class EpisodeMeta:
    """Everything the record needs that comes from outside the loop."""
    protocol_id: str
    dataset_sha256: str
    env_id: str
    model_id: str
    provider: str
    inference_seed: int
    config_sha256: str
    code_sha256_at_launch: str
    partition: str = "dev"
    block_id: str = ""
    attempt_id: str = ""


def run_episode(arm: str, adapter: TaskAdapter, caller: Caller, meta: EpisodeMeta,
                task: str = "f4") -> EpisodeRecord:
    """Run one J-arm episode.  Offline-testable with any :class:`Caller`."""
    started = utc_now()
    t0 = time.time()
    adapter.initial_state()

    events: list[Event] = []
    queries: list[Query] = []          # one entry per ledger key, in first-seen order
    ledger_index: dict[tuple, Query] = {}
    model_calls: list[ModelCall] = []
    invalid_kinds: dict[str, int] = {}
    parse_failures = 0
    transport_failures = 0
    code_errors = 0
    code_submissions = 0
    oracle_attempts = 0
    oracle_served = 0
    draft: Optional[str] = None
    termination = "internal_error"
    note = ""
    state_initial = sha256_text(adapter.render_public_state())

    oracle_cap = adapter.oracle_cap
    action_cap = adapter.action_cap
    model_call_cap = j3l_model_call_cap(oracle_cap, action_cap) if arm == "J3L" else None
    last_failure_kind: list = ["transport"]

    def failed_termination() -> str:
        """A failed call ends the episode; context overflow is its own reason."""
        return "context_failure" if last_failure_kind[0] == "context" else "transport_failure"

    def call(system: str, user: str, max_tokens: int, round_i: int) -> Optional[ModelResponse]:
        nonlocal transport_failures
        resp = caller.call(system, user, max_tokens)
        model_calls.append(ModelCall(
            index=len(model_calls) + 1, round=round_i,
            request_sha256=sha256_text(system + "\0" + user),
            response_sha256=sha256_text(resp.text) if resp.text is not None else None,
            response_text=resp.text, finish_reason=resp.finish_reason,
            attempts=resp.attempts, transport_error=resp.transport_error,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            reasoning_tokens=resp.reasoning_tokens, cached_tokens=resp.cached_tokens,
            wall_seconds=resp.wall_seconds,
        ))
        if resp.failed:
            transport_failures += 1
            last_failure_kind[0] = resp.failure_kind or "transport"
            return None
        return resp

    def note_invalid(kind: str) -> None:
        nonlocal parse_failures
        parse_failures += 1
        invalid_kinds[kind] = invalid_kinds.get(kind, 0) + 1

    def record_action(action: Optional[int], kind: str, status: str) -> None:
        before = adapter.state_handle()
        before_text = adapter.render_event_row_state(before)
        if status == "ok":
            adapter.apply_real_action(action)
        else:
            adapter.consume_invalid_action(kind)
        after = adapter.state_handle()
        after_text = adapter.render_event_row_state(after)
        events.append(Event(
            index=len(events), status=status, action=action,
            state_before=before_text, state_after=after_text,
            state_before_sha256=sha256_text(before_text),
            state_after_sha256=sha256_text(after_text),
            note="" if status == "ok" else kind,
        ))

    def ask(query: tuple) -> None:
        nonlocal oracle_attempts, oracle_served
        oracle_attempts += 1
        outcome: OracleOutcome = adapter.oracle_transition(query)
        if outcome.served:
            oracle_served += 1
        existing = ledger_index.get(query)
        if existing is not None:
            existing.count += 1
            existing.status = outcome.status
            if outcome.answer is not None:
                existing.answer = list(outcome.answer)
            existing.served = existing.served or outcome.served
            return
        entry = Query(index=len(queries) + 1, key=list(query),
                      answer=list(outcome.answer) if outcome.answer else None,
                      status=outcome.status, count=1, served=outcome.served)
        queries.append(entry)
        ledger_index[query] = entry

    def execute_plan(plan: list, execution_cap: int) -> str:
        """Run a plan on the real environment; returns the termination reason."""
        for action in plan:
            if len(events) >= execution_cap:
                return "action_budget_exhausted"
            ok = 0 <= action < adapter.n_actions
            record_action(action, "action_out_of_range", "ok" if ok else "invalid")
            if not ok:
                continue
            if adapter.is_success():
                return "success"
            if adapter.is_done():
                return "action_budget_exhausted"
        return "plan_exhausted"

    if arm == "J5":
        plan = adapter.bfs_plan(cap=min(action_cap, J2_PLAN_CAP))
        if plan is None:
            termination = "oracle_exhausted_terminal"
            note = "BFS reference found no plan within the shared oracle budget"
        else:
            termination = execute_plan(plan, action_cap)
        # The reference's oracle use is the search itself.
        oracle_attempts = adapter.oracle_calls_served
        oracle_served = adapter.oracle_calls_served
        return _finish(arm, adapter, meta, started, t0, events, queries, model_calls,
                       invalid_kinds, parse_failures, transport_failures, code_errors,
                       code_submissions, oracle_attempts, oracle_served, termination,
                       note, state_initial, draft)

    if arm == "J2":
        system = SYSTEM_J2
        user = build_state_block(adapter, events)
        resp = call(system, user, ARM_RESPONSE_TOKENS["J2"], 0)
        if resp is None:
            termination = failed_termination()
        else:
            verdict = parse_plan_j2(resp.text, adapter.n_actions)
            if verdict.kind != "plan":
                note_invalid(verdict.failure or "other")
                termination = ("plan_over_cap" if verdict.failure == "plan_over_cap"
                               else "parse_failed")
                note = f"plan rejected: {verdict.failure}"
            else:
                termination = execute_plan(verdict.plan, action_cap)
        return _finish(arm, adapter, meta, started, t0, events, queries, model_calls,
                       invalid_kinds, parse_failures, transport_failures, code_errors,
                       code_submissions, oracle_attempts, oracle_served, termination,
                       note, state_initial, draft)

    if arm == "J4":
        system = system_prompt("J4", task)
        plan: Optional[list] = None
        prior_code: list[str] = []
        prior_errors: list[str] = []
        for round_i in range(CODE_SUBMISSIONS_MAX):
            if adapter.is_success():
                break
            user = build_state_block(adapter, events, queries=queries,
                                     extra=_j4_extra(prior_code, prior_errors))
            resp = call(system, user, ARM_RESPONSE_TOKENS["J4"], round_i)
            if resp is None:
                termination = failed_termination()
                break
            code = extract_code(resp.text)
            if code is None:
                code_errors += 1
                prior_errors.append("no fenced python block found")
                continue
            code_submissions += 1
            result, error = _run_program(adapter, code)
            if error is not None:
                code_errors += 1
                prior_code.append(code)
                prior_errors.append(error)
                continue
            try:
                plan = _coerce_program_result(result, adapter)
            except ValueError as exc:
                code_errors += 1
                prior_code.append(code)
                prior_errors.append(str(exc))
                continue
            break
        if termination == "internal_error":
            if plan is None:
                termination = "plan_invalid" if code_errors else "internal_error"
                note = "no usable plan after the allowed submissions"
            else:
                termination = execute_plan(plan, action_cap)
        oracle_attempts = adapter.oracle_calls_served
        oracle_served = adapter.oracle_calls_served
        return _finish(arm, adapter, meta, started, t0, events, queries, model_calls,
                       invalid_kinds, parse_failures, transport_failures, code_errors,
                       code_submissions, oracle_attempts, oracle_served, termination,
                       note, state_initial, draft)

    # ── stepwise arms: J0, J1, J3L ────────────────────────────────────────
    system = system_prompt(arm, task)
    rounds = 0
    while len(events) < action_cap:
        if model_call_cap is not None and rounds >= model_call_cap:
            termination = "model_call_cap"
            note = f"model-call cap {model_call_cap} reached"
            break
        rounds += 1
        use_ledger = arm == "J3L"
        user = build_state_block(adapter, events,
                                 queries=queries if use_ledger else None,
                                 draft=draft)
        resp = call(system, user, ARM_RESPONSE_TOKENS[arm], rounds)
        if resp is None:
            termination = failed_termination()
            break
        if arm == "J0":
            verdict = parse_action_j0(resp.text, adapter.n_actions)
        elif arm == "J1":
            verdict = parse_draft_action_j1(resp.text, adapter.n_actions)
        else:
            verdict = parse_j3l(resp.text, adapter)

        if verdict.draft is not None:
            draft = verdict.draft

        if verdict.kind == "none":
            note_invalid(verdict.failure or "other")
            record_action(None, verdict.failure or "other", "invalid")
            if len(events) >= action_cap:
                termination = "action_budget_exhausted"
                break
            continue
        if verdict.kind == "query":
            ask(verdict.query)
            continue
        record_action(verdict.action, "ok", "ok")
        if adapter.is_success():
            termination = "success"
            break
        if adapter.is_done() or len(events) >= action_cap:
            termination = "action_budget_exhausted"
            break
    else:
        termination = "action_budget_exhausted"

    return _finish(arm, adapter, meta, started, t0, events, queries, model_calls,
                   invalid_kinds, parse_failures, transport_failures, code_errors,
                   code_submissions, oracle_attempts, oracle_served, termination,
                   note, state_initial, draft)


def _j4_extra(prior_code: list[str], prior_errors: list[str]) -> str:
    if not prior_code and not prior_errors:
        return ""
    lines = ["YOUR SUBMISSIONS SO FAR (variables do not persist between submissions):"]
    for i, (code, err) in enumerate(zip(prior_code, prior_errors), start=1):
        lines.append(f"--- submission {i} ---")
        lines.append(code.strip())
        lines.append(f"--- error {i}: {err} ---")
    if len(prior_errors) > len(prior_code):
        for err in prior_errors[len(prior_code):]:
            lines.append(f"--- error: {err} ---")
    return "\n".join(lines)


def _run_program(adapter: TaskAdapter, code: str):
    """Execute J4's program in the frozen sandbox with the shared oracle."""
    from alienbody.agents.code_tool_agent import _run_sandbox

    env = {
        "STATE": adapter.sandbox_state(),
        "next_state": adapter.sandbox_next_state(),
    }
    try:
        return _run_sandbox(code, env, timeout_s=10.0), None
    except Exception as exc:  # noqa: BLE001 — the repair loop consumes it
        return None, f"{type(exc).__name__}: {exc}"


def _coerce_program_result(result: dict, adapter: TaskAdapter) -> list:
    from alienbody.interface_ladder_v2 import _coerce_plan_v2
    plan = _coerce_plan_v2(result, _PlanCapView(adapter), J2_PLAN_CAP)
    if plan is None:
        raise ValueError("program returned neither PLAN nor ACTION")
    return plan


class _PlanCapView:
    """``_coerce_plan_v2`` reads ``config.n_actions``; keep it task-agnostic."""
    def __init__(self, adapter: TaskAdapter):
        self.n_actions = adapter.n_actions


def _finish(arm, adapter, meta: EpisodeMeta, started, t0, events, queries, model_calls,
            invalid_kinds, parse_failures, transport_failures, code_errors,
            code_submissions, oracle_attempts, oracle_served, termination, note,
            state_initial, draft) -> EpisodeRecord:

    def _sum(field_name):
        vals = [getattr(c, field_name) for c in model_calls]
        if not vals or any(v is None for v in vals):
            return None
        return sum(int(v) for v in vals)

    success = termination == "success" and adapter.is_success()
    return EpisodeRecord(
        protocol_id=meta.protocol_id,
        dataset_sha256=meta.dataset_sha256,
        env_id=meta.env_id,
        model_id=meta.model_id,
        provider=meta.provider,
        inference_seed=meta.inference_seed,
        arm=arm,
        config_sha256=meta.config_sha256,
        code_sha256_at_launch=meta.code_sha256_at_launch,
        prompt_sha256=prompt_sha256(arm),
        started_at=started,
        success=success,
        termination_reason=termination,
        real_action_attempts=len(events),
        valid_real_actions=sum(1 for e in events if e.status == "ok"),
        oracle_attempts=oracle_attempts,
        oracle_served=oracle_served,
        distinct_queries=len(queries),
        model_calls=len(model_calls),
        input_tokens=_sum("input_tokens"),
        output_tokens=_sum("output_tokens"),
        reasoning_tokens=_sum("reasoning_tokens"),
        cached_tokens=_sum("cached_tokens"),
        wall_seconds=round(time.time() - t0, 3),
        parse_failures=parse_failures,
        transport_failures=transport_failures,
        code_errors=code_errors,
        events=[e.to_json() for e in events],
        queries=[q.to_json() for q in queries],
        raw_responses=[c.to_json() for c in model_calls],
        attempt_id=meta.attempt_id or f"{meta.env_id}:{arm}:{meta.inference_seed}",
        partition=meta.partition,
        block_id=meta.block_id,
        code_submissions=code_submissions,
        invalid_attempt_kinds=invalid_kinds,
        termination_note=note,
        arm_budgets={
            "action_cap": adapter.action_cap,
            "oracle_cap": adapter.oracle_cap,
            "response_max_tokens": ARM_RESPONSE_TOKENS[arm],
            "draft_max_chars": DRAFT_MAX_CHARS if arm in ("J1", "J3L") else None,
            "model_call_cap": j3l_model_call_cap(adapter.oracle_cap, adapter.action_cap)
            if arm == "J3L" else None,
        },
        state_sha256_initial=state_initial,
    )
