"""Interface Ladder v3 — R1 protocol fixes (2026-09-12).

New module.  The frozen v2 implementation (``interface_ladder_v2``) and the
frozen shared contract files (``state_v2``, ``prompts``, ``llm_agent``) are
**not** modified, so every v2.0/v2.1/v2.2 artifact keeps its exact provenance.
This module re-implements the *runner-side* episode logic for I0–I3 and re-uses
the frozen v2 primitives unchanged: the strict parsers, ``CounterfactualOracle``
and the I4/I5 episode functions are imported, never copied.

Defects fixed here (``notes/INDEPENDENT_REVIEW_2026-09-12.md`` §2, TODO R1):

a. **Step budget / STATE disagreement.**  The ladder ran 30 Phase-2 actions
   while handing the model a config with ``max_total_steps=50``, so STATE_V2
   reported ``actions_remaining=50``.  Root cause is the runner, not the
   renderer: ``state_v2.actions_remaining`` faithfully reports
   ``config.max_total_steps - state.step_count``, and the runner passed a suite
   config whose budget did not describe the episode it actually ran.  v3 scopes
   the environment config to the ladder episode (:func:`ladder_config`) so ONE
   number — 30 — governs the decision loop, the environment truncation and the
   STATE bytes; :func:`assert_state_budget` fails loudly if a raw suite config
   is ever rendered for a ladder arm.
b. **I3 oracle budget.**  The I3 arm now passes the protocol budget (1000,
   PROTOCOL v2.2) explicitly instead of inheriting ``CounterfactualOracle``'s
   120 default, and the budget is stated to the model.
c. **Accounting.**  Response-level parse failures are counted separately from
   environment budget units consumed.  A well-formed plan that only violates
   the 30-action cap is reported as one ``wellformed_over_cap`` response, never
   as 30 parse failures.
d. **Budget exhaustion and memory.**  After the query budget is exhausted the
   runner actually re-requests the model (bounded by
   ``I3_EXHAUSTION_HINT_RETRIES``) instead of parsing the previous ``CALL`` as
   an action; and the dialogue history persists across decisions, so query
   results stay visible at later steps.
e. **Truncation.**  Every failed response carries a token-cap cut verdict
   (:func:`truncation_suspected`), and the I1 response budget is large enough
   for the 2000-character scratchpad the prompt requires
   (``I1_RESPONSE_MAX_TOKENS``).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from alienbody.env.grid import EnvConfig, GridState
from alienbody.interface_ladder_v2 import (  # frozen primitives, reused as-is
    CounterfactualOracle,
    I4_ORACLE_CALLS,
    I4_PLAN_CAP,
    parse_action_strict,
    parse_plan_strict,
    parse_scratchpad_action_strict,
    run_i4_episode,
    run_i5_episode,
)
from alienbody.state_v2 import render_state_v2, state_v2_payload, state_v2_sha256

# ── Protocol budgets (PROTOCOL_INTERFACE_LADDER.md, v2.2 amendment) ────────
#
# These are the numbers the manifest must record and the arms must actually
# use; nothing here may silently fall back to a library default.
PHASE2_ACTION_BUDGET = 30
ORACLE_CALLS = I4_ORACLE_CALLS              # 1000, shared by I3/I4/I5
I0_RESPONSE_MAX_TOKENS = 128                # protocol table, I0 row
I1_SCRATCHPAD_MAX_CHARS = 2000
# I1's response budget is NOT in the protocol table (which fixes only the
# 2000-character table); it must simply be able to encode any *compliant*
# response.  Measured 2026-09-12 with cl100k_base on the worst compliant
# scratchpad: 2000 CJK characters, which JSON escaping (ensure_ascii) turns
# into ~12.0k characters / ~6.3k tokens (quotes or newlines: ~4.0k chars /
# ~2.0k tokens).  8192 therefore covers every compliant response; the v2 value
# of 256 could not even hold 400 characters of it.
I1_RESPONSE_MAX_TOKENS = 8192
I2_RESPONSE_MAX_TOKENS = 2048               # protocol table, I2 row
I3_RESPONSE_MAX_TOKENS = 256                # protocol table, I3 row (v2 used 128)
I3_EXHAUSTION_HINT_RETRIES = 3
PROTOCOL_VERSION = "PROTOCOL_INTERFACE_LADDER.md v2.3 (R1 fixes, 2026-09-12)"
STATE_CONTRACT = "STATE_V2 (renderer alienbody/state_v2.py, unmodified)"

BUDGET_EXHAUSTED_HINT = (
    "Simulator budget exhausted. You must now output ONLY one integer action id."
)

SYSTEM_I0 = """You receive a STATE_V2 JSON block for a grid-navigation task. The
action mapping and target are exact. Output ONLY one integer action id."""

SYSTEM_I1 = """You receive a STATE_V2 JSON block for a grid-navigation task.
Maintain a persistent scratchpad of at most 2000 characters. Output ONLY a JSON
object with exactly two keys: {"scratchpad":"your updated notes","action":N}.
N must be one legal action id. Do not use tools or explanations outside JSON."""

SYSTEM_I2 = """You receive a STATE_V2 JSON block for a grid-navigation task.
Return a complete, fixed navigation plan as ONLY a JSON list of 1 to 30 action
ids, for example [2,1,2]. You cannot revise the plan or query a simulator."""

SYSTEM_I3_TEMPLATE = """You receive a STATE_V2 JSON block for a grid-navigation task.
The action mapping and target are exact. You may output CALL next_state(r,c,a)
to query a counterfactual without spending an environment action, or output
ONLY one integer action id to act. The exact call syntax is
CALL next_state(0,0,0); this is a syntax example only, not a recommended
query. You may query the simulator at most {oracle_budget} times in total for
this episode. Use the simulator to test candidate actions before acting. Do not
output any explanation."""

QUERY = re.compile(r"^\s*CALL\s+next_state\((\d+),(\d+),(\d+)\)\s*$", re.I)

_JSON_CLOSERS = ("}", "]")

# A markdown fence at the very end of a non-empty response.  A fenced reply is
# *complete by construction*: the model closed what it opened, so the text is
# not a token-cap cut however much the parser dislikes the wrapper.
_FENCE_TAIL = re.compile(r"```[A-Za-z0-9_+-]*\s*$")

# Quoted span, escapes honored: blanked before bracket counting so that a
# bracket inside a scratchpad string cannot fake a cut.
_STRING_SPAN = re.compile(r'"(?:\\.|[^"\\])*"')

# Arm -> protocol-visible response budget.  The manifest and the arm MUST read
# the same numbers from here; defect (b) was exactly this block disagreeing
# (manifest 1000, I3 arm's oracle 120).
ARM_RESPONSE_TOKENS: dict[str, Optional[int]] = {
    "I0": I0_RESPONSE_MAX_TOKENS,
    "I1": I1_RESPONSE_MAX_TOKENS,
    "I2": I2_RESPONSE_MAX_TOKENS,
    "I3": I3_RESPONSE_MAX_TOKENS,
    "I4": I2_RESPONSE_MAX_TOKENS,   # run_i4_episode(response_max_tokens=...)
    "I5": None,
}


def system_prompt(arm: str, oracle_calls: int = ORACLE_CALLS) -> Optional[str]:
    """The exact protocol-visible system prompt for an arm (None for I4/I5)."""
    if arm == "I0":
        return SYSTEM_I0
    if arm == "I1":
        return SYSTEM_I1
    if arm == "I2":
        return SYSTEM_I2
    if arm == "I3":
        return SYSTEM_I3_TEMPLATE.format(oracle_budget=oracle_calls)
    return None


def arm_budget_block(arm: str, oracle_calls: int = ORACLE_CALLS) -> dict:
    """Budgets recorded in the manifest for one arm, with the prompt digest."""
    prompt = system_prompt(arm, oracle_calls)
    return {
        "phase2_environment_actions": PHASE2_ACTION_BUDGET,
        "next_state_oracle_calls": oracle_calls if arm in ("I3", "I4", "I5") else None,
        "response_max_tokens": ARM_RESPONSE_TOKENS[arm],
        "system_prompt_sha256": (hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                                 if prompt else None),
        "system_prompt_chars": len(prompt) if prompt else None,
    }


# ── Config scoping (defect a) ──────────────────────────────────────────────

def ladder_config(config: EnvConfig, action_budget: int = PHASE2_ACTION_BUDGET) -> EnvConfig:
    """Return the suite config scoped to the ladder episode.

    The frozen F4 suite carries ``max_total_steps=50`` (20 calibration + 30
    execution).  The ladder skips calibration and grants exactly 30 Phase-2
    actions, so the episode config must say 30 — otherwise STATE_V2 advertises
    50 remaining actions the runner will never grant.
    """
    if config.max_total_steps == action_budget:
        return config
    return replace(config, max_total_steps=action_budget)


def state_budget_violation(config: EnvConfig, state: GridState,
                           action_budget: int = PHASE2_ACTION_BUDGET) -> Optional[str]:
    """Return a human-readable reason when STATE_V2 misreports the budget."""
    reported = state_v2_payload(config, state)["actions_remaining"]
    expected = max(0, action_budget - state.step_count)
    if reported != expected:
        return (f"STATE_V2 reports actions_remaining={reported} but the ladder "
                f"grants {expected} (max_total_steps={config.max_total_steps}, "
                f"step_count={state.step_count}); pass a config through "
                f"ladder_config() before rendering")
    return None


def assert_state_budget(config: EnvConfig, state: GridState,
                        action_budget: int = PHASE2_ACTION_BUDGET) -> None:
    violation = state_budget_violation(config, state, action_budget)
    if violation:
        raise AssertionError(violation)


def episode_budget(config: EnvConfig, action_budget: Optional[int] = None) -> int:
    """The one number that governs the episode: the config's own budget.

    Callers may pass an explicit ``action_budget`` (short preflight runs), but
    it must equal what the config — and therefore STATE_V2 — reports; the
    per-decision :func:`assert_state_budget` enforces that.
    """
    return config.max_total_steps if action_budget is None else action_budget


# ── Response classification (defects c and e) ──────────────────────────────

def fenced_response(response: Optional[str]) -> bool:
    """True when a non-empty response ends by closing a markdown fence.

    Regression for the first live I2 run (2026-09-13, A800): all 27
    ``truncated_json`` verdicts were 96-character responses of the form
    ```` ```json\\n[2, 2, ...]\\n``` ```` — a *complete*, fenced constant plan
    that the strict parser rejects for the wrapper alone.  They are format
    failures (reported as ``fenced_json``), not token-cap cuts, and reading
    them as truncation would have mis-stated the arm's failure mode.
    """
    if not response or not response.strip():
        return False
    return _FENCE_TAIL.search(response.rstrip()) is not None


def _brackets_unbalanced(text: str) -> bool:
    """True when the reply opens more brackets than it closes, strings ignored.

    A ``max_tokens`` cut lands mid-structure, so the opened ``[``/``{`` outnumber
    the closed ones.  Counting is done on the text with quoted spans blanked out
    so that brackets inside a scratchpad string cannot fake the signal.
    """
    stripped = _STRING_SPAN.sub('""', text)
    return (stripped.count("[") > stripped.count("]")
            or stripped.count("{") > stripped.count("}"))


def truncation_suspected(response: Optional[str], kind: str) -> bool:
    """Token-cap cut heuristic, applied to responses that failed strict parsing.

    Returns True when the text is consistent with a ``max_tokens`` cut.  This is
    a text-level signal only — the frozen clients (``llm_agent.py``) do not
    expose ``finish_reason`` — so the bar is deliberately conservative: an
    *unbalanced* bracket count, i.e. the reply stops mid-structure.  Two classes
    of false positive are excluded by construction:

    * a reply that ends by closing a markdown fence was wrapped, not cut
      (:func:`fenced_response`);
    * a reply whose brackets balance was formatted wrong, not cut short.

    A truncated *integer* reply is indistinguishable from a malformed one, so
    ``kind="action"`` never claims truncation (the char count is still recorded).
    """
    if not response or not response.strip():
        return False
    if kind == "action":
        return False
    if fenced_response(response):
        return False
    return _brackets_unbalanced(response)


def _json_failure_reason(response: str, truncated: bool) -> str:
    """Name the shape of a non-parseable JSON reply.

    ``truncated`` is the token-cap verdict; a *fenced* reply is separated out
    because it is complete (the model closed the fence) and the wrapper alone
    is what the strict parser rejects.  The distinction is what keeps an arm's
    failure taxonomy honest: "cut off mid-array" and "wrapped in ```` ``` ````"
    are different findings about a model.
    """
    if truncated:
        return "truncated_json"
    if fenced_response(response):
        return "fenced_json"
    return "not_json"


@dataclass(frozen=True)
class ResponseVerdict:
    """Strict-parse verdict for one model response."""
    ok: bool
    reason: str = ""
    truncated: bool = False
    wellformed_over_cap: bool = False


def classify_response(response: Optional[str], kind: str,
                      n_actions: int, max_actions: int = I4_PLAN_CAP) -> ResponseVerdict:
    """Classify a response against the arm's output schema.

    ``kind`` is ``"action"``, ``"plan"`` or ``"scratchpad"``.  The verdict keeps
    apart (i) transport/format failures, (ii) token-cap cuts, and (iii)
    well-formed outputs that only violate a protocol cap — the distinction the
    R1 review asked for.
    """
    if not response or not response.strip():
        return ResponseVerdict(False, "empty_response")
    truncated = truncation_suspected(response, kind)

    if kind == "action":
        if parse_action_strict(response, n_actions) is not None:
            return ResponseVerdict(True)
        return ResponseVerdict(False, "not_a_single_integer", truncated)

    if kind == "plan":
        if parse_plan_strict(response, n_actions, max_actions) is not None:
            return ResponseVerdict(True)
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            return ResponseVerdict(False, _json_failure_reason(response, truncated),
                                   truncated)
        if not isinstance(value, list):
            return ResponseVerdict(False, "not_a_json_list")
        if not value:
            return ResponseVerdict(False, "empty_plan")
        if any(isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < n_actions
               for a in value):
            return ResponseVerdict(False, "plan_entry_not_a_legal_action")
        if len(value) > max_actions:
            return ResponseVerdict(False, f"plan_too_long:{len(value)}>{max_actions}",
                                   wellformed_over_cap=True)
        return ResponseVerdict(False, "plan_rejected")

    if kind == "scratchpad":
        if parse_scratchpad_action_strict(response, n_actions,
                                          I1_SCRATCHPAD_MAX_CHARS) is not None:
            return ResponseVerdict(True)
        try:
            value = json.loads(response)
        except json.JSONDecodeError:
            return ResponseVerdict(False, _json_failure_reason(response, truncated),
                                   truncated)
        if not isinstance(value, dict) or set(value) != {"scratchpad", "action"}:
            return ResponseVerdict(False, "not_a_scratchpad_object")
        if not isinstance(value["scratchpad"], str):
            return ResponseVerdict(False, "scratchpad_not_a_string")
        if len(value["scratchpad"]) > I1_SCRATCHPAD_MAX_CHARS:
            return ResponseVerdict(
                False, f"scratchpad_too_long:{len(value['scratchpad'])}>{I1_SCRATCHPAD_MAX_CHARS}",
                wellformed_over_cap=True)
        return ResponseVerdict(False, "scratchpad_action_out_of_range")

    raise ValueError(f"unknown response kind: {kind!r}")


# ── Accounting (defect c) ──────────────────────────────────────────────────

@dataclass
class Accounting:
    """Per-episode counters, keeping response failures apart from env budget."""
    model_calls: int = 0
    response_parse_failures: int = 0      # responses that had to be an action/plan
    truncated_responses: int = 0          # subset of the above, token-cap cut
    wellformed_over_cap: int = 0          # subset: valid JSON, protocol cap only
    empty_responses: int = 0
    invalid_actions_consumed: int = 0     # environment budget units, no action
    oracle_calls_served: int = 0
    oracle_calls_after_exhaustion: int = 0
    api_errors: int = 0                   # transport/context failures, episode-terminal
    failure_reasons: dict = field(default_factory=dict)

    def record_failure(self, verdict: ResponseVerdict) -> None:
        self.response_parse_failures += 1
        if verdict.truncated:
            self.truncated_responses += 1
        if verdict.wellformed_over_cap:
            self.wellformed_over_cap += 1
        if verdict.reason == "empty_response":
            self.empty_responses += 1
        self.failure_reasons[verdict.reason] = self.failure_reasons.get(verdict.reason, 0) + 1

    def as_dict(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "response_parse_failures": self.response_parse_failures,
            "truncated_responses": self.truncated_responses,
            "wellformed_over_cap": self.wellformed_over_cap,
            "empty_responses": self.empty_responses,
            "invalid_actions_consumed": self.invalid_actions_consumed,
            "oracle_calls_served": self.oracle_calls_served,
            "oracle_calls_after_exhaustion": self.oracle_calls_after_exhaustion,
            "api_errors": self.api_errors,
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
        }


def _tokens_used(client: Any, previous: int) -> int:
    """Delta of the client's token counter, 0 when the client has none."""
    total = getattr(client, "total_tokens", None)
    if total is None:
        return 0
    return max(0, int(total) - previous)


def _row(config: EnvConfig, arm: str, model: str, env, transcript: list,
         acc: Accounting, tokens_used: int, query_log: list,
         extra: Optional[dict] = None) -> dict:
    trajectory = env.get_trajectory()
    row = {
        "env_id": config.env_id,
        "family": config.family,
        "arm": arm,
        "model": model,
        "success": trajectory["success"],
        # v3 semantics: response-level failures only.  In v2 this field counted
        # consumed environment budget units (an I2 rejection showed up as 30).
        "parse_failures": acc.response_parse_failures,
        "invalid_actions_consumed": acc.invalid_actions_consumed,
        "tokens_used": tokens_used,
        "accounting": acc.as_dict(),
        "query_log": query_log,
        "trajectory": trajectory,
        "transcript": transcript,
    }
    if extra:
        row.update(extra)
    return row


def _consume_invalid(env, acc: Accounting, reason: str):
    _, _, terminated, truncated, _ = env.consume_invalid_action(reason)
    acc.invalid_actions_consumed += 1
    return terminated, truncated


# ── Episodes ───────────────────────────────────────────────────────────────

def run_i0_episode(config: EnvConfig, env, complete: Callable[..., str], model: str,
                   response_max_tokens: int = I0_RESPONSE_MAX_TOKENS,
                   action_budget: Optional[int] = None,
                   tokens_before: int = 0, client: Any = None) -> dict:
    """I0 step-wise: one real action per response, persistent dialogue."""
    budget = episode_budget(config, action_budget)
    messages = [{"role": "system", "content": SYSTEM_I0}]
    transcript: list = []
    acc = Accounting()
    for decision in range(budget):
        assert_state_budget(config, env.state, budget)
        messages.append({"role": "user", "content": render_state_v2(config, env.state)})
        response = complete(messages, response_max_tokens)
        acc.model_calls += 1
        messages.append({"role": "assistant", "content": response})
        action = parse_action_strict(response, config.n_actions)
        record = {
            "decision": decision + 1,
            "state_sha256": state_v2_sha256(config, env.state),
            "response": response,
            "response_chars": len(response or ""),
            "action": action,
        }
        if action is None:
            verdict = classify_response(response, "action", config.n_actions)
            record.update({"parse_failure_reason": verdict.reason,
                           "truncation_suspected": verdict.truncated})
            acc.record_failure(verdict)
        transcript.append(record)
        if action is None:
            terminated, truncated = _consume_invalid(env, acc, verdict.reason)
        else:
            _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return _row(config, "I0", model, env, transcript, acc,
                _tokens_used(client, tokens_before), [])


def run_i1_episode(config: EnvConfig, env, complete: Callable[..., str], model: str,
                   response_max_tokens: int = I1_RESPONSE_MAX_TOKENS,
                   action_budget: Optional[int] = None,
                   tokens_before: int = 0, client: Any = None) -> dict:
    """I1 scratchpad: the table is the memory channel; it survives a bad turn."""
    budget = episode_budget(config, action_budget)
    scratchpad = ""
    transcript: list = []
    acc = Accounting()
    for decision in range(budget):
        assert_state_budget(config, env.state, budget)
        state_text = render_state_v2(config, env.state)
        messages = [
            {"role": "system", "content": SYSTEM_I1},
            {"role": "user",
             "content": f"Previous scratchpad:\n{scratchpad}\n\n{state_text}"},
        ]
        response = complete(messages, response_max_tokens)
        acc.model_calls += 1
        parsed = parse_scratchpad_action_strict(response, config.n_actions,
                                                I1_SCRATCHPAD_MAX_CHARS)
        verdict: Optional[ResponseVerdict] = None
        if parsed is None:
            action = None
            verdict = classify_response(response, "scratchpad", config.n_actions)
            acc.record_failure(verdict)
        else:
            scratchpad, action = parsed
        record = {
            "decision": decision + 1,
            "state_sha256": state_v2_sha256(config, env.state),
            "response": response,
            "response_chars": len(response or ""),
            "scratchpad": scratchpad,
            "action": action,
        }
        if verdict is not None:
            record.update({"parse_failure_reason": verdict.reason,
                           "truncation_suspected": verdict.truncated})
        transcript.append(record)
        if action is None:
            terminated, truncated = _consume_invalid(env, acc, verdict.reason)
        else:
            _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return _row(config, "I1", model, env, transcript, acc,
                _tokens_used(client, tokens_before), [])


def run_i2_episode(config: EnvConfig, env, complete: Callable[..., str], model: str,
                   response_max_tokens: int = I2_RESPONSE_MAX_TOKENS,
                   action_budget: Optional[int] = None,
                   tokens_before: int = 0, client: Any = None) -> dict:
    """I2 plan-first: one plan, no revision; a rejected plan is ONE failure.

    The environment budget is still consumed (unplayed steps are not a free
    pass), but that consumption is reported as ``invalid_actions_consumed``,
    not as 30 parse failures.
    """
    budget = episode_budget(config, action_budget)
    assert_state_budget(config, env.state, budget)
    state_text = render_state_v2(config, env.state)
    messages = [{"role": "system", "content": SYSTEM_I2},
                {"role": "user", "content": state_text}]
    response = complete(messages, response_max_tokens)
    acc = Accounting(model_calls=1)
    plan = parse_plan_strict(response, config.n_actions, budget)
    verdict = None
    if plan is None:
        verdict = classify_response(response, "plan", config.n_actions, budget)
        acc.record_failure(verdict)
    record = {
        "decision": 0,
        "state_sha256": state_v2_sha256(config, env.state),
        "response": response,
        "response_chars": len(response or ""),
        "plan": plan,
        "plan_len": len(plan) if plan else 0,
    }
    if verdict is not None:
        record.update({"parse_failure_reason": verdict.reason,
                       "truncation_suspected": verdict.truncated,
                       "wellformed_over_cap": verdict.wellformed_over_cap})
    transcript = [record]

    terminated = truncated = False
    if plan is None:
        # Consume the environment budget the rejected plan would have spent.
        while not truncated:
            terminated, truncated = _consume_invalid(
                env, acc, f"invalid_plan:{verdict.reason}")
            if terminated:
                break
    else:
        for action in plan:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    return _row(config, "I2", model, env, transcript, acc,
                _tokens_used(client, tokens_before), [])


def run_i3_episode(config: EnvConfig, env, complete: Callable[..., str], model: str,
                   oracle_calls: int = ORACLE_CALLS,
                   response_max_tokens: int = I3_RESPONSE_MAX_TOKENS,
                   exhaustion_hint_retries: int = I3_EXHAUSTION_HINT_RETRIES,
                   action_budget: Optional[int] = None,
                   tokens_before: int = 0, client: Any = None,
                   max_calls_per_decision: Optional[int] = None) -> dict:
    """I3 dialogue oracle.

    Two R1 fixes live here:

    * **the last query is answered** — a ``CALL`` is never parsed as an action.
      When the budget is exhausted the model is told so and the model is
      re-requested (bounded by ``exhaustion_hint_retries``); only then can the
      turn end, and if it ends without an action that is a *response* failure.
    * **cross-step memory persists** — one dialogue per episode; every state
      block, query and simulator reply stays in the message list.
    """
    budget = episode_budget(config, action_budget)
    oracle = CounterfactualOracle(config, max_queries=oracle_calls)
    if max_calls_per_decision is None:
        # Bounded by the oracle budget itself: every servable CALL consumes one
        # unit, so the dialogue terminates when the budget plus the hint
        # retries are exhausted.  Deliberately NOT a small constant — clipping
        # a legitimate search is exactly the v2.1 defect this release fixes.
        max_calls_per_decision = oracle_calls + exhaustion_hint_retries + 1
    messages = [{"role": "system",
                 "content": SYSTEM_I3_TEMPLATE.format(oracle_budget=oracle_calls)}]
    transcript: list = []
    acc = Accounting()
    for decision in range(budget):
        assert_state_budget(config, env.state, budget)
        messages.append({"role": "user", "content": render_state_v2(config, env.state)})
        action = None
        verdict: Optional[ResponseVerdict] = None
        calls_this_decision = 0
        hints_this_decision = 0
        answered = False
        fatal = False
        last_response = ""
        while True:
            try:
                response = complete(messages, response_max_tokens)
            except Exception as exc:      # transport or context-window failure
                # A persistent API failure cannot be answered by re-prompting,
                # so the episode ends here instead of taking the process down
                # with it.  The turn is recorded exactly like any other
                # unanswered turn (one failure reason, one budget unit) and the
                # row is still written, so a context-bounded model shows up as
                # an episode failure rate rather than as a lost arm.
                acc.api_errors += 1
                verdict = ResponseVerdict(False, f"api_error:{type(exc).__name__}")
                fatal = True
                break
            acc.model_calls += 1
            calls_this_decision += 1
            last_response = response
            messages.append({"role": "assistant", "content": response})
            query = QUERY.fullmatch(response or "")
            if query is not None:
                reply = oracle.query(*map(int, query.groups()))
                transcript.append({
                    "decision": decision + 1,
                    "kind": "query",
                    "state_sha256": state_v2_sha256(config, env.state),
                    "response": response,
                    "response_chars": len(response or ""),
                    "query": (oracle.query_log[-1] if reply.status != "budget_exhausted"
                              else {"status": reply.status}),
                })
                if reply.status == "budget_exhausted":
                    acc.oracle_calls_after_exhaustion += 1
                    hints_this_decision += 1
                    messages.append({"role": "user", "content": BUDGET_EXHAUSTED_HINT})
                    if hints_this_decision > exhaustion_hint_retries:
                        verdict = ResponseVerdict(False, "budget_exhausted_unanswered")
                        break
                    continue
                acc.oracle_calls_served = len(oracle.query_log)
                messages.append({"role": "user",
                                 "content": f"Simulator: {reply.status}; "
                                            f"next position={reply.position}"})
                if calls_this_decision >= max_calls_per_decision:
                    verdict = ResponseVerdict(False, "max_calls_per_decision")
                    break
                continue
            action = parse_action_strict(response, config.n_actions)
            record = {
                "decision": decision + 1,
                "kind": "action",
                "state_sha256": state_v2_sha256(config, env.state),
                "response": response,
                "response_chars": len(response or ""),
                "action": action,
                "model_calls_this_decision": calls_this_decision,
            }
            if action is None:
                verdict = classify_response(response, "action", config.n_actions)
                record.update({"parse_failure_reason": verdict.reason,
                               "truncation_suspected": verdict.truncated})
            transcript.append(record)
            answered = True
            break
        if not answered:
            # The turn ended without the model ever answering with an action
            # (query budget exhausted and the hint ignored, or the per-decision
            # call ceiling hit).  Record the terminal failure so every decision
            # has exactly one action-kind record; the CALLs stay in the query
            # records above and are never parsed as actions.
            if verdict is None:                      # defensive; unreachable today
                verdict = ResponseVerdict(False, "no_action_produced")
            transcript.append({
                "decision": decision + 1,
                "kind": "action",
                "state_sha256": state_v2_sha256(config, env.state),
                "response": last_response,
                "response_chars": len(last_response or ""),
                "action": None,
                "model_calls_this_decision": calls_this_decision,
                "parse_failure_reason": verdict.reason,
                "truncation_suspected": verdict.truncated,
            })
        if action is None:
            if verdict is None:                      # defensive; unreachable today
                verdict = ResponseVerdict(False, "no_action_produced")
            acc.record_failure(verdict)
            terminated, truncated = _consume_invalid(env, acc, verdict.reason)
        else:
            _, _, terminated, truncated, _ = env.step(action)
        if fatal:
            break
        if terminated or truncated:
            break
    return _row(config, "I3", model, env, transcript, acc,
                _tokens_used(client, tokens_before), oracle.query_log)


def run_arm(arm: str, config: EnvConfig, env, complete: Callable[..., str],
            model: str, client: Any = None, tokens_before: int = 0,
            oracle_calls: int = ORACLE_CALLS) -> dict:
    """Dispatch one ladder episode to its arm.  Single source of truth.

    This lives in the module rather than in the CLI script so that the test
    suite exercises the *dispatch* and not only the episode functions.  The I4
    arm shipped with a ``NameError`` on the runner's local spelling of the I2
    response budget precisely because the dispatch path had no test: the budget
    now comes from :data:`ARM_RESPONSE_TOKENS`, the same table the manifest
    records.
    """
    if arm == "I0":
        return run_i0_episode(config, env, complete, model=model,
                              tokens_before=tokens_before, client=client)
    if arm == "I1":
        return run_i1_episode(config, env, complete, model=model,
                              tokens_before=tokens_before, client=client)
    if arm == "I2":
        return run_i2_episode(config, env, complete, model=model,
                              tokens_before=tokens_before, client=client)
    if arm == "I3":
        return run_i3_episode(config, env, complete, model=model,
                              oracle_calls=oracle_calls,
                              tokens_before=tokens_before, client=client)
    if arm == "I4":
        row = run_i4_episode(config, env, complete, model=model,
                             oracle_calls=oracle_calls,
                             response_max_tokens=ARM_RESPONSE_TOKENS["I4"])
        row["tokens_used"] = _tokens_used(client, tokens_before)
        return row
    if arm == "I5":
        return run_i5_episode(config, env, oracle_calls=oracle_calls)
    raise ValueError(f"unknown arm: {arm!r}")


def summarize_rows(rows: list) -> dict:
    """Aggregate counters for the manifest (response failures vs budget units).

    Rows produced by the imported frozen *v2* episode functions (I4, I5) carry
    the v2 row shape: no ``accounting`` block and no ``invalid_actions_consumed``
    key.  They are counted under ``legacy_v2_rows`` and contribute only the
    fields they actually carry, so no v3 counter is silently invented for them
    (the first live I5 run crashed here with ``KeyError`` before this guard).
    """
    def acct(row: dict, key: str) -> int:
        return int(row.get("accounting", {}).get(key, 0))

    return {
        "n_envs": len(rows),
        "successes": sum(1 for r in rows if r["success"]),
        "success_rate": (sum(1 for r in rows if r["success"]) / len(rows)) if rows else 0.0,
        "parse_failures": sum(r["parse_failures"] for r in rows),
        "invalid_actions_consumed": sum(
            r.get("invalid_actions_consumed", 0) for r in rows),
        "truncated_responses": sum(acct(r, "truncated_responses") for r in rows),
        "wellformed_over_cap": sum(acct(r, "wellformed_over_cap") for r in rows),
        "model_calls": sum(acct(r, "model_calls") for r in rows),
        "oracle_calls_served": sum(acct(r, "oracle_calls_served") for r in rows),
        "oracle_calls_after_exhaustion": sum(
            acct(r, "oracle_calls_after_exhaustion") for r in rows),
        "api_errors": sum(acct(r, "api_errors") for r in rows),
        "tokens_used": sum(r.get("tokens_used", 0) for r in rows),
        "legacy_v2_rows": sum(1 for r in rows if "accounting" not in r),
        "failure_reasons": _merge_reasons(rows),
    }


def _merge_reasons(rows: list) -> dict:
    merged: dict = {}
    for row in rows:
        for reason, count in row.get("accounting", {}).get("failure_reasons", {}).items():
            merged[reason] = merged.get(reason, 0) + count
    return dict(sorted(merged.items()))
