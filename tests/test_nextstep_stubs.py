"""Offline regressions for the NEXTSTEP J-protocol (§8.3, no API, no GPU).

Seven scenarios the plan requires *before* any model is called, plus the two
acceptance checks (identical common STATE across arms; summary recomputable
from the event log).  Every test drives the real runners with a scripted
:class:`StubCaller`; nothing here touches a network.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alienbody.env.grid import EnvConfig
from alienbody.nextstep import interfaces as I
from alienbody.nextstep.adapters import (
    F4Adapter, GatewayCaller, NullCaller, StubCaller, WorkflowAdapter,
)
from alienbody.nextstep.records import (
    EpisodeRecord, EpisodeWriter, resummarize_from_events, summarize_records,
)

_ROW = (0, 0, 0, 0)

# The J-protocol launcher (scripts/run_nextstep.py) and the frozen launch
# configs it reads are not part of this release — they encode the run
# bookkeeping of the paper's own cluster runs.  The two runner-configuration
# tests below stay here as executable documentation of the policy they assert;
# they skip when the launcher is absent.
_RUN_NEXTSTEP = (Path(__file__).resolve().parents[1] / "scripts"
                 / "run_nextstep.py")


def tiny_config(action_cap: int = 4, grid_size: int = 4,
                target: tuple = (2, 2)) -> EnvConfig:
    """4x4 Type-B grid, start (0,0) -> target, scoped to the episode budget."""
    return EnvConfig(
        env_id="unit_nextstep_000", family=2, seed=1, grid_size=grid_size,
        cell_colors=tuple(_ROW for _ in range(grid_size)),
        agent_start=(0, 0), agent_start_color=0, agent_start_dir=0,
        target_pos=target, action_type="B",
        action_mapping=("down", "up", "right", "left"), n_actions=4,
        phase1_budget=20, max_total_steps=action_cap, obstacles=(),
    )


def adapter(action_cap: int = 4, oracle_cap: int = 1000, **kw) -> F4Adapter:
    return F4Adapter(config=tiny_config(action_cap=action_cap),
                     action_cap=action_cap, oracle_cap=oracle_cap, **kw)


def meta(env_id: str = "unit_nextstep_000", config_sha: str = "cfg") -> I.EpisodeMeta:
    return I.EpisodeMeta(
        protocol_id="test", dataset_sha256="ds", env_id=env_id, model_id="stub",
        provider="stub", inference_seed=0, config_sha256=config_sha,
        code_sha256_at_launch="code",
    )


def solve_plan(action_cap: int = 4) -> list:
    """The reference plan for the tiny config (BFS through the shared oracle)."""
    ad = adapter(action_cap=action_cap)
    ad.initial_state()
    return ad.bfs_plan(cap=action_cap)


# ── R1: budgets ────────────────────────────────────────────────────────────

class TestBudgets(unittest.TestCase):
    def test_action_cap_matches_real_attempts_and_stops_at_success(self):
        plan = solve_plan(4)
        self.assertTrue(plan and len(plan) <= 4, plan)
        caller = StubCaller(script=[str(a) for a in plan])
        rec = I.run_episode("J0", adapter(4), caller, meta())
        self.assertEqual(rec.termination_reason, "success")
        self.assertTrue(rec.success)
        # no action after success: one model call per executed action
        self.assertEqual(rec.real_action_attempts, len(plan))
        self.assertEqual(rec.valid_real_actions, len(plan))
        self.assertEqual(rec.model_calls, len(plan))

    def test_invalid_attempts_spend_real_steps(self):
        rec = I.run_episode("J0", adapter(3), StubCaller(script=["not a number"] * 5), meta())
        self.assertEqual(rec.termination_reason, "action_budget_exhausted")
        self.assertEqual(rec.real_action_attempts, 3)
        self.assertEqual(rec.valid_real_actions, 0)
        self.assertEqual(rec.parse_failures, 3)
        self.assertEqual(rec.model_calls, 3)

    def test_actions_remaining_is_the_episode_budget(self):
        plan = solve_plan(4)
        caller = StubCaller(script=[str(a) for a in plan])
        I.run_episode("J0", adapter(4), caller, meta())
        first_prompt = caller.prompts[0][1]
        self.assertIn("ACTIONS REMAINING: 4", first_prompt)


# ── R2: oracle exhaustion ──────────────────────────────────────────────────

class TestOracleExhaustion(unittest.TestCase):
    def test_last_query_served_then_refused_and_acting_still_works(self):
        ad = adapter(4, oracle_cap=2)
        q = lambda r, c, a: json.dumps({"kind": "query", "state": [r, c], "action": a})
        script = [q(0, 0, 0), q(0, 1, 0), q(0, 2, 0),
                  json.dumps({"kind": "action", "action": 0})]
        rec = I.run_episode("J3L", ad, StubCaller(script=script), meta())
        self.assertEqual(rec.oracle_attempts, 3)
        self.assertEqual(rec.oracle_served, 2)          # the third is refused
        self.assertEqual(rec.queries[-1]["status"], "budget_exhausted")
        self.assertEqual(rec.valid_real_actions, 1)     # acting continues
        self.assertEqual(rec.queries[0]["status"], "ok")


# ── R3: ledger bookkeeping ─────────────────────────────────────────────────

class TestLedger(unittest.TestCase):
    def test_repeat_costs_budget_updates_count_and_keeps_the_fact(self):
        ad = adapter(4, oracle_cap=10)
        q = json.dumps({"kind": "query", "state": [0, 0], "action": 1})
        script = [q, q, json.dumps({"kind": "action", "action": 0})]
        rec = I.run_episode("J3L", ad, StubCaller(script=script), meta())
        self.assertEqual(len(rec.queries), 1)           # one distinct query
        self.assertEqual(rec.queries[0]["count"], 2)
        self.assertEqual(rec.distinct_queries, 1)
        self.assertEqual(rec.oracle_served, 2)          # repeats are not free
        self.assertEqual(rec.oracle_attempts, 2)

    def test_ledger_renders_first_seen_order_with_counts(self):
        ad = adapter(4, oracle_cap=10)
        ad.initial_state()
        qs = [
            I.Query(index=1, key=[0, 0, 1], answer=[1, 0], status="ok", count=2, served=True),
            I.Query(index=2, key=[0, 0, 2], answer=[0, 1], status="ok", count=1, served=True),
        ]
        text = I.render_query_ledger(ad, qs)
        self.assertLess(text.index("[1]"), text.index("[2]"))
        self.assertIn("x2", text)


# ── R4: format rejections are counted, never repaired silently ─────────────

class TestFormatRejections(unittest.TestCase):
    def test_float_and_fenced_action_are_distinct_labels(self):
        rec = I.run_episode("J0", adapter(3), StubCaller(script=["3.0", "```json\n2\n```"]), meta())
        self.assertEqual(rec.invalid_attempt_kinds.get("not_json"), 1)
        self.assertEqual(rec.invalid_attempt_kinds.get("fenced_json"), 1)
        self.assertEqual(rec.events[0]["status"], "invalid")
        self.assertEqual(rec.events[0]["state_before"], rec.events[0]["state_after"])

    def test_draft_over_cap_is_rejected_not_trimmed(self):
        big = "x" * (I.DRAFT_MAX_CHARS + 1)
        script = [json.dumps({"draft": big, "action": 0})]
        rec = I.run_episode("J1", adapter(2), StubCaller(script=script), meta())
        self.assertEqual(rec.invalid_attempt_kinds.get("draft_over_cap"), 1)
        self.assertEqual(rec.real_action_attempts, 1)

    def test_plan_over_cap_ends_the_episode_without_executing(self):
        script = [json.dumps([0] * (I.J2_PLAN_CAP + 1))]
        rec = I.run_episode("J2", adapter(30), StubCaller(script=script), meta())
        self.assertEqual(rec.termination_reason, "plan_over_cap")
        self.assertEqual(rec.real_action_attempts, 0)
        self.assertFalse(rec.success)

    def test_plan_is_executed_in_order(self):
        plan = solve_plan(4)
        rec = I.run_episode("J2", adapter(4), StubCaller(script=[json.dumps(plan)]), meta())
        self.assertEqual(rec.termination_reason, "success")
        self.assertEqual([e["action"] for e in rec.events], plan)


# ── R5: sandbox permissions ────────────────────────────────────────────────

class TestSandboxPermissions(unittest.TestCase):
    def test_safe_program_reads_public_state_and_returns_a_plan(self):
        code = ("```python\n"
                "n = len(STATE['cell_colors_row_major'])\n"
                "PLAN = [0]\n"
                "```")
        rec = I.run_episode("J4", adapter(4), StubCaller(script=[code]), meta())
        self.assertEqual(rec.real_action_attempts, 1)
        self.assertEqual(rec.code_errors, 0)
        self.assertEqual(rec.code_submissions, 1)

    def test_file_read_is_rejected_and_counted_as_a_code_error(self):
        code = "```python\nPLAN = open('secrets.txt').read()\n```"
        script = [code, "```python\nPLAN = [0]\n```"]
        rec = I.run_episode("J4", adapter(4), StubCaller(script=script), meta())
        self.assertGreaterEqual(rec.code_errors, 1)
        self.assertEqual(rec.code_submissions, 2)

    def test_import_and_private_names_are_rejected(self):
        for bad in ("```python\nimport os\nPLAN=[0]\n```",
                    "```python\nPLAN=[env.config.target_pos[0]]\n```",
                    "```python\nPLAN=[next_state_budget]\n```"):
            rec = I.run_episode("J4", adapter(4), StubCaller(script=[bad]), meta())
            self.assertGreaterEqual(rec.code_errors, 1, bad)

    def test_sandbox_payload_has_no_private_fields(self):
        ad = adapter(4)
        ad.initial_state()
        public = ad.sandbox_state()
        self.assertEqual(set(public), {
            "contract", "grid_size", "cell_colors_row_major", "obstacles", "agent",
            "target", "phase", "n_actions", "action_mapping", "actions_remaining",
        })


# ── R6: failures end the episode instead of looping ────────────────────────

class TestTermination(unittest.TestCase):
    def test_transport_failure_ends_the_episode(self):
        rec = I.run_episode("J0", adapter(4),
                            StubCaller(script=[{"transport_error": "HTTP 502"}]), meta())
        self.assertEqual(rec.termination_reason, "transport_failure")
        self.assertEqual(rec.transport_failures, 1)
        self.assertEqual(rec.real_action_attempts, 0)

    def test_context_overflow_is_its_own_reason(self):
        rec = I.run_episode("J0", adapter(4),
                            StubCaller(script=[{"context_error":
                                                "maximum context length exceeded"}]), meta())
        self.assertEqual(rec.termination_reason, "context_failure")

    def test_model_call_cap_stops_a_query_loop(self):
        ad = adapter(action_cap=2, oracle_cap=2)
        cap = I.j3l_model_call_cap(2, 2)
        script = [json.dumps({"kind": "query", "state": [0, 0], "action": 1})] * 40
        rec = I.run_episode("J3L", ad, StubCaller(script=script), meta())
        self.assertEqual(rec.termination_reason, "model_call_cap")
        self.assertEqual(rec.model_calls, cap)

    def test_gateway_caller_retries_one_identical_payload_then_gives_up(self):
        class FakeClient:
            _api = {"end_point": "http://invalid"}
            def _build_headers(self):
                return {}
            def _build_body(self, messages, max_tokens):
                return {"messages": messages, "max_tokens": max_tokens}
            def _parse_response(self, data):
                return data["choices"][0]["message"]["content"], 0

        calls = []

        def boom(url, **kw):
            calls.append(kw["json"])
            raise RuntimeError("connection reset")

        caller = GatewayCaller(FakeClient(), "fake-model", post=boom)
        resp = caller.call("sys", "user", 32)
        self.assertTrue(resp.failed)
        self.assertEqual(resp.attempts, 2)
        self.assertEqual(resp.failure_kind, "transport")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])          # identical payload

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "7"},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                  "total_tokens": 7}}

        n = {"i": 0}

        def flaky(url, **kw):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("transient")
            return FakeResponse()

        resp2 = GatewayCaller(FakeClient(), "fake-model", post=flaky).call("s", "u", 8)
        self.assertEqual(resp2.text, "7")
        self.assertEqual(resp2.attempts, 2)
        self.assertEqual(resp2.input_tokens, 5)
        self.assertEqual(resp2.output_tokens, 2)
        self.assertEqual(resp2.finish_reason, "stop")

    def test_retry_waits_before_resending_the_identical_payload(self):
        """The delay is the whole point: an immediate retry lands in the same
        502/429 window (measured on the gateway 2026-09-15)."""
        class FakeClient:
            _api = {"end_point": "http://invalid"}
            def _build_headers(self):
                return {}
            def _build_body(self, messages, max_tokens):
                return {"messages": messages, "max_tokens": max_tokens}
            def _parse_response(self, data):
                return data["choices"][0]["message"]["content"], 0

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "ok"},
                                     "finish_reason": "stop"}], "usage": {}}

        payloads, slept, times = [], [], []

        def flaky(url, **kw):
            times.append(len(slept))       # which attempt this is, by sleeps so far
            payloads.append(kw["json"])
            if len(payloads) == 1:
                raise RuntimeError("HTTP 502: bad gateway")
            return FakeResponse()

        resp = GatewayCaller(FakeClient(), "fake-model", post=flaky,
                          retry_backoff_s=3.0, sleep=slept.append).call("s", "u", 8)
        self.assertEqual(resp.text, "ok")
        self.assertEqual(resp.attempts, 2)
        self.assertEqual(slept, [3.0], "exactly one wait, of the configured length")
        self.assertEqual(times, [0, 1], "the wait happens between the two attempts")
        self.assertEqual(payloads[0], payloads[1], "still an identical payload")

    def test_no_wait_when_the_backoff_is_zero(self):
        class FakeClient:
            _api = {"end_point": "http://invalid"}
            def _build_headers(self):
                return {}
            def _build_body(self, messages, max_tokens):
                return {"messages": messages, "max_tokens": max_tokens}
            def _parse_response(self, data):
                return data["choices"][0]["message"]["content"], 0

        slept = []
        resp = GatewayCaller(FakeClient(), "fake-model",
                          post=lambda url, **kw: (_ for _ in ()).throw(RuntimeError("x")),
                          retry_backoff_s=0.0, sleep=slept.append).call("s", "u", 8)
        self.assertTrue(resp.failed)
        self.assertEqual(resp.attempts, 2, "zero backoff must not drop the retry")
        self.assertEqual(slept, [])

    @unittest.skipUnless(_RUN_NEXTSTEP.exists(),
                         "J-protocol launcher is not part of this release")
    def test_runner_delays_the_retry_even_when_the_config_is_silent(self):
        """A missing key must not silently disable the delay on a real run."""
        from scripts.run_nextstep import resolve_transport_backoff
        self.assertEqual(resolve_transport_backoff({}), 3.0)
        self.assertEqual(resolve_transport_backoff({"decoding": {}}), 3.0)
        self.assertEqual(resolve_transport_backoff({"decoding": {"transport_backoff_s": 5}}), 5.0)
        self.assertEqual(resolve_transport_backoff({"decoding": {"transport_backoff_s": 5}}, 0.0), 0.0)
        # an unset/null value is "no policy recorded" — take the safe default,
        # never 0. Only an explicit numeric 0 (or an override) disables it.
        self.assertEqual(resolve_transport_backoff({"decoding": {"transport_backoff_s": None}}), 3.0)
        # a prose value (the dev config's house style for policy keys) must not
        # crash the launch nor silently disable the delay
        self.assertEqual(resolve_transport_backoff({"decoding": {"transport_backoff_s": "3 s, see D26"}}), 3.0)
        self.assertEqual(resolve_transport_backoff({"decoding": {"transport_backoff_s": "3 s"}}, 0.0), 0.0)

    @unittest.skipUnless(_RUN_NEXTSTEP.exists(),
                         "J-protocol launcher is not part of this release")
    def test_runner_never_defaults_the_temperature_to_zero(self):
        """D27: the degenerate setting must be unreachable by omission.

        At temperature 0 F4 J3L loops on one query (500-1051 repeats, ~1064
        calls, 1-3 real actions), so a forgotten --temperature or a prose
        config value may never resolve to 0.
        """
        from scripts.run_nextstep import resolve_temperature
        from pathlib import Path as _Path
        import json as _json
        self.assertEqual(resolve_temperature({}), 1.0)
        self.assertEqual(resolve_temperature({"decoding": {}}), 1.0)
        self.assertEqual(resolve_temperature({"decoding": {"temperature": None}}), 1.0)
        self.assertEqual(resolve_temperature({"decoding": {"temperature": "fixed later"}}), 1.0)
        self.assertEqual(resolve_temperature({"decoding": {"temperature": 0.7}}), 0.7)
        self.assertEqual(resolve_temperature({"decoding": {"temperature": 1.0}}, 0.0), 0.0)
        config = _json.loads((_Path(__file__).resolve().parents[2]
                              / "protocols" / "nextstep" / "config.dev.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(config["decoding"]["temperature"], 1.0)
        self.assertIn("D27", config["decoding"]["temperature_note"])


# ── R7: the writer's file discipline ───────────────────────────────────────

class TestWriter(unittest.TestCase):
    def _record(self, env_id="e0", config_sha="cfg") -> EpisodeRecord:
        return EpisodeRecord(
            protocol_id="test", dataset_sha256="ds", env_id=env_id, model_id="m",
            provider="stub", inference_seed=0, arm="J0", config_sha256=config_sha,
            code_sha256_at_launch="c", prompt_sha256=None, started_at="t",
            success=False, termination_reason="action_budget_exhausted",
            real_action_attempts=0, valid_real_actions=0, oracle_attempts=0,
            oracle_served=0, distinct_queries=0, model_calls=0, input_tokens=None,
            output_tokens=None, reasoning_tokens=None, cached_tokens=None,
            wall_seconds=0.0, parse_failures=0, transport_failures=0, code_errors=0,
        )

    def test_refuses_to_overwrite_and_resumes_into_a_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = EpisodeWriter(tmp, "J0", "m", "dev", "b1")
            w.write(self._record())
            w.close()
            with self.assertRaises(FileExistsError):
                EpisodeWriter(tmp, "J0", "m", "dev", "b1")
            w2 = EpisodeWriter(tmp, "J0", "m", "dev", "b1", resume=True)
            self.assertTrue(w2.path.name.endswith(".resume1.jsonl"))
            self.assertIsNotNone(w2.already_done("e0", "J0", 0, "cfg"))
            self.assertIsNone(w2.already_done("e0", "J0", 0, "different-config"))
            w2.close()

    def test_torn_tail_is_not_a_completed_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = EpisodeWriter(tmp, "J0", "m", "dev", "b1")
            w.write(self._record("e0"))
            w._fh.write('{"env_id": "e1", "arm": "J0"')   # torn line
            w._fh.write("\n")
            w.write(self._record("e2"))                   # after the tear
            w.close()
            w2 = EpisodeWriter(tmp, "J0", "m", "dev", "b1", resume=True)
            self.assertIsNotNone(w2.already_done("e0", "J0", 0, "cfg"))
            self.assertIsNone(w2.already_done("e1", "J0", 0, "cfg"))   # torn
            self.assertIsNone(w2.already_done("e2", "J0", 0, "cfg"))   # after the tear
            w2.close()


# ── acceptance: identical common STATE, summary from events ────────────────

class TestSharedStateAndSummary(unittest.TestCase):
    def test_first_round_prompt_is_identical_across_stepwise_arms(self):
        """The common block (STATE + EVENT TABLE + budget) is one shared renderer.

        Only J3L appends its ledger, and only after the shared block — that
        tail is the arm's own affordance, not a different state rendering.
        """
        texts = []
        for arm in ("J0", "J1", "J3L"):
            caller = StubCaller(script=["0"])
            I.run_episode(arm, adapter(2), caller, meta())
            texts.append(caller.prompts[0][1])
        shared = [t[:t.index("ACTIONS REMAINING: 2") + len("ACTIONS REMAINING: 2")]
                  for t in texts]
        self.assertEqual(shared[0], shared[1])
        self.assertEqual(shared[0], shared[2])
        self.assertIn("QUERY LEDGER: (empty)", texts[2][len(shared[2]):])

    def test_public_state_bytes_are_stable(self):
        a1, a2 = adapter(4), adapter(4)
        a1.initial_state()
        a2.initial_state()
        self.assertEqual(a1.render_public_state(), a2.render_public_state())
        self.assertEqual(I.sha256_text(a1.render_public_state()),
                         I.sha256_text(a2.render_public_state()))

    def test_summary_is_recomputable_from_the_event_log(self):
        records = []
        for script in ([str(a) for a in solve_plan(4)],
                       ["bogus", "bogus"],
                       ["```fenced```"]):
            rec = I.run_episode("J0", adapter(3), StubCaller(script=script), meta())
            records.append(rec.to_json())
        summary = summarize_records(records)
        recount = resummarize_from_events(records)
        self.assertEqual(summary["n_episodes"], 3)
        self.assertEqual(summary["real_action_attempts"],
                         sum(e["real_action_attempts"] for e in recount["episodes"]))
        for e in recount["episodes"]:
            self.assertTrue(e["attempts_match"])
            self.assertTrue(e["valid_real_actions_match"])


# ── workflow (S2, W-K): the same contract on the second environment ────────

# k=3 flags, 2 tools: raise f0 unconditionally, then raise f1 once f0 is on.
_WF_TINY = {
    "instance_id": "unit_workflow_000",
    "k": 3, "m": 2, "start": 0b000,
    "goal_on": 0b011, "goal_off": 0b000,
    "actions": [[0, 0, 0b001, 0], [0b001, 0, 0b010, 0]],
    "schema_fingerprint": "unit", "difficulty": "easy", "shortest_plan_len": 2,
}

_WF_DEV = (Path(__file__).resolve().parents[1] / "data" / "frozen_suites"
           / "nextstep_workflow_dev" / "dev_instances.json")
_F4_DEV_DIR = (Path(__file__).resolve().parents[1] / "data" / "frozen_suites"
               / "nextstep_f4_dev")
_F4_PUBLISHED = (Path(__file__).resolve().parents[1] / "data" / "frozen_suites"
                 / "replication_primary" / "family4" / "replication_primary")


def wf_adapter(action_cap: int = 4, oracle_cap: int = 64) -> WorkflowAdapter:
    return WorkflowAdapter.from_payload(_WF_TINY, action_cap=action_cap,
                                        oracle_cap=oracle_cap)


class TestWorkflowAdapter(unittest.TestCase):
    def test_the_unit_instance_is_a_legal_instance(self):
        from alienbody.workflow.env import WorkflowInstance, instance_problems
        self.assertEqual(instance_problems(WorkflowInstance.from_dict(_WF_TINY)), ())

    def test_query_accepts_only_the_call_syntax(self):
        """Grammar is exact: no space after '(' and none before ')' (as in F4)."""
        ad = wf_adapter()
        ad.initial_state()
        self.assertEqual(ad.parse_query("CALL next_state(3, 0)"), (3, 0))
        self.assertEqual(ad.parse_query("  call NEXT_STATE(5 , 2) "), (5, 2))
        self.assertIsNone(ad.parse_query("next_state(3,0)"))
        self.assertIsNone(ad.parse_query("CALL next_state(3.0, 0)"))
        self.assertIsNone(ad.parse_query("CALL next_state( 3, 0)"))

    def test_j3l_bitvec_query_flows_through_the_ledger(self):
        """The workflow query is an integer state, not a grid cell."""
        ad = wf_adapter(oracle_cap=8)
        script = [json.dumps({"kind": "query", "state": 0, "action": 0}),
                  json.dumps({"kind": "query", "state": 1, "action": 1}),
                  json.dumps({"kind": "action", "action": 0})]
        rec = I.run_episode("J3L", ad, StubCaller(script=script), meta(),
                            task="workflow")
        self.assertEqual([q["key"] for q in rec.queries], [[0, 0], [1, 1]])
        self.assertEqual(rec.queries[0]["answer"], [1])     # tool 0 sets f0
        self.assertEqual(rec.queries[1]["answer"], [3])     # tool 1 sets f1 after f0
        self.assertEqual(rec.oracle_served, 2)
        self.assertEqual(rec.valid_real_actions, 1)

    def test_j3l_grid_shaped_state_is_a_query_failure_not_an_action(self):
        ad = wf_adapter(oracle_cap=8)
        bad = json.dumps({"kind": "query", "state": [0, 0], "action": 0})
        script = [bad, json.dumps({"kind": "action", "action": 0})]
        rec = I.run_episode("J3L", ad, StubCaller(script=script), meta(),
                            task="workflow")
        self.assertEqual(rec.invalid_attempt_kinds.get("query_out_of_range"), 1)
        self.assertEqual(rec.oracle_attempts, 0)
        self.assertEqual(rec.real_action_attempts, 2)       # the refusal spends a step

    def test_oracle_accounting_follows_the_f4_rules(self):
        ad = wf_adapter(oracle_cap=3)
        ad.initial_state()
        ok = ad.oracle_transition((0, 0))
        self.assertEqual((ok.status, ok.served, ok.answer), ("ok", True, (1,)))
        bad_state = ad.oracle_transition((8, 0))       # outside 2**3 states
        self.assertEqual((bad_state.status, bad_state.served), ("invalid_state", True))
        bad_action = ad.oracle_transition((0, 9))
        self.assertEqual((bad_action.status, bad_action.served), ("invalid_action", True))
        refused = ad.oracle_transition((0, 0))
        self.assertEqual((refused.status, refused.served), ("budget_exhausted", False))
        self.assertEqual(ad.oracle_calls_served, 3)    # only refusals are free

    def test_masks_are_public_but_grading_labels_are_not(self):
        ad = wf_adapter()
        ad.initial_state()
        payload = ad.payload()
        self.assertEqual(set(payload), {
            "contract", "k", "m", "state_bits", "state", "goal_on", "goal_off",
            "goal_literals", "actions_remaining", "tools",
        })
        for tool in payload["tools"]:
            self.assertEqual(set(tool), {"id", "pre_on", "pre_off", "set_on",
                                         "set_off", "schema"})
            self.assertIn("if ", tool["schema"])       # the exact schema, W-K
        self.assertEqual(payload["tools"][1]["schema"], "if f0=1 -> f1:=1")
        rendered = ad.render_public_state()
        for hidden in ("unit_workflow_000", "difficulty", "shortest_plan_len",
                       "schema_fingerprint"):
            self.assertNotIn(hidden, rendered)

    def test_bitvec_query_round_trips_through_rendering(self):
        ad = wf_adapter()
        ad.initial_state()
        self.assertEqual(ad.render_query((3, 1)), "next_state(3,1)")
        text = ad.render_query_answer((3, 1), (7,), "ok")
        self.assertIn("011", text)                     # k=3 bit string
        self.assertIn("111", text)
        self.assertIn("budget_exhausted", ad.render_query_answer((3, 1), None,
                                                                "budget_exhausted"))

    def test_j5_episode_runs_without_a_model(self):
        rec = I.run_episode("J5", wf_adapter(), NullCaller(), meta())
        self.assertEqual(rec.termination_reason, "success")
        self.assertEqual(rec.model_calls, 0)
        self.assertEqual(rec.valid_real_actions, 2)
        self.assertEqual(rec.oracle_served, rec.oracle_attempts)

    def test_workflow_first_round_prompt_is_shared_too(self):
        texts = []
        for arm in ("J0", "J1", "J3L"):
            caller = StubCaller(script=["0"])
            I.run_episode(arm, wf_adapter(2), caller, meta(), task="workflow")
            texts.append(caller.prompts[0][1])
        shared = [t[:t.index("ACTIONS REMAINING: 2") + len("ACTIONS REMAINING: 2")]
                  for t in texts]
        self.assertEqual(shared[0], shared[1])
        self.assertEqual(shared[0], shared[2])
        self.assertIn("QUERY LEDGER: (empty)", texts[2][len(shared[2]):])


class TestWorkflowReference(unittest.TestCase):
    @unittest.skipUnless(_WF_DEV.exists(), "dev instances not generated here")
    def test_j5_solves_every_dev_instance_exactly(self):
        instances = json.loads(_WF_DEV.read_text(encoding="utf-8"))
        self.assertEqual(len(instances), 24)          # D21: full strata at m=16
        strata = {}
        for payload in instances:
            strata[payload["difficulty"]] = strata.get(payload["difficulty"], 0) + 1
        self.assertEqual(strata, {"easy": 8, "medium": 8, "hard": 8})
        for payload in instances:
            ad = WorkflowAdapter.from_payload(payload)
            ad.initial_state()
            plan = ad.bfs_plan(cap=ad.action_cap)
            noted = payload["instance_id"]
            self.assertIsNotNone(plan, noted)
            for action in plan:
                ad.apply_real_action(action)
            self.assertTrue(ad.is_success(), noted)
            self.assertEqual(len(plan), payload["shortest_plan_len"], noted)


class TestF4DevSetReference(unittest.TestCase):
    """The F4 dev set is evidence: all 24 envs, full strata, reference-solvable."""

    @unittest.skipUnless(_F4_DEV_DIR.exists(), "f4 dev set not generated here")
    def test_dev_set_is_complete_and_pinned(self):
        manifest = json.loads((_F4_DEV_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocol"], "nextstep-f4-v1")
        self.assertEqual(manifest["counts"], {"easy": 8, "medium": 8, "hard": 8})
        self.assertTrue(manifest["complete"])
        self.assertEqual(len(manifest["environments"]), 24)
        self.assertEqual(
            manifest["result_sha256"],
            "f551e3239df5b35576c59d507591a4a3982bd256d39479621d42cfb0eb8a417e")
        for entry in manifest["environments"]:
            self.assertTrue((_F4_DEV_DIR / entry["file"]).exists())

    @unittest.skipUnless(_F4_DEV_DIR.exists(), "f4 dev set not generated here")
    def test_reference_plan_solves_every_dev_env(self):
        manifest = json.loads((_F4_DEV_DIR / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["environments"]:
            path = _F4_DEV_DIR / entry["file"]
            ad = F4Adapter.from_suite(str(path), action_cap=30, oracle_cap=1000)
            ad.initial_state()
            plan = ad.bfs_plan(cap=ad.action_cap)
            self.assertIsNotNone(plan, entry["env_id"])
            for action in plan:
                ad.apply_real_action(action)
            self.assertTrue(ad.is_success(), entry["env_id"])
            self.assertEqual(len(plan), entry["optimal_steps"], entry["env_id"])
            self.assertEqual(ad.action_cap, 30, "dev envs must be judged at the frozen cap")

    @unittest.skipUnless(_F4_DEV_DIR.exists() and _F4_PUBLISHED.exists(),
                         "needs both the dev set and the published suite")
    def test_dev_envs_are_content_disjoint_from_the_published_suite(self):
        # The published family-4 suite is read-only evidence; a dev env that
        # reproduced one of its environments would violate 1.3 / D10.
        def signature(payload: dict) -> str:
            import hashlib
            keep = {k: payload[k] for k in (
                "family", "grid_size", "cell_colors", "agent_start",
                "agent_start_color", "agent_start_dir", "target_pos",
                "action_type", "action_mapping", "n_actions", "obstacles")}
            return hashlib.sha256(
                json.dumps(keep, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

        published = {signature(json.loads(p.read_text(encoding="utf-8")))
                     for p in _F4_PUBLISHED.rglob("env_*.json")}
        self.assertTrue(published, "the published suite must be present for this check")
        for path in _F4_DEV_DIR.glob("env_*.json"):
            self.assertNotIn(signature(json.loads(path.read_text(encoding="utf-8"))),
                             published, path.name)


if __name__ == "__main__":
    unittest.main()
