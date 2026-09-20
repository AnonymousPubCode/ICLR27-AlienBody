# PROTOCOL — the J interfaces (S1)

    protocol_id      nextstep-interfaces-v0
    config           protocols/nextstep/config.frozen.json
    plan             TODO_NEXTSTEP.md §8.1–§8.3
    status           FROZEN at 8.5 step 6 — see §9 for what may still change

This document is the interface side of the frozen protocol: what every arm is
allowed to see and do, what an episode is, what counts against what, and which
numbers the analysis is allowed to read. The task side (the two environments,
their generators and their reference searchers) is in
[`PROTOCOL_WORKFLOW.md`](PROTOCOL_WORKFLOW.md) for the workflow task and in
§1.2 below for F4.

Every constant quoted here is checkable against the frozen config and the code
hashes it records; nothing in this document is a promise about behaviour, it is
a description of the machinery in `code/alienbody/nextstep/`.

## 1. The two tasks

The same six interfaces run on two environments. A claim is per task unless it
is stated as cross-task.

* **F4** — the frozen family-4 grid environment of the published ladder, reached
  through `ladder_config` and reused read-only (`alienbody/env/*`,
  `alienbody/interface_ladder_v2.py`). The observable state is the JSON state of
  `render_state_v2`; the simulator answers ``(row, col, action) -> (row', col')``
  for the canonical fresh state at a cell.
* **workflow** — the pure-discrete tool-selection environment of S2
  (`alienbody/workflow/`), defined fully in
  [`PROTOCOL_WORKFLOW.md`](PROTOCOL_WORKFLOW.md). The observable state is the
  integer flag vector; the simulator answers ``(bit_state, action_id) ->
  bit_state``.

The **adapter is the only place a task enters an arm**: an arm calls
`initial_state` / `render_public_state` / `apply_real_action` /
`oracle_transition` / `is_success` and nothing task-specific. Adding the
workflow task to a protocol written for F4 did not touch the arms.

Instance sets, both stratified by the reference searcher's optimal plan length
and both split A/B 200/200 with 60/80/60 per block:

| | where | n | strata | observed optimal steps |
|---|---|---|---|---|
| F4 dev | `data/frozen_suites/nextstep_f4_dev/` | 24 | 8/8/8 | 5–20 |
| F4 formal | `data/frozen_suites/nextstep_f4_formal/` | 400 | 120/160/120 | 5–22 |
| workflow dev | `data/frozen_suites/nextstep_workflow_dev/` | 24 | 8/8/8 | 2–9 |
| workflow formal | `data/frozen_suites/nextstep_workflow_formal/` | 400 | 120/160/120 | 2–9 |

F4's bands are easy 5–8, medium 9–14, hard 15–22 (22 = the 30-action cap − 8
slack); the workflow's are in its protocol document. Dev and formal sets are
disjoint by content signature on F4 and by isomorphism on the workflow, and
neither may be pooled with the published family-4 suite or with any pre-S1
series.

## 2. The six interfaces

| arm | what the model is given | memory carried across calls | simulator | program |
|---|---|---|---|---|
| J0 | event table | the table | no | no |
| J1 | event table | the table + a free-text draft (≤ 2000 chars) | no | no |
| J2 | the initial STATE only | none — one plan, no revision | no | no |
| J3L | event table + query ledger | the table + ledger + draft; **each round is a fresh context** | yes | no |
| J4 | event table + query ledger as a callable | the sandbox's own memory within a round | yes | yes |
| J5 | — (no model) | — | same simulator | deterministic BFS |

What is held constant across all arms: the explicit STATE (same JSON, same
renderer), the action mapping, the goal, the remaining-action count, the
simulator's answer set, and the decoder settings (§8). The arms differ only in
**channel**: whether the model may write to itself (J1/J3L draft), whether it
may ask the simulator (J3L/J4/J5), and whether it may write a program instead of
an action (J4).

J5 is the **reference**, not a competitor: it is the environments' own
deterministic searcher and it exists to bound what a correct search costs
(queries and steps) on the same instance set. It calls no model.

## 3. One episode

An episode is one instance, one arm, one model, one seed, starting from a fresh
environment. Rounds are stepwise: the arm renders the observation, the caller
makes at most one request, the arm parses, the environment applies the effect,
and the record is appended. Nothing is carried between episodes.

Budgets per episode (frozen; `budgets` in the config):

| | F4 | workflow |
|---|---|---|
| real actions | 30 | 24 |
| simulator queries served | 1024 | 4096 |
| J3L model-call ceiling | 1088 | 4148 |
| J4 program submissions | 4 | 4 |
| J2 plan length cap | 30 | 30 |

What each event costs:

* a **real action** — valid or not — spends one step of the action budget
  (`invalid_real_action: counts as one spent real step`);
* a **query** — valid or not — spends one query of the oracle budget and, for
  J3L, one model call (`invalid_oracle_query: counts as one spent oracle
  attempt; real state unchanged`);
* a **J4 program submission** spends one of the four rounds; every oracle call
  the program makes draws on the same query budget as J3L's;
* a **parse failure or format error** costs a model call and, in the stepwise
  arms, the round: no repair, no re-ask beyond the arm's own loop.

The J3L call ceiling `oracle_cap + 2*action_cap + 4` is a *ceiling*, not an
expectation: it bounds the arm's call count so that a looping model terminates
deterministically rather than by wall clock. Reaching it terminates the episode
as `model_call_cap` — a measured outcome, not a crash.

Truncation is never silent: an arm response that hits the token limit is a
recorded outcome with its own termination class, and J2's plan is validated
whole — a list that does not parse, is empty or exceeds the cap ends the
episode (`plan_over_cap` / `plan_invalid`) instead of being executed partially.

## 4. What is never in an observation

The observations are *public*: the rendering a model receives is derived from
`render_public_state` / `state_handle` and never from grading labels. In
particular no arm ever sees: the instance's `difficulty`, its
`shortest_plan_len`, its `schema_fingerprint` (workflow), the hidden transition
schema beyond what a query answered, the simulator's internal query log, the
action budget's *identity*, or any other arm's record. The workflow observation
never names which precondition of a tool failed — a blocked action is
observationally identical to a different blocked action.

This is asserted offline, not by inspection: `code/tests/test_nextstep_stubs.py`
checks that the public payload carries no grading label and that the common
STATE is identical across J0/J1/J3L.

## 5. Termination vocabulary

Every episode ends in exactly one of these, recorded in
`termination_reason` — the vocabulary `records.py:TERMINATION_REASONS`
validates against, listed here in its order:

`success` · `action_budget_exhausted` · `plan_invalid` · `plan_over_cap` ·
`plan_exhausted` · `parse_failed` · `model_call_cap` · `context_failure` ·
`transport_failure` · `oracle_exhausted_terminal` · `internal_error`

Where the J2/J4 endings come from, exactly: `plan_over_cap` is J2's plan
exceeding the 30-step cap; `parse_failed` is J2's one-shot reply being unusable
as a plan for any other reason (not a list, empty, unparseable) — the difference
matters because the first is a budget statement and the second is a format
failure; `plan_invalid` is J4 exhausting its four submissions without a working
program, with code errors recorded; `internal_error` is the same path reached
with no code errors, i.e. a harness state the runner did not expect, and it is
the only reason in this list that indicates the machinery rather than the
model.

`success` requires both the arm's termination *and* the adapter's
`is_success()` — an episode that ends on the environment's goal test. Transport
and context failures are recorded with `None` token fields where the provider
reported nothing (never 0). The retry chain is the whole allowance: at most one
re-send of an identical payload after 3 s, no error text is ever shown to the
model, and a call that fails after its single retry **ends the episode** as
`transport_failure` (or `context_failure` when the failure is a context
overflow) — there is no episode-level re-try.

## 6. Accounting

Each episode records its events, its model calls (tokens in/out/cached/
reasoning, attempts, wall seconds) and its counters. Each cell writes one JSONL
(append is impossible: the writer creates the file exclusively) and one
manifest carrying:

* `dataset` — path, n, `dataset_sha256`, and for a narrowed block a
  `selection_sha256` over (id, canonical instance hash) pairs;
* `code_hashes_at_launch`, `prompt_sha256`, `parser_sha256` — the code that
  produced the numbers;
* `budgets`, `transport`, `decoding` — including the resolved temperature;
* `summary` — success count, termination histogram, call/oracle/code counters,
  token totals, wall seconds;
* `result_file` + `result_sha256`, and `resume_chain` when a run was resumed.

The manifest's `summary` is recomputed from the stored episodes by the same
function the tests use (`summarize_records`), so a summary is checkable against
its record file rather than trusted.

## 7. Endpoints and the analysis family

Pre-registered before any formal episode ran (8.5 step 4; `analysis` in the
config):

* **Primary endpoint** — `J4 − J3L`, paired by environment, one test per model:
  env-paired bootstrap CI + exact McNemar.
* **Per task** — the three models as one family, Holm, α = .05/3 = .01667.
* **Cross-task** ("the advantage is universal") — the six per-task tests as one
  family, Holm, α = .05/6 = .00833.

`no_post_hoc_change`: the members and the alpha split are frozen. With n = 400
per task the design's detection limit at discordance .4 is ~79% power at 10 pp
(see `notes/NEXTSTEP_POWER.md`); at discordance .2 it is 98.7%. The other arms
are secondary and are reported with the same test machinery but without the
pre-registered status.

## 8. Frozen constants

| constant | value | where |
|---|---|---|
| `ARM_RESPONSE_TOKENS` | J0 2048, J1 16000, J2 32768, J3L 16000, J4 16000, J5 0 | `interfaces.py` |
| `DRAFT_MAX_CHARS` | 2000 | `interfaces.py` |
| `CODE_SUBMISSIONS_MAX` | 4 | `interfaces.py` |
| `J2_PLAN_CAP` | 30 | `interfaces.py` |
| decoding | temperature 1.0, provider default, identical across arms; never 0 | `decoding` |
| retry | ≤ 1 identical payload, 3 s backoff, no error-text steering | `decoding` |
| budgets | see §3 | `budgets` |

The prompt and parser hashes of every arm on both tasks, the code hashes of the
modules that run them, and the arm response caps are written into
`config.frozen.json` under `code.frozen_code_hashes` at freeze time. **A formal
manifest whose hashes differ from the frozen config was produced by different
code and is not analysable** — the batch is re-run rather than interpreted.

## 9. Change control

The frozen config, the two frozen sets and their A/B split are fixed at 8.5
step 6. After the first formal episode exists, any change to an interface,
budget, prompt, parser, dataset or family is a **protocol deviation**: it is
recorded in `DECISIONS.md` with its reason, the affected cells are named, and
the run is reported with the deviation visible rather than folded in silently.
Numbers from the superseded dev series and from the older ladder (I-arms,
GPT-5.1, m=6 workflow) must never be pooled with S1 numbers.
