# PROTOCOL — the workflow task (S2)

    task name        AlienWorkflow (flags + tools)
    environment      code/alienbody/workflow/   (spec workflow-dev-v1)
    instance sets    code/data/frozen_suites/nextstep_workflow_dev  (dev, 24)
                     code/data/frozen_suites/nextstep_workflow_formal (formal, 400)
    status           FROZEN at 8.5 step 6

The interface side of the protocol — the six arms, the episode loop, the
budgets, the endpoints — is [`PROTOCOL_NEXTSTEP_INTERFACES.md`](PROTOCOL_NEXTSTEP_INTERFACES.md).
This document defines the *task*: what the environment is, what the agent may
see, how instances are generated and stratified, and what the reference
searcher establishes.

## 1. The environment

A state is a vector of ``k = 8`` boolean device flags (``f7 … f0``, rendered
most-significant flag first), so there are **256 states**. An action ("tool") is
four bitmasks ``(pre_on, pre_off, set_on, set_off)``:

* it **fires** iff every flag in ``pre_on`` is set and every flag in
  ``pre_off`` is clear;
* firing applies the simultaneous assignment
  ``state = (state | set_on) & ~set_off``;
* a **blocked** tool leaves the state untouched, still consumes one step, and
  the observation does not say which precondition failed.

A schema is bounded: at most 2 precondition literals and 1–2 assignment targets
per tool. The goal is a partial assignment of ``3`` literals (``goal_on`` /
``goal_off``); the episode succeeds when every ``goal_on`` flag is set and every
``goal_off`` flag is clear. The action budget is **24** steps — a blocked tool
therefore costs a real step, which is what makes the flag space worth probing.

## 2. What the agent may see

Public (in every arm's STATE): the current flag vector, the goal's partial
assignment as literals, the remaining-action count, and **the exact four masks
of every tool**, by tool id.

Grading-only, never rendered: ``instance_id``, ``difficulty``,
``shortest_plan_len``, ``schema_fingerprint``. `observation.assert_public`
enforces the boundary and the stub regressions assert the public payload carries
no grading label.

**Design consequence, recorded for the analysis.** The masks are public, so a
code arm (J4) can compute the transition itself without spending a query, while
a conversational arm (J3L) can only learn a transition by intervening or asking.
`oracle_served == 0` therefore *must not* be read as "the arm did not simulate";
it means the arm did not use the *oracle*. This is a property of the task, not
an information leak, and it is exactly the difference the J4 − J3L endpoint is
about.

## 3. Instance generation

Generator `code/scripts/gen_workflow_dev.py` (spec ``workflow-dev-v1``). A
candidate is a *schema*, drawn without ever rolling out a trajectory, so the
goal is not biased towards short plans. Each candidate is then solved by an
exhaustive BFS over the whole state space, which decides solvability and the
shortest plan length; the length picks the stratum:

| stratum | shortest plan | quota (formal) |
|---|---|---|
| easy | 2–3 | 120 |
| medium | 4–6 | 160 |
| hard | 7–10 | 120 |

**Isomorphism de-duplication.** Two instances are the same workflow up to (a)
renaming the flags and (b) renumbering the tools. The key canonicalises over all
``k!`` flag relabellings (exact, brute force with prefix pruning at ``k=8``) and
sorts the tool schemas to kill the action-id permutation. Duplicates are
rejected at generation and counted under their own reason. The **formal** set is
additionally generated with `--exclude` against the dev set, so no formal
instance is isomorphic to a dev instance (the cross-set exclusion of §2.3.4,
mechanism landed 2026-09-15 as D24).

**Frozen distribution.** ``k = 8``, ``m = 16`` tools, ``goal_literals = 3``,
same seed family for dev and formal (D21). The dev set was regenerated at
``m = 16`` because at ``m = 6`` the hard stratum was effectively unfillable
(1 instance in 100 000 draws); dev and formal must share one distribution, so
neither may be quoted across that change.

## 4. The reference searcher (J5)

J5 is exhaustive BFS over the flag space using the same oracle the other arms
use, with no model in the loop. It exists to bound what a correct search costs,
not to compete. On the frozen sets it reproduces every instance's exact
``shortest_plan_len``, and its *uncapped* oracle cost is the basis for the
frozen oracle cap:

| | min | median | p90 | max |
|---|---|---|---|---|
| dev 24 (uncapped) | 20 | 263.5 | — | 796 |
| formal 400 | 17 | 255 | 746 | **2178** |

The formal maximum is why the workflow oracle cap is **4096** rather than 2048
(D31): one environment of 400 (`wf8_formal_hard_37`) would have exhausted the
reference arm under the old cap. The set was not re-drawn instead, because
oracle cost correlates with search difficulty and re-drawing would bias the
hardest stratum. 4096 is the smallest power of two above 2178.

## 5. What the two task sides share

Everything else is inherited unchanged from the interfaces protocol: the six
arms, the draft rules, the J4 sandbox, the query ledger, the termination
vocabulary, the accounting and the pre-registered analysis. The workflow adapter
is a drop-in — adding this task to a protocol written for the F4 grid did not
touch any arm, which is the check that §8.3's "the adapter is the only place a
task enters an arm" actually holds.

The frozen instance sets, their A/B split (200/200, 60/80/60 per block) and the
set hashes are recorded in `config.frozen.json`; a formal manifest whose
`dataset_sha256` / `selection_sha256` differs from the frozen entry is not
analysable.
