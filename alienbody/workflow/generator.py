"""Stratified workflow-instance generation with isomorphism de-duplication.

Generation draws *schemas* only — never a rolled-out trajectory — so the goal is
not biased towards short plans.  Each candidate is scored by an exhaustive BFS
over the whole ``2 ** k`` state space, which decides solvability and the
shortest plan length; the length then selects the difficulty stratum
(easy 2-3, medium 4-6, hard 7-10).

Two instances are duplicates when they are the same workflow up to (a) renaming
the flags and (b) renumbering the tools.  The isomorphism key therefore
canonicalises the schema over all ``k!`` flag relabellings (exact, brute force
with prefix pruning — fine at ``k=8``) and sorts the tool schemas to kill the
action-id permutation.

Determinism: every draw comes from an explicit ``random.Random(config.seed)``;
nothing reads global RNG state.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from alienbody.workflow.baselines import (
    DEFAULT_STRATA,
    shortest_plan_length,
    stratum_for_length,
)
from alienbody.workflow.env import (
    Action,
    WorkflowInstance,
    action_problems,
    goal_satisfied,
    instance_problems,
)

PROTOCOL = "workflow-dev-v1"
DEFAULT_SEED = 20260914
DEFAULT_MAX_ATTEMPTS = 100_000
DEFAULT_PER_DIFFICULTY = 8

# One raw instance schema: (start, goal_on, goal_off, actions).
Schema = tuple[int, int, int, tuple[Action, ...]]


@dataclass(frozen=True)
class GenConfig:
    """Frozen generation settings for one dev set."""

    seed: int = DEFAULT_SEED
    k: int = 8
    m: int = 16          # D21: m=6 cannot reach the hard stratum (1/100000 draws)
    goal_literals: int = 3
    per_difficulty: int | dict[str, int] = DEFAULT_PER_DIFFICULTY
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    strata: tuple[tuple[str, int, int], ...] = DEFAULT_STRATA
    set_tag: str = ""    # provenance tag in every instance_id ("" for the dev set)

    def quota_for(self, stratum: str) -> int:
        """The requested count for one stratum.

        ``per_difficulty`` is normally the uniform int the dev set uses; the
        formal set needs the 30/40/30 shape (D25), which an int cannot express,
        so a mapping is accepted instead.  One code path reads quotas through
        this method so a mapping can never be honoured in one place and
        silently ignored in another.
        """
        if isinstance(self.per_difficulty, dict):
            return int(self.per_difficulty[stratum])
        return int(self.per_difficulty)

    @property
    def requested_counts(self) -> dict[str, int]:
        """How many instances each stratum should end up with."""
        return {name: self.quota_for(name) for name, _low, _high in self.strata}


@dataclass
class GenerationResult:
    """Outcome of one stratified generation run, including the honest books."""

    config: GenConfig
    instances: list[WorkflowInstance] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    attempts: int = 0
    rejection_histogram: dict[str, int] = field(default_factory=dict)
    forbidden_keys_count: int = 0

    @property
    def strata_filled(self) -> dict[str, bool]:
        """Per-stratum: did this run reach the requested count?"""
        return {
            name: self.counts.get(name, 0) >= self.config.quota_for(name)
            for name, _low, _high in self.config.strata
        }

    @property
    def complete(self) -> bool:
        """True iff every stratum reached its requested count."""
        return all(self.strata_filled.values())

    @property
    def missing(self) -> tuple[str, ...]:
        """Names of the strata that could not be filled."""
        return tuple(name for name, filled in self.strata_filled.items() if not filled)


# ── Candidate drawing (frozen spec) ────────────────────────────────

def draw_action(rng: random.Random, k: int) -> Action:
    """Draw one tool: 1-2 precondition literals, then 1-2 assignment targets.

    Flags are distinct within the precondition draw and within the effect draw;
    effects may re-target precondition flags, as the spec allows.
    """
    pre_on = pre_off = 0
    for flag in rng.sample(range(k), rng.choice((1, 2))):
        if rng.random() < 0.5:
            pre_on |= 1 << flag
        else:
            pre_off |= 1 << flag

    set_on = set_off = 0
    for flag in rng.sample(range(k), rng.choice((1, 2))):
        if rng.random() < 0.5:
            set_on |= 1 << flag
        else:
            set_off |= 1 << flag

    return (pre_on, pre_off, set_on, set_off)


def draw_candidate(rng: random.Random, config: GenConfig) -> tuple[Schema | None, str | None]:
    """Draw one raw schema.  Returns ``(schema, None)`` or ``(None, reason)``."""
    actions: list[Action] = []
    for _ in range(config.m):
        action = draw_action(rng, config.k)
        problems = action_problems(action, config.k)
        if problems:
            reason = "vacuous_action" if "vacuous" in problems else "invalid_action_schema"
            return None, reason
        actions.append(action)
    if len(set(actions)) != config.m:
        return None, "duplicate_action_schema"

    start = rng.randrange(1 << config.k)
    goal_on = goal_off = 0
    for flag in rng.sample(range(config.k), config.goal_literals):
        if rng.random() < 0.5:
            goal_on |= 1 << flag
        else:
            goal_off |= 1 << flag
    if goal_satisfied(goal_on, goal_off, start):
        return None, "start_satisfies_goal"
    return (start, goal_on, goal_off, tuple(actions)), None


# ── Isomorphism key ────────────────────────────────────────────────

def permute_mask(mask: int, dest: tuple[int, ...]) -> int:
    """Relabel one mask: source flag ``i`` becomes flag ``dest[i]``."""
    if mask == 0:
        return 0
    if mask & (mask - 1) == 0:  # single flag, the common case
        return 1 << dest[mask.bit_length() - 1]
    out = 0
    while mask:
        low = mask & -mask
        out |= 1 << dest[low.bit_length() - 1]
        mask ^= low
    return out


def permute_action(action: Action, dest: tuple[int, ...]) -> Action:
    """Relabel all four masks of one action."""
    return tuple(permute_mask(mask, dest) for mask in action)  # type: ignore[return-value]


def inverse_permutation(dest: tuple[int, ...]) -> tuple[int, ...]:
    """The permutation that undoes ``dest``."""
    inverse = [0] * len(dest)
    for source, target in enumerate(dest):
        inverse[target] = source
    return tuple(inverse)


def sorted_actions(actions: tuple[Action, ...] | list[Action]) -> tuple[Action, ...]:
    """Sort tool schemas, which kills the action-id permutation."""
    return tuple(sorted(tuple(int(mask) for mask in action) for action in actions))


def relabel(
    start: int,
    goal_on: int,
    goal_off: int,
    actions: tuple[Action, ...] | list[Action],
    dest: tuple[int, ...],
) -> Schema:
    """Apply a flag relabelling to a raw schema."""
    return (
        permute_mask(start, dest),
        permute_mask(goal_on, dest),
        permute_mask(goal_off, dest),
        tuple(permute_action(action, dest) for action in actions),
    )


def canonical_key(
    k: int,
    m: int,
    start: int,
    goal_on: int,
    goal_off: int,
    actions: tuple[Action, ...] | list[Action],
) -> tuple:
    """Exact isomorphism key: minimal canonical form over all ``k!`` relabellings.

    The minimum is taken over the whole permutation orbit, so isomorphic
    schemas always produce the same key (tool order is normalised by sorting).
    Permutations are pruned as soon as their partially built key already
    exceeds the best complete key, which keeps ``k=8`` (8! = 40320) cheap.
    """
    schemas = sorted_actions(actions)
    best: tuple | None = None
    for dest in itertools.permutations(range(k)):
        start_p = permute_mask(start, dest)
        if best is not None and start_p > best[0]:
            continue
        goal_on_p = permute_mask(goal_on, dest)
        if best is not None and start_p == best[0] and goal_on_p > best[1]:
            continue
        goal_off_p = permute_mask(goal_off, dest)
        if (best is not None and start_p == best[0] and goal_on_p == best[1]
                and goal_off_p > best[2]):
            continue
        actions_p = tuple(sorted(permute_action(action, dest) for action in schemas))
        key = (start_p, goal_on_p, goal_off_p, actions_p)
        if best is None or key < best:
            best = key
    assert best is not None, "k must be at least 1"
    return (k, m, best[0], best[1], best[2], best[3])


def canonical_key_of(instance: WorkflowInstance) -> tuple:
    """Isomorphism key of an instance (same key <=> same workflow)."""
    return canonical_key(
        instance.k, instance.m, instance.start,
        instance.goal_on, instance.goal_off, instance.actions,
    )


def schema_fingerprint(key: tuple) -> str:
    """Stable short digest of an isomorphism key (first 16 hex chars of sha256)."""
    payload = json.dumps(key, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def is_isomorphic(first: WorkflowInstance, second: WorkflowInstance) -> bool:
    """True iff the two instances differ only by flag and tool renumbering."""
    return canonical_key_of(first) == canonical_key_of(second)


class IsomorphismRegistry:
    """Remembers accepted workflows so renumberings are not generated twice.

    ``forbidden_keys`` are keys that are refused without being registered:
    they belong to another set (e.g. the dev set when generating the formal
    set) and cross-set isomorphism is a rejection, per 2.3.4.  The two
    refusals are distinguishable — :meth:`is_forbidden` is checked before
    :meth:`add`, so a caller can count them under separate reasons.
    """

    def __init__(self, forbidden_keys: "Iterable[tuple]" = ()) -> None:
        self._by_key: dict[tuple, WorkflowInstance] = {}
        self._forbidden: set[tuple] = set(forbidden_keys)

    def __len__(self) -> int:
        return len(self._by_key)

    @property
    def keys(self) -> tuple[tuple, ...]:
        """Accepted isomorphism keys, in insertion order."""
        return tuple(self._by_key)

    @property
    def forbidden_keys(self) -> frozenset[tuple]:
        """Keys refused by construction (another set's keys)."""
        return frozenset(self._forbidden)

    def is_forbidden(self, instance: WorkflowInstance, key: tuple | None = None) -> bool:
        """True iff this instance's key belongs to the excluded set."""
        if key is None:
            key = canonical_key_of(instance)
        return key in self._forbidden

    def contains(self, instance: WorkflowInstance) -> bool:
        """True iff an isomorphic instance was already accepted."""
        return canonical_key_of(instance) in self._by_key

    def add(self, instance: WorkflowInstance, key: tuple | None = None) -> bool:
        """Register an instance.  False (and no change) if it is a duplicate
        or one of the forbidden keys.

        ``key`` may be passed when the caller already computed
        :func:`canonical_key` for this schema.
        """
        if key is None:
            key = canonical_key_of(instance)
        if key in self._by_key or key in self._forbidden:
            return False
        self._by_key[key] = instance
        return True


# ── Stratified generation ──────────────────────────────────────────

def generate(config: GenConfig, forbidden_keys: Iterable[tuple] = ()) -> GenerationResult:
    """Fill every stratum up to ``per_difficulty`` or exhaust ``max_attempts``.

    ``forbidden_keys`` are isomorphism keys of another set (the dev set, when
    generating the formal one): candidates whose key is in that set are
    rejected under their own reason, ``excluded_isomorph``, and never appear in
    the result (2.3.4 cross-set exclusion).

    A run that exhausts the attempt cap reports the strata it could not fill and
    the per-reason rejection counts.  It never substitutes a candidate from a
    different stratum, and it never invents instances.
    """
    rng = random.Random(config.seed)
    forbidden = set(forbidden_keys)
    registry = IsomorphismRegistry(forbidden_keys=forbidden)
    result = GenerationResult(config=config)
    result.forbidden_keys_count = len(forbidden)
    result.counts = {name: 0 for name, _low, _high in config.strata}
    rejection: Counter[str] = Counter()

    def needs_more() -> bool:
        return any(result.counts[name] < config.quota_for(name)
                   for name, _low, _high in config.strata)

    while result.attempts < config.max_attempts and needs_more():
        result.attempts += 1
        schema, reason = draw_candidate(rng, config)
        if schema is None:
            rejection[reason] += 1
            continue
        start, goal_on, goal_off, actions = schema

        probe = WorkflowInstance(
            instance_id="probe", k=config.k, m=config.m, start=start,
            goal_on=goal_on, goal_off=goal_off, actions=actions,
            schema_fingerprint="", difficulty="", shortest_plan_len=0,
        )
        plan_len = shortest_plan_length(probe)
        if plan_len is None:
            rejection["unsolvable"] += 1
            continue
        stratum = stratum_for_length(plan_len, config.strata)
        if stratum is None:
            rejection["length_out_of_range"] += 1
            continue
        if result.counts[stratum] >= config.quota_for(stratum):
            rejection["stratum_full"] += 1
            continue

        key = canonical_key(config.k, config.m, start, goal_on, goal_off, actions)
        instance = WorkflowInstance(
            instance_id=f"wf{config.k}{config.set_tag}_{stratum}_{result.counts[stratum]:02d}",
            k=config.k, m=config.m, start=start,
            goal_on=goal_on, goal_off=goal_off, actions=actions,
            schema_fingerprint=schema_fingerprint(key),
            difficulty=stratum, shortest_plan_len=plan_len,
        )
        problems = instance_problems(instance)
        if problems:
            rejection[f"invalid_instance:{problems[0]}"] += 1
            continue
        if registry.is_forbidden(instance, key=key):
            rejection["excluded_isomorph"] += 1
            continue
        if not registry.add(instance, key=key):
            rejection["duplicate_isomorph"] += 1
            continue

        result.counts[stratum] += 1
        result.instances.append(instance)

    order = {name: index for index, (name, _low, _high) in enumerate(config.strata)}
    result.instances.sort(key=lambda item: (order[item.difficulty], item.instance_id))
    result.rejection_histogram = dict(sorted(rejection.items()))
    return result


# ── Manifest ───────────────────────────────────────────────────────

def build_manifest(result: GenerationResult, result_sha256: str) -> dict:
    """Manifest for a written dev set; ``result_sha256`` hashes the instance file."""
    return {
        "protocol": PROTOCOL,
        "seed": result.config.seed,
        "k": result.config.k,
        "m": result.config.m,
        "counts": dict(result.counts),
        "requested_counts": result.config.requested_counts,
        "per_difficulty": result.config.per_difficulty,
        "instances": [
            {
                "instance_id": instance.instance_id,
                "schema_fingerprint": instance.schema_fingerprint,
                "difficulty": instance.difficulty,
                "shortest_plan_len": instance.shortest_plan_len,
            }
            for instance in result.instances
        ],
        "result_sha256": result_sha256,
        "attempts": result.attempts,
        "max_attempts": result.config.max_attempts,
        "strata_filled": result.strata_filled,
        "missing_strata": list(result.missing),
        "complete": result.complete,
        "rejection_histogram": dict(result.rejection_histogram),
        "forbidden_keys_count": result.forbidden_keys_count,
    }


def stable_json(value: object) -> str:
    """Deterministic JSON text: sorted keys, LF line endings, trailing newline."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
