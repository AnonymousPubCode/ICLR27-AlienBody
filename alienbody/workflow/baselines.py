"""Deterministic BFS baselines over the workflow transition.

The state space is tiny (``2 ** k`` = 256 by default) so the optimal plan is
always computable exactly.  Expansion is breadth-first with ascending action
ids and first-discovery parents, which makes the returned plan the
lexicographically smallest among the shortest ones — no RNG, no tie-breaking by
dictionary order of anything.

Every function takes the transition as a parameter so the tests can run the
same search over :func:`alienbody.workflow.env.oracle_next_state` and over the
independent :func:`alienbody.workflow.reference.reference_next_state`.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

from alienbody.workflow.env import WorkflowInstance, is_success, oracle_next_state

# (instance, bit_state, action_id) -> next bit_state
Transition = Callable[[WorkflowInstance, int, int], int]

# Difficulty bucketing frozen by the dev-set protocol: shortest plan length.
DEFAULT_STRATA: tuple[tuple[str, int, int], ...] = (
    ("easy", 2, 3),
    ("medium", 4, 6),
    ("hard", 7, 10),
)


def stratum_for_length(
    length: int | None,
    strata: tuple[tuple[str, int, int], ...] = DEFAULT_STRATA,
) -> str | None:
    """Difficulty bucket of a plan length, or ``None`` if it falls outside."""
    if length is None:
        return None
    for name, low, high in strata:
        if low <= length <= high:
            return name
    return None


def shortest_plan(
    instance: WorkflowInstance,
    transition: Transition = oracle_next_state,
    max_len: int | None = None,
) -> tuple[int, ...] | None:
    """Lexicographically smallest shortest plan, or ``None`` if unreachable.

    ``max_len`` stops the search early once that many actions have been laid
    down; without it the whole reachable component is explored.
    """
    start = instance.start
    if is_success(instance, start):
        return ()

    parent: dict[int, tuple[int, int]] = {}
    frontier = [start]
    depth = 0
    while frontier:
        if max_len is not None and depth >= max_len:
            break
        depth += 1
        next_frontier: list[int] = []
        for state in frontier:
            for action_id in range(instance.m):
                successor = transition(instance, state, action_id)
                if successor in parent or successor == start:
                    continue  # already reached (self-loops included)
                parent[successor] = (state, action_id)
                if is_success(instance, successor):
                    return _unroll_plan(parent, start, successor)
                next_frontier.append(successor)
        frontier = next_frontier
    return None


def _unroll_plan(parent: dict[int, tuple[int, int]], start: int, goal: int) -> tuple[int, ...]:
    """Walk first-discovery parents back to the start state."""
    plan: list[int] = []
    state = goal
    while state != start:
        state, action_id = parent[state]
        plan.append(action_id)
    plan.reverse()
    return tuple(plan)


def shortest_plan_length(
    instance: WorkflowInstance,
    transition: Transition = oracle_next_state,
    max_len: int | None = None,
) -> int | None:
    """Length of the shortest plan, or ``None`` if the goal is unreachable."""
    plan = shortest_plan(instance, transition, max_len)
    return None if plan is None else len(plan)


def replay_plan(
    instance: WorkflowInstance,
    plan: tuple[int, ...] | list[int],
    transition: Transition = oracle_next_state,
) -> tuple[int, ...]:
    """State trace of ``plan``: ``len(plan) + 1`` states, start state first.

    Raises ``ValueError`` (from the transition) on an out-of-range action id.
    """
    state = instance.start
    trace = [state]
    for action_id in plan:
        state = transition(instance, state, action_id)
        trace.append(state)
    return tuple(trace)


def plan_succeeds(
    instance: WorkflowInstance,
    plan: tuple[int, ...] | list[int],
    transition: Transition = oracle_next_state,
) -> bool:
    """True iff executing ``plan`` from the start state reaches the goal."""
    trace = replay_plan(instance, plan, transition)
    return is_success(instance, trace[-1])


def shortest_distances(
    instance: WorkflowInstance,
    transition: Transition = oracle_next_state,
) -> dict[int, int]:
    """Distance from the start state to every reachable state."""
    start = instance.start
    distances = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        state = queue.popleft()
        for action_id in range(instance.m):
            successor = transition(instance, state, action_id)
            if successor not in distances:
                distances[successor] = distances[state] + 1
                queue.append(successor)
    return distances


def reachable_states(
    instance: WorkflowInstance,
    transition: Transition = oracle_next_state,
) -> tuple[int, ...]:
    """Every state reachable from the start, in ascending order."""
    return tuple(sorted(shortest_distances(instance, transition)))
