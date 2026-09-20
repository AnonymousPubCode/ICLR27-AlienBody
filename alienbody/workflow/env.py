"""Pure-discrete workflow environment: bitmask transition, budget, goal check.

Frozen spec ``workflow-dev-v1``.  A state is ``k`` boolean device flags rendered
as a bit vector (flag ``f_i`` is bit ``i``); default ``k=8`` so there are 256
states.  An action ("tool") is four bitmasks ``(pre_on, pre_off, set_on,
set_off)``: it fires iff every ``pre_on`` flag is set and every ``pre_off`` flag
is clear, and firing applies the *simultaneous* assignment
``state = (state | set_on) & ~set_off``.  A blocked action leaves the state
untouched and still consumes one step; the observation never says which
precondition failed (see :mod:`alienbody.workflow.observation`).

Nothing here touches the network, the GPU, or the global RNG.
"""
from __future__ import annotations

from dataclasses import dataclass

# One action: (pre_on, pre_off, set_on, set_off).
Action = tuple[int, int, int, int]

# Frozen spec limits on one action's schema.
MAX_PRECONDITION_LITERALS = 2
MAX_ASSIGNMENT_TARGETS = 2
MIN_ASSIGNMENT_TARGETS = 1

# Default hidden-state dimension (256 states) and tool vocabulary size.
DEFAULT_K = 8
DEFAULT_M = 6


@dataclass(frozen=True)
class WorkflowInstance:
    """One workflow instance: the hidden transition schema and its goal.

    ``actions`` is the tool vocabulary as a sequence of four-mask tuples; the
    action id is the position in that sequence.  ``schema_fingerprint``,
    ``difficulty`` and ``shortest_plan_len`` are *grading-only* labels: they
    never enter the agent's observation.
    """

    instance_id: str
    k: int
    m: int
    start: int
    goal_on: int
    goal_off: int
    actions: tuple[Action, ...]
    schema_fingerprint: str
    difficulty: str
    shortest_plan_len: int

    @property
    def n_states(self) -> int:
        """Number of states in the flag space (``2 ** k``)."""
        return 1 << self.k

    @property
    def state_mask(self) -> int:
        """Mask keeping a state inside the ``k``-flag universe."""
        return (1 << self.k) - 1

    def to_dict(self) -> dict:
        """JSON-ready payload with the full schema."""
        return {
            "instance_id": self.instance_id,
            "k": self.k,
            "m": self.m,
            "start": self.start,
            "goal_on": self.goal_on,
            "goal_off": self.goal_off,
            "actions": [list(action) for action in self.actions],
            "schema_fingerprint": self.schema_fingerprint,
            "difficulty": self.difficulty,
            "shortest_plan_len": self.shortest_plan_len,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkflowInstance":
        """Rebuild an instance from :meth:`to_dict` output."""
        return cls(
            instance_id=payload["instance_id"],
            k=int(payload["k"]),
            m=int(payload["m"]),
            start=int(payload["start"]),
            goal_on=int(payload["goal_on"]),
            goal_off=int(payload["goal_off"]),
            actions=tuple(tuple(int(mask) for mask in action) for action in payload["actions"]),
            schema_fingerprint=payload["schema_fingerprint"],
            difficulty=payload["difficulty"],
            shortest_plan_len=int(payload["shortest_plan_len"]),
        )


# ── Transition ─────────────────────────────────────────────────────

def preconditions_hold(action: Action, bit_state: int) -> bool:
    """True iff ``bit_state`` satisfies every literal of ``action``."""
    pre_on, pre_off, _set_on, _set_off = action
    return (bit_state & pre_on) == pre_on and (bit_state & pre_off) == 0


def assign_effects(action: Action, bit_state: int, k: int) -> int:
    """Apply the simultaneous assignment ``(state | set_on) & ~set_off``."""
    _pre_on, _pre_off, set_on, set_off = action
    return (bit_state | set_on) & ~set_off & ((1 << k) - 1)


def oracle_next_state(instance: WorkflowInstance, bit_state: int, action_id: int) -> int:
    """Ground-truth transition.  Raises ``ValueError`` on out-of-range input."""
    if not 0 <= bit_state < (1 << instance.k):
        raise ValueError(f"state {bit_state} outside [0, {1 << instance.k})")
    if not 0 <= action_id < instance.m:
        raise ValueError(f"action id {action_id} outside [0, {instance.m})")
    action = instance.actions[action_id]
    if not preconditions_hold(action, bit_state):
        return bit_state
    return assign_effects(action, bit_state, instance.k)


def goal_satisfied(goal_on: int, goal_off: int, bit_state: int) -> bool:
    """True iff ``bit_state`` matches the partial goal assignment."""
    return (bit_state & goal_on) == goal_on and (bit_state & goal_off) == 0


def is_success(instance: WorkflowInstance, bit_state: int) -> bool:
    """True iff ``bit_state`` satisfies the instance goal."""
    return goal_satisfied(instance.goal_on, instance.goal_off, bit_state)


# ── Schema validation ──────────────────────────────────────────────

def action_is_vacuous(action: Action) -> bool:
    """True iff the assignment is a no-op in *every* precondition-satisfying state.

    A ``set_on`` bit that no precondition forces to 1 can still be observed as
    0 in some satisfying state, and symmetrically for ``set_off``/``pre_off``; if
    neither exists the action can never change anything and must be rejected.
    """
    pre_on, pre_off, set_on, set_off = action
    return (set_on & ~pre_on) == 0 and (set_off & ~pre_off) == 0


def action_is_effective(action: Action, k: int) -> bool:
    """Brute-force twin of :func:`action_is_vacuous` over all ``2 ** k`` states."""
    for bit_state in range(1 << k):
        if preconditions_hold(action, bit_state):
            if assign_effects(action, bit_state, k) != bit_state:
                return True
    return False


def action_problems(action: Action, k: int) -> tuple[str, ...]:
    """Frozen-spec violations of one action schema (empty tuple = valid)."""
    pre_on, pre_off, set_on, set_off = action
    problems: list[str] = []
    if any(mask >> k for mask in action):
        problems.append("flag index outside 0..k-1")
    if pre_on & pre_off:
        problems.append("overlapping precondition literals")
    if set_on & set_off:
        problems.append("conflicting assignment targets")
    if pre_on.bit_count() + pre_off.bit_count() > MAX_PRECONDITION_LITERALS:
        problems.append("more than 2 precondition literals")
    targets = set_on.bit_count() + set_off.bit_count()
    if targets > MAX_ASSIGNMENT_TARGETS:
        problems.append("more than 2 assignment targets")
    if targets < MIN_ASSIGNMENT_TARGETS:
        problems.append("no assignment targets")
    if action_is_vacuous(action):
        problems.append("vacuous")
    return tuple(problems)


def instance_problems(instance: WorkflowInstance) -> tuple[str, ...]:
    """Frozen-spec violations of a whole instance (empty tuple = valid)."""
    problems: list[str] = []
    if len(instance.actions) != instance.m:
        problems.append(f"expected {instance.m} actions, got {len(instance.actions)}")
    if len(set(instance.actions)) != len(instance.actions):
        problems.append("duplicate action schemas")
    if not 0 <= instance.start < (1 << instance.k):
        problems.append("start state outside the flag space")
    if is_success(instance, instance.start):
        problems.append("start state already satisfies the goal")
    if instance.goal_on & instance.goal_off:
        problems.append("overlapping goal literals")
    for action_id, action in enumerate(instance.actions):
        for problem in action_problems(action, instance.k):
            problems.append(f"action {action_id}: {problem}")
    return tuple(problems)


# ── Environment ────────────────────────────────────────────────────

class WorkflowEnv:
    """Step-wise wrapper around an instance: transition, budget, goal check.

    The instance is fixed, so ``reset()`` just restores the start state and the
    step budget.  One call to :meth:`step` consumes exactly one step, whether or
    not the tool fired, and whether or not the action id was in range.
    """

    def __init__(self, instance: WorkflowInstance, step_budget: int = 24) -> None:
        if step_budget < 0:
            raise ValueError(f"step_budget {step_budget} must be non-negative")
        self.instance = instance
        self.step_budget = step_budget
        self._state = instance.start
        self._steps_used = 0

    # -- introspection -------------------------------------------------

    @property
    def state(self) -> int:
        """Current flag bit vector."""
        return self._state

    @property
    def steps_used(self) -> int:
        """Steps consumed since the last :meth:`reset`."""
        return self._steps_used

    @property
    def steps_left(self) -> int:
        """Remaining steps; saturates at 0 once the budget is spent."""
        return max(0, self.step_budget - self._steps_used)

    # -- api -----------------------------------------------------------

    def reset(self) -> tuple[int, int, int]:
        """Restore the start state and the step budget.

        Returns ``(state, goal_on, goal_off)`` — the public observation of the
        episode's first state.
        """
        self._state = self.instance.start
        self._steps_used = 0
        return self._state, self.instance.goal_on, self.instance.goal_off

    def step(self, action_id: int) -> tuple[int, bool, int]:
        """Execute one tool.  Returns ``(state, success, steps_left)``.

        An out-of-range ``action_id`` changes nothing and reports no success but
        still consumes a step, exactly like a tool whose preconditions failed.
        ``success`` is recomputed from the resulting state on every step rather
        than latched.
        """
        self._steps_used += 1
        if not 0 <= action_id < self.instance.m:
            return self._state, False, self.steps_left
        self._state = oracle_next_state(self.instance, self._state, action_id)
        return self._state, is_success(self.instance, self._state), self.steps_left

    def observation(self) -> tuple[int, int, int]:
        """Current public observation ``(state, goal_on, goal_off)``."""
        return self._state, self.instance.goal_on, self.instance.goal_off
