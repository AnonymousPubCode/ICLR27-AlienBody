"""Independent reference transition: per-flag boolean logic over plain lists.

This module exists to cross-check :mod:`alienbody.workflow.env`.  It decodes the
four-mask action schema into lists of ``(flag, value)`` literal pairs and runs
the transition as a literal walk over a ``list[bool]`` — no bitmask state
arithmetic — so a bit-level bug in the fast implementation cannot hide behind a
shared code path.  ``tests/test_workflow.py`` asserts the two agree on every
``(state, action)`` pair of every dev instance.

The decode step itself reads the masks (the on-disk encoding is masks); what is
independent is the *transition*: condition evaluation and simultaneous
assignment are expressed per flag.
"""
from __future__ import annotations

from typing import Sequence

from alienbody.workflow.env import Action, WorkflowInstance

# A literal is ``(flag_index, required_value)``.
Literal = tuple[int, bool]


# ── Mask <-> literal-list decoding ─────────────────────────────────

def condition_literals(action: Action, k: int) -> list[Literal]:
    """Precondition literals as ``(flag, value)`` pairs, low flag first."""
    pre_on, pre_off, _set_on, _set_off = action
    literals = [(flag, True) for flag in range(k) if (pre_on >> flag) & 1]
    literals += [(flag, False) for flag in range(k) if (pre_off >> flag) & 1]
    return literals


def assignment_literals(action: Action, k: int) -> list[Literal]:
    """Assignment targets as ``(flag, value)`` pairs, low flag first."""
    _pre_on, _pre_off, set_on, set_off = action
    literals = [(flag, True) for flag in range(k) if (set_on >> flag) & 1]
    literals += [(flag, False) for flag in range(k) if (set_off >> flag) & 1]
    return literals


def decode_flags(bit_state: int, k: int) -> list[bool]:
    """Bit vector -> per-flag booleans, index ``i`` is flag ``f_i``."""
    return [bool((bit_state >> flag) & 1) for flag in range(k)]


def encode_flags(flags: Sequence[bool]) -> int:
    """Per-flag booleans -> bit vector."""
    value = 0
    for flag, is_set in enumerate(flags):
        if is_set:
            value |= 1 << flag
    return value


# ── Reference transition ───────────────────────────────────────────

def reference_preconditions_hold(conditions: Sequence[Literal], flags: Sequence[bool]) -> bool:
    """True iff every literal in ``conditions`` matches ``flags``."""
    for flag, required in conditions:
        if bool(flags[flag]) != required:
            return False
    return True


def reference_apply(flags: list[bool], assignments: Sequence[Literal]) -> None:
    """Simultaneous assignment: targets are distinct, so order cannot matter."""
    for flag, value in assignments:
        flags[flag] = value


def reference_next_state(instance: WorkflowInstance, bit_state: int, action_id: int) -> int:
    """Reference transition; ``ValueError`` on out-of-range state or action id."""
    if not 0 <= bit_state < (1 << instance.k):
        raise ValueError(f"state {bit_state} outside [0, {1 << instance.k})")
    if not 0 <= action_id < instance.m:
        raise ValueError(f"action id {action_id} outside [0, {instance.m})")

    action = instance.actions[action_id]
    flags = decode_flags(bit_state, instance.k)
    conditions = condition_literals(action, instance.k)
    if reference_preconditions_hold(conditions, flags):
        reference_apply(flags, assignment_literals(action, instance.k))
    return encode_flags(flags)


def reference_is_success(instance: WorkflowInstance, bit_state: int) -> bool:
    """Reference goal check, spelled out per goal literal (no bitmask test)."""
    flags = decode_flags(bit_state, instance.k)
    on_literals = [(flag, True) for flag in range(instance.k) if (instance.goal_on >> flag) & 1]
    off_literals = [(flag, False) for flag in range(instance.k) if (instance.goal_off >> flag) & 1]
    return reference_preconditions_hold(on_literals + off_literals, flags)


def reference_action_is_vacuous(action: Action, k: int) -> bool:
    """Brute-force no-op check: enumerate every precondition-satisfying state."""
    conditions = condition_literals(action, k)
    assignments = assignment_literals(action, k)
    for bit_state in range(1 << k):
        flags = decode_flags(bit_state, k)
        if not reference_preconditions_hold(conditions, flags):
            continue
        before = list(flags)
        reference_apply(flags, assignments)
        if flags != before:
            return False
    return True
