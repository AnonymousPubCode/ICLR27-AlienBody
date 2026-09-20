"""Canonical public rendering of a workflow state, and the public/hidden boundary.

What a solver may see
    the current flag vector, the goal's partial assignment, and the tool
    schemas.  After a step it sees the new flag vector — nothing else.

What is grading-only
    ``instance_id``, ``difficulty``, ``shortest_plan_len`` and
    ``schema_fingerprint``.  In particular a blocked action is
    *indistinguishable* from an action that fired and left the state unchanged:
    no rendering here ever names the precondition that failed, so the only way
    to learn the schema is to intervene and watch.

Bit vectors render most-significant flag first, ``f(k-1) ... f0``.
"""
from __future__ import annotations

from alienbody.workflow.env import Action, WorkflowInstance

# Fields of an instance that may never reach the agent.
HIDDEN_FIELDS = ("instance_id", "difficulty", "shortest_plan_len", "schema_fingerprint")
# Fields of an instance that the observation may expose.
PUBLIC_FIELDS = ("k", "m", "start", "goal_on", "goal_off", "actions")


# ── Labels ─────────────────────────────────────────────────────────

def flag_label(flag: int) -> str:
    """Display name of flag ``f_i``."""
    return f"f{flag}"


def tool_label(action_id: int) -> str:
    """Display name of tool ``i``."""
    return f"tool_{action_id}"


# ── State rendering ────────────────────────────────────────────────

def state_bits(bit_state: int, k: int) -> tuple[int, ...]:
    """Flag values of ``bit_state``, most-significant flag first."""
    return tuple((bit_state >> flag) & 1 for flag in range(k - 1, -1, -1))


def format_state(bit_state: int, k: int) -> str:
    """Canonical bit string, e.g. ``"10110010"`` for ``k=8``."""
    return "".join(str(bit) for bit in state_bits(bit_state, k))


def format_state_literals(bit_state: int, k: int) -> str:
    """Every flag as an explicit literal, e.g. ``"f7=1 f6=0 f5=1"``."""
    return " ".join(f"{flag_label(flag)}={bit}" for flag, bit in zip(range(k - 1, -1, -1),
                                                                    state_bits(bit_state, k)))


def format_goal(goal_on: int, goal_off: int, k: int) -> str:
    """The goal's partial assignment, most-significant flag first."""
    literals = [
        f"{flag_label(flag)}={1 if (goal_on >> flag) & 1 else 0}"
        for flag in range(k - 1, -1, -1)
        if ((goal_on | goal_off) >> flag) & 1
    ]
    return " ".join(literals)


# ── Tool rendering ─────────────────────────────────────────────────

def format_action(action: Action, k: int) -> str:
    """One tool schema as ``if ... then ...``, e.g. ``"if f3=1 & f1=0 -> f5:=1"``."""
    pre_on, pre_off, set_on, set_off = action
    conditions = [f"{flag_label(flag)}=1" for flag in range(k - 1, -1, -1) if (pre_on >> flag) & 1]
    conditions += [f"{flag_label(flag)}=0" for flag in range(k - 1, -1, -1) if (pre_off >> flag) & 1]
    assignments = [f"{flag_label(flag)}:=1" for flag in range(k - 1, -1, -1) if (set_on >> flag) & 1]
    assignments += [f"{flag_label(flag)}:=0" for flag in range(k - 1, -1, -1) if (set_off >> flag) & 1]
    guard = " & ".join(conditions) if conditions else "always"
    return f"if {guard} -> {', '.join(assignments)}"


def format_tools(instance: WorkflowInstance) -> str:
    """All tool schemas, one per line, in action-id order."""
    return "\n".join(
        f"{tool_label(action_id)}: {format_action(action, instance.k)}"
        for action_id, action in enumerate(instance.actions)
    )


# ── Observations ───────────────────────────────────────────────────

def step_observation(bit_state: int, k: int) -> dict:
    """Observation handed back after a step: the new state, and nothing else.

    Both renderings in the returned dict are functions of ``bit_state`` alone,
    so a blocked action and a no-op action are literally the same observation.
    """
    return {"state": state_bits(bit_state, k), "state_text": format_state(bit_state, k)}


def render_observation(instance: WorkflowInstance, bit_state: int) -> str:
    """Canonical public observation text for one state of an instance."""
    return "\n".join([
        f"state: {format_state(bit_state, instance.k)}",
        f"flags: {format_state_literals(bit_state, instance.k)}",
        f"goal: {format_goal(instance.goal_on, instance.goal_off, instance.k)}",
        "tools:",
        format_tools(instance),
    ])


# ── Public / hidden boundary ───────────────────────────────────────

def public_instance_view(instance: WorkflowInstance) -> dict:
    """Instance fields an agent may see (schema + goal, no grading labels)."""
    return {
        "k": instance.k,
        "m": instance.m,
        "start": instance.start,
        "goal_on": instance.goal_on,
        "goal_off": instance.goal_off,
        "actions": [list(action) for action in instance.actions],
        "tools": [format_action(action, instance.k) for action in instance.actions],
    }


def hidden_instance_view(instance: WorkflowInstance) -> dict:
    """Grading-only instance fields (never part of an observation)."""
    return {
        "instance_id": instance.instance_id,
        "difficulty": instance.difficulty,
        "shortest_plan_len": instance.shortest_plan_len,
        "schema_fingerprint": instance.schema_fingerprint,
    }


def assert_public(view: dict) -> dict:
    """Return ``view`` unchanged, raising if it leaks a hidden field."""
    leaked = [key for key in HIDDEN_FIELDS if key in view]
    if leaked:
        raise ValueError(f"hidden field(s) in public view: {', '.join(leaked)}")
    return view
