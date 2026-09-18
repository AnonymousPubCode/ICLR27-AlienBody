"""Prompt templates for LLM/VLM agents.

4 variants: MINIMAL, COT, EXPLORE_FIRST, EXPERT_DEMO
Supports "familiar" mode where action names are revealed (for Table 3 experiment).
Supports 5-level diagnostic ladder (assist_level 0-4) for calibration gap decomposition.
Includes robust response parser with regex fallback.
"""
from __future__ import annotations

import re
from enum import Enum


class PromptVariant(str, Enum):
    MINIMAL = "minimal"
    COT = "cot"
    EXPLORE_FIRST = "explore_first"
    EXPERT_DEMO = "expert_demo"


# Assist level descriptions for the diagnostic ladder experiment:
# 0 = Normal (no help)
# 1 = Familiar (know action space = directions, not which is which)
# 2 = Summary (clean Phase 1 observation summary injected at Phase 2 start)
# 3 = Mapping Given (skip Phase 1, action mapping provided upfront)
# 4 = Oracle (code-level, not prompt-level — OracleAgent with BFS)


# ── System Prompts ─────────────────────────────────────────────────

_SYSTEM_BASE = """\
You are in a grid world. You can see your position (a white square with a red arrow showing your facing direction) and the environment.

You have {n_actions} available actions: {action_list}.
You do NOT know what these actions do — their effects are unknown and randomized.

The episode has two phases:
- Phase 1 (Calibration): Explore freely to learn what each action does. You can say "DONE" to end exploration early.
- Phase 2 (Execution): A target location (cyan border) will appear. Navigate to it efficiently.

Output ONLY a single number ({action_range}) to choose an action, or "DONE" to end Phase 1."""

_SYSTEM_COT_ADDON = """

Before choosing your action, reason step-by-step:
1. What have I learned about each action so far?
2. Which actions remain untested?
3. What should I do next?

Then output your action on the LAST line as just a number."""

_SYSTEM_EXPLORE_ADDON = """

IMPORTANT: In Phase 1, systematically test each action at least once before repeating any action. A good strategy is to try Action 0, then Action 1, then Action 2, then Action 3, then say "DONE"."""

_SYSTEM_EXPERT_DEMO = """

Here is an example of efficient exploration from a different environment:
  Step 1: Action 0 → position changed from (5,8) to (5,9). Learned: Action 0 moves right.
  Step 2: Action 1 → position changed from (5,9) to (6,9). Learned: Action 1 moves down.
  Step 3: Action 2 → position changed from (6,9) to (6,8). Learned: Action 2 moves left.
  Step 4: Action 3 → position changed from (6,8) to (5,8). Learned: Action 3 moves up.
  → DONE (all 4 actions tested in 4 steps)

Note: the actual action mappings in YOUR environment are DIFFERENT. You must discover them yourself."""

_SYSTEM_FAMILIAR = """

Note: Your actions correspond to standard movement directions (up, down, left, right) but in a SCRAMBLED order. You need to figure out which action maps to which direction."""

# ── Level 3: Mapping Given ────────────────────────────────────────

_SYSTEM_MAPPING_GIVEN = """\
You are in a grid world. You can see your position and the environment.

You have {n_actions} actions with the following effects:
{mapping_description}

A target location is marked on the grid. Navigate to it as efficiently as possible.

Output ONLY a single number ({action_range}) to choose an action."""

_SYSTEM_L3_COT_ADDON = """

Before choosing your action, reason step-by-step:
1. Where am I relative to the target?
2. Which action's effect moves me closer to the target?
3. What is the best action to take now?

Then output your action on the LAST line as just a number."""

# ── Level 2: Summary Assist ───────────────────────────────────────

_PHASE2_SUMMARY_HEADER = """
=== Phase 1 Observation Summary ===
Here is a structured summary of what you observed during exploration:
{summary_table}
=== End Summary ===

Use this information to navigate to the target efficiently."""


def build_system_prompt(
    variant: PromptVariant = PromptVariant.MINIMAL,
    n_actions: int = 4,
    familiar: bool = False,
    assist_level: int = 0,
    action_mapping: list[str] | None = None,
    action_labels: list[str] | None = None,
) -> str:
    """Build the system prompt for an LLM/VLM agent.

    Args:
        variant: Prompt template variant.
        n_actions: Number of available actions.
        familiar: If True, reveal that actions are scrambled directions.
        assist_level: Diagnostic ladder level (0-4).
            0 = normal, 1 = familiar, 2 = summary (injected at Phase 2),
            3 = mapping given (skip Phase 1), 4 = oracle (code-level).
        action_mapping: Ground truth mapping for assist_level=3.
        action_labels: Button labels to reveal (name-prior ablation).
            None = anonymous buttons (default benchmark setting).
            The labels are presented as interface labels, with no claim
            about whether they match the actions' actual effects.
    """
    action_list = ", ".join(f"Action {i}" for i in range(n_actions))
    action_range = f"0-{n_actions - 1}"

    # Level 3: completely different prompt — mapping is given
    if assist_level == 3 and action_mapping is not None:
        mapping_desc = "\n".join(
            f"  Action {i} = {effect}" for i, effect in enumerate(action_mapping)
        )
        prompt = _SYSTEM_MAPPING_GIVEN.format(
            n_actions=n_actions,
            mapping_description=mapping_desc,
            action_range=action_range,
        )
        if variant == PromptVariant.COT:
            prompt += _SYSTEM_L3_COT_ADDON
        return prompt

    # Levels 0, 1, 2: normal base prompt (Level 2 summary injected at turn level)
    prompt = _SYSTEM_BASE.format(
        n_actions=n_actions,
        action_list=action_list,
        action_range=action_range,
    )

    # Name-prior ablation: interface labels on the buttons, inserted BEFORE
    # the "effects unknown" statement — labels are interface information;
    # whether they describe the effects is for the agent to find out.
    if action_labels:
        label_lines = "\n".join(
            f'  Action {i} — "{lbl}"' for i, lbl in enumerate(action_labels)
        )
        prompt = prompt.replace(
            "You do NOT know what these actions do",
            f"The interface labels the buttons as follows:\n{label_lines}\n\n"
            f"You do NOT know what these actions do",
        )

    if variant == PromptVariant.COT:
        prompt += _SYSTEM_COT_ADDON
    elif variant == PromptVariant.EXPLORE_FIRST:
        prompt += _SYSTEM_EXPLORE_ADDON
    elif variant == PromptVariant.EXPERT_DEMO:
        prompt += _SYSTEM_EXPERT_DEMO

    # Level 1: familiar framing (also applies if assist_level=1 explicitly)
    if familiar or assist_level == 1:
        prompt += _SYSTEM_FAMILIAR

    return prompt


def build_phase1_summary(history: list[dict]) -> str:
    """Build a per-action structured summary of Phase 1 observations.

    Groups observations by action and explicitly notes conditional patterns
    (color-dependence, direction-dependence, position-dependence).
    Used for assist_level=2 (Summary Assist). Injected at Phase 2 transition.
    """
    from alienbody.env.grid import DIRECTION_NAMES
    from collections import defaultdict

    # Group observations by action
    action_obs = defaultdict(list)
    for record in history:
        if record.get("phase") != 1:
            continue
        action_obs[record["action"]].append(record)

    lines = ["=== Phase 1 Observation Summary (per action) ==="]
    dir_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    for action in sorted(action_obs.keys()):
        records = action_obs[action]
        lines.append(f"\nAction {action}: tested {len(records)} time(s)")

        effects = []
        colors_seen = set()
        for rec in records:
            parts = []
            conditions = []
            if rec.get("color_changed"):
                conditions.append(f"color={rec['prev_color']}")
                colors_seen.add(rec['prev_color'])
            if rec.get("direction_changed"):
                conditions.append(f"facing {dir_names.get(rec['prev_dir'], rec['prev_dir'])}")
            if rec.get("prev_action", -1) >= 0:
                conditions.append(f"after Action {rec.get('prev_action', '?')}")

            if rec.get("position_changed"):
                pr, pc = rec["prev_pos"]
                nr, nc = rec["new_pos"]
                dr, dc = nr - pr, nc - pc
                delta_to_dir = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}
                direction = delta_to_dir.get((dr, dc), f"({dr:+d},{dc:+d})")
                parts.append(f"moved {direction}")
            if rec.get("direction_changed"):
                prev_d = dir_names.get(rec["prev_dir"], str(rec["prev_dir"]))
                new_d = dir_names.get(rec["new_dir"], str(rec["new_dir"]))
                parts.append(f"rotated {prev_d}->{new_d}")
            if rec.get("color_changed"):
                parts.append(f"color {rec['prev_color']}->{rec['new_color']}")
            if not parts:
                parts.append("no visible effect")

            cond_str = f"[{', '.join(conditions)}] " if conditions else ""
            effects.append(f"  {cond_str}{'; '.join(parts)}")

        lines.extend(effects)

        # Add pattern annotation if effects vary by condition
        if len(colors_seen) > 1:
            lines.append(f"  NOTE: Effect varies by color state. Colors tested: {sorted(colors_seen)}")

    lines.append(f"\n=== End Summary ({len(history)} steps) ===")
    return "\n".join(lines)


# ── Turn Prompts ───────────────────────────────────────────────────

def build_turn_prompt(
    variant: PromptVariant,
    observation: dict,
    phase: int,
    step: int,
    feedback: str | None = None,
    phase2_summary: str | None = None,
) -> str:
    """Build the per-turn user message.

    For VLM: the image is sent separately; this provides text context.
    For LLM: the text grid is included in this prompt.

    Args:
        phase2_summary: If provided (assist_level=2), inject observation summary
            at the first Phase 2 turn.
    """
    parts = []

    # Phase indicator
    if phase == 1:
        parts.append(f"[Phase 1 — Exploration] Step {step}")
    else:
        parts.append(f"[Phase 2 — Navigate to Target] Step {step}")

    # Inject Phase 1 summary at Phase 2 start (assist_level=2)
    if phase2_summary:
        parts.append(phase2_summary)

    # Feedback from last action
    if feedback:
        parts.append(f"Last action result: {feedback}")

    # Text grid (always included for LLM; for VLM, image sent separately)
    if "text" in observation:
        parts.append(f"\n{observation['text']}")

    # Action prompt
    if phase == 1:
        parts.append("\nChoose an action (0-3) to explore, or say DONE to end exploration:")
    else:
        parts.append("\nChoose an action (0-3) to move toward the target:")

    return "\n".join(parts)


# ── Response Parser ────────────────────────────────────────────────

# Patterns to extract action number from LLM response
_ACTION_PATTERNS = [
    re.compile(r"^(\d)$"),                          # just a digit
    re.compile(r"^Action\s*(\d)", re.IGNORECASE),   # "Action 2"
    re.compile(r"action[:\s]*(\d)", re.IGNORECASE),  # "action: 2"
    re.compile(r"choose\s*(\d)", re.IGNORECASE),     # "I choose 2"
    re.compile(r"\b(\d)\b"),                          # any single digit (fallback)
]

_DONE_PATTERNS = [
    re.compile(r"\bdone\b", re.IGNORECASE),
    re.compile(r"\bfinish", re.IGNORECASE),
    re.compile(r"\bstop\s+explor", re.IGNORECASE),
    re.compile(r"\bend\s+phase", re.IGNORECASE),
]


def parse_action_response(response: str, n_actions: int = 4) -> int | str:
    """Parse LLM response into action index or 'done' signal.

    Returns:
        int: action index (0 to n_actions-1)
        str: "done" if agent wants to end Phase 1

    Robust: handles various response formats including reasoning before action.
    """
    response = response.strip()

    # Check for "done" first
    for pattern in _DONE_PATTERNS:
        if pattern.search(response):
            # Make sure it's not "I'm not done yet"
            if not re.search(r"\bnot\b.*\bdone\b", response, re.IGNORECASE):
                return "done"

    # Try to extract action number
    # For CoT responses, check the last line first
    lines = response.strip().split("\n")
    search_order = [lines[-1], response] if len(lines) > 1 else [response]

    for text in search_order:
        for pattern in _ACTION_PATTERNS:
            match = pattern.search(text)
            if match:
                action = int(match.group(1))
                if 0 <= action < n_actions:
                    return action

    # Fallback: return 0 if nothing parsed
    return 0
