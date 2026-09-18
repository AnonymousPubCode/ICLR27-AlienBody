"""Action types for AlienBody.

Each action type is a pure function: (GridState, EnvConfig, action_idx) -> GridState.
Inspired by NovelGridworlds' precondition/effect pattern, but simpler.

Action Types (from paper):
  Type A: Rotation-based movement (egocentric spatial reasoning)
  Type B: Remapped cardinal movement (motor adaptation)
  Type C: Relational movement (object-dependent effects)
  Type D: State-modifying actions (conditional reasoning)
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

from alienbody.env.grid import (
    GridState, EnvConfig, Position,
    DIRECTION_DELTAS, DIRECTION_NAMES, Phase,
)


class ActionType(str, Enum):
    A = "A"  # Rotation-based
    B = "B"  # Remapped cardinal
    C = "C"  # Relational
    D = "D"  # State-modifying
    E = "E"  # Compositional (two-step combo)
    F = "F"  # Temporal (prev-action dependent)


# ── Type B: Remapped Cardinal ──────────────────────────────────────
# Simplest type. Each action = one of {up, down, left, right}, randomly mapped.
# Action mapping example: ("right", "up", "left", "down") means
#   Action 0 → move right, Action 1 → move up, etc.

TYPE_B_EFFECTS = {
    "up":    (-1, 0),
    "down":  (1, 0),
    "left":  (0, -1),
    "right": (0, 1),
}


def apply_type_b(state: GridState, config: EnvConfig, action_idx: int) -> GridState:
    """Type B: remapped cardinal directions."""
    effect_name = config.action_mapping[action_idx]
    delta = TYPE_B_EFFECTS[effect_name]
    new_pos = state.agent_pos + delta
    obs = _get_obstacles(config)
    if _in_bounds(new_pos, config.grid_size, obs):
        state.agent_pos = new_pos
    return state


# ── Type A: Rotation-Based ─────────────────────────────────────────
# Agent has a facing direction. Actions: rotate_cw, forward, rotate_ccw, backward.

TYPE_A_EFFECTS = ["rotate_cw", "rotate_ccw", "forward", "backward"]  # list for stable ordering


def apply_type_a(state: GridState, config: EnvConfig, action_idx: int) -> GridState:
    """Type A: rotation-based movement."""
    effect_name = config.action_mapping[action_idx]
    obs = _get_obstacles(config)

    if effect_name == "rotate_cw":
        state.agent_dir = (state.agent_dir + 1) % 4
    elif effect_name == "rotate_ccw":
        state.agent_dir = (state.agent_dir - 1) % 4
    elif effect_name == "forward":
        delta = DIRECTION_DELTAS[state.agent_dir]
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
    elif effect_name == "backward":
        backward_dir = (state.agent_dir + 2) % 4
        delta = DIRECTION_DELTAS[backward_dir]
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
    return state


# ── Type C: Relational Movement ───────────────────────────────────
# Actions depend on spatial relationships with colored cells in the grid.

TYPE_C_EFFECTS = ["move_to_nearest_diff_color", "move_to_nearest_same_color",
                  "move_toward_brightest", "flee_same_color",
                  "move_to_farthest_same_color", "move_away_from_brightest",
                  # Tier-XL extensions (n=12)
                  "move_to_farthest_diff_color", "flee_nearest_diff_color",
                  "move_toward_darkest", "move_away_from_darkest",
                  "move_to_nearest_brighter", "move_to_nearest_darker"]


def apply_type_c(state: GridState, config: EnvConfig, action_idx: int) -> GridState:
    """Type C: relational movement based on colored cells."""
    effect_name = config.action_mapping[action_idx]
    agent_cell_color = config.cell_colors[state.agent_pos.row][state.agent_pos.col]
    obs = _get_obstacles(config)

    if effect_name == "move_to_nearest_diff_color":
        target = _find_nearest_by_color(state.agent_pos, config, same=False,
                                         ref_color=agent_cell_color)
        if target:
            state.agent_pos = target

    elif effect_name == "move_to_nearest_same_color":
        target = _find_nearest_by_color(state.agent_pos, config, same=True,
                                         ref_color=agent_cell_color)
        if target:
            state.agent_pos = target

    elif effect_name == "move_to_farthest_same_color":
        # Teleport to the FARTHEST cell (Manhattan) of the same color
        best_pos = None
        best_dist = -1
        for r in range(config.grid_size):
            for c in range(config.grid_size):
                if r == state.agent_pos.row and c == state.agent_pos.col:
                    continue
                if config.cell_colors[r][c] == agent_cell_color:
                    dist = abs(r - state.agent_pos.row) + abs(c - state.agent_pos.col)
                    if dist > best_dist:
                        best_dist = dist
                        best_pos = Position(r, c)
        if best_pos:
            state.agent_pos = best_pos

    elif effect_name == "move_toward_brightest":
        # Move 1 step toward 4-connected neighbor with highest color value
        best_pos = None
        best_color = -1
        for d in range(4):
            delta = DIRECTION_DELTAS[d]
            nb = state.agent_pos + delta
            if _in_bounds(nb, config.grid_size, obs):
                nc = config.cell_colors[nb.row][nb.col]
                if nc > best_color:
                    best_color = nc
                    best_pos = nb
        if best_pos and best_color > 0:
            state.agent_pos = best_pos

    elif effect_name == "move_away_from_brightest":
        # Move 1 step AWAY from the brightest 4-connected neighbor
        best_pos = None
        best_color = -1
        for d in range(4):
            delta = DIRECTION_DELTAS[d]
            nb = state.agent_pos + delta
            if _in_bounds(nb, config.grid_size, obs):
                nc = config.cell_colors[nb.row][nb.col]
                if nc > best_color:
                    best_color = nc
                    best_pos = nb
        if best_pos and best_color > 0:
            dr = state.agent_pos.row - best_pos.row
            dc = state.agent_pos.col - best_pos.col
            new_pos = state.agent_pos + (dr, dc)
            if _in_bounds(new_pos, config.grid_size, obs):
                state.agent_pos = new_pos

    elif effect_name == "flee_same_color":
        # Move 1 step AWAY from nearest same-color cell
        nearest = _find_nearest_by_color(state.agent_pos, config, same=True,
                                          ref_color=agent_cell_color)
        if nearest:
            # Direction away from nearest
            dr = state.agent_pos.row - nearest.row
            dc = state.agent_pos.col - nearest.col
            # Normalize to unit step
            if abs(dr) >= abs(dc):
                step = (1 if dr > 0 else -1, 0)
            else:
                step = (0, 1 if dc > 0 else -1)
            new_pos = state.agent_pos + step
            if _in_bounds(new_pos, config.grid_size, _get_obstacles(config)):
                state.agent_pos = new_pos

    elif effect_name == "move_to_farthest_diff_color":
        # Teleport to the FARTHEST cell (Manhattan) of a different color
        best_pos = None
        best_dist = -1
        for r in range(config.grid_size):
            for c in range(config.grid_size):
                if r == state.agent_pos.row and c == state.agent_pos.col:
                    continue
                if config.cell_colors[r][c] != agent_cell_color:
                    dist = abs(r - state.agent_pos.row) + abs(c - state.agent_pos.col)
                    if dist > best_dist:
                        best_dist = dist
                        best_pos = Position(r, c)
        if best_pos:
            state.agent_pos = best_pos

    elif effect_name == "flee_nearest_diff_color":
        # Move 1 step AWAY from nearest different-color cell
        nearest = _find_nearest_by_color(state.agent_pos, config, same=False,
                                          ref_color=agent_cell_color)
        if nearest:
            dr = state.agent_pos.row - nearest.row
            dc = state.agent_pos.col - nearest.col
            if abs(dr) >= abs(dc):
                step = (1 if dr > 0 else -1, 0)
            else:
                step = (0, 1 if dc > 0 else -1)
            new_pos = state.agent_pos + step
            if _in_bounds(new_pos, config.grid_size, _get_obstacles(config)):
                state.agent_pos = new_pos

    elif effect_name == "move_toward_darkest":
        # Move 1 step toward the 4-connected neighbor with LOWEST color value
        best_pos = None
        best_color = float("inf")
        for d in range(4):
            delta = DIRECTION_DELTAS[d]
            nb = state.agent_pos + delta
            if _in_bounds(nb, config.grid_size, obs):
                nc = config.cell_colors[nb.row][nb.col]
                if nc < best_color:
                    best_color = nc
                    best_pos = nb
        if best_pos is not None:
            state.agent_pos = best_pos

    elif effect_name == "move_away_from_darkest":
        # Move 1 step AWAY from the darkest 4-connected neighbor
        best_pos = None
        best_color = float("inf")
        for d in range(4):
            delta = DIRECTION_DELTAS[d]
            nb = state.agent_pos + delta
            if _in_bounds(nb, config.grid_size, obs):
                nc = config.cell_colors[nb.row][nb.col]
                if nc < best_color:
                    best_color = nc
                    best_pos = nb
        if best_pos is not None:
            dr = state.agent_pos.row - best_pos.row
            dc = state.agent_pos.col - best_pos.col
            new_pos = state.agent_pos + (dr, dc)
            if _in_bounds(new_pos, config.grid_size, obs):
                state.agent_pos = new_pos

    elif effect_name == "move_to_nearest_brighter":
        # Teleport to nearest cell whose color is HIGHER than current cell's
        best_pos = None
        best_dist = float("inf")
        for r in range(config.grid_size):
            for c in range(config.grid_size):
                if r == state.agent_pos.row and c == state.agent_pos.col:
                    continue
                if config.cell_colors[r][c] > agent_cell_color:
                    dist = abs(r - state.agent_pos.row) + abs(c - state.agent_pos.col)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = Position(r, c)
        if best_pos:
            state.agent_pos = best_pos

    elif effect_name == "move_to_nearest_darker":
        # Teleport to nearest cell whose color is LOWER than current cell's
        best_pos = None
        best_dist = float("inf")
        for r in range(config.grid_size):
            for c in range(config.grid_size):
                if r == state.agent_pos.row and c == state.agent_pos.col:
                    continue
                if config.cell_colors[r][c] < agent_cell_color:
                    dist = abs(r - state.agent_pos.row) + abs(c - state.agent_pos.col)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = Position(r, c)
        if best_pos:
            state.agent_pos = best_pos

    return state


# ── Type D: State-Modifying ────────────────────────────────────────
# Actions modify agent's internal state (color) and movement depends on state.

TYPE_D_EFFECTS = ["color_inc", "color_dec", "color_move", "color_interact"]  # list for stable ordering

# Color-to-direction mapping for "color_move" (4 colors → 4 directions)
COLOR_DIR_MAP = {0: 0, 1: 1, 2: 2, 3: 3}  # color 0→up, 1→right, 2→down, 3→left
N_COLORS_TYPE_D = 4  # Only use 4 colors for Type D (learnable in 20 steps)


def apply_type_d(state: GridState, config: EnvConfig, action_idx: int) -> GridState:
    """Type D: state-modifying actions."""
    effect_name = config.action_mapping[action_idx]

    if effect_name == "color_inc":
        state.agent_color = (state.agent_color + 1) % N_COLORS_TYPE_D

    elif effect_name == "color_dec":
        state.agent_color = (state.agent_color - 1) % N_COLORS_TYPE_D

    elif effect_name == "color_move":
        # Direction determined by current agent color
        move_dir = COLOR_DIR_MAP.get(state.agent_color, 0)
        delta = DIRECTION_DELTAS[move_dir]
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, _get_obstacles(config)):
            state.agent_pos = new_pos

    elif effect_name == "color_interact":
        # If agent color matches cell color → move forward; else → rotate CW
        cell_color = config.cell_colors[state.agent_pos.row][state.agent_pos.col]
        if state.agent_color == (cell_color % N_COLORS_TYPE_D):
            delta = DIRECTION_DELTAS[state.agent_dir]
            new_pos = state.agent_pos + delta
            if _in_bounds(new_pos, config.grid_size, _get_obstacles(config)):
                state.agent_pos = new_pos
        else:
            state.agent_dir = (state.agent_dir + 1) % 4

    return state


# ── Type E: Compositional (two-primitive combos) ─────────────────
# Each action is a composition of two Type A primitives chained together.
# E.g., "move_and_rotate_cw" = forward first, then rotate_cw.

TYPE_E_EFFECTS = ["move_and_rotate_cw", "rotate_and_strafe",
                  "double_forward", "retreat_and_spin"]

# Each effect is a pair of (primitive1, primitive2) applied sequentially
TYPE_E_COMBOS: dict[str, tuple[str, str]] = {
    "move_and_rotate_cw": ("forward", "rotate_cw"),
    "rotate_and_strafe":  ("rotate_ccw", "forward"),
    "double_forward":     ("forward", "forward"),
    "retreat_and_spin":   ("backward", "rotate_cw"),  # rotate_cw applied twice below
}


def _apply_type_a_primitive(state: GridState, config: EnvConfig, effect_name: str) -> GridState:
    """Apply a single Type A primitive effect (internal helper for Type E)."""
    obs = _get_obstacles(config)
    if effect_name == "rotate_cw":
        state.agent_dir = (state.agent_dir + 1) % 4
    elif effect_name == "rotate_ccw":
        state.agent_dir = (state.agent_dir - 1) % 4
    elif effect_name == "forward":
        delta = DIRECTION_DELTAS[state.agent_dir]
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
    elif effect_name == "backward":
        backward_dir = (state.agent_dir + 2) % 4
        delta = DIRECTION_DELTAS[backward_dir]
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
    return state


def apply_type_e(state: GridState, config: EnvConfig, action_idx: int) -> GridState:
    """Type E: compositional actions — each is two Type A primitives chained."""
    effect_name = config.action_mapping[action_idx]
    if effect_name == "retreat_and_spin":
        # Special case: backward + rotate 180° (two rotate_cw)
        state = _apply_type_a_primitive(state, config, "backward")
        state = _apply_type_a_primitive(state, config, "rotate_cw")
        state = _apply_type_a_primitive(state, config, "rotate_cw")
    elif effect_name in TYPE_E_COMBOS:
        p1, p2 = TYPE_E_COMBOS[effect_name]
        state = _apply_type_a_primitive(state, config, p1)
        state = _apply_type_a_primitive(state, config, p2)
    return state


# ── Type F: Temporal (prev-action dependent) ─────────────────────
# Action effects depend on which action was taken previously.
# Uses GridState.prev_action to track history.

TYPE_F_EFFECTS = ["temporal_0", "temporal_1", "temporal_2", "temporal_3"]

# Mapping: (action_name, same_as_prev) → direction index
# "same_as_prev" means the action idx equals state.prev_action
TYPE_F_SAME_DIR = {    # when action == prev_action
    "temporal_0": 0,   # north
    "temporal_1": 2,   # south
    "temporal_2": -1,  # repeat (handled specially)
    "temporal_3": -2,  # inverse (handled specially)
}
TYPE_F_DIFF_DIR = {    # when action != prev_action
    "temporal_0": 1,   # east
    "temporal_1": 3,   # west
    "temporal_2": -1,  # repeat
    "temporal_3": -2,  # inverse
}


def apply_type_f(state: GridState, config: EnvConfig, action_idx: int) -> GridState:
    """Type F: temporal actions — effects depend on previous action.

    The last positional delta is stored on the state (``state.last_move_delta``)
    so that ``repeat``/``inverse`` remain correct across cloning, replay and
    garbage collection. (Previously a module-level dict keyed on ``id(state)``
    was used, which silently broke whenever a state was cloned or its id reused.)
    """
    effect_name = config.action_mapping[action_idx]
    obs = _get_obstacles(config)
    is_same = (action_idx == state.prev_action)

    if effect_name == "temporal_2":
        # Repeat: apply the same movement as prev action did
        if state.prev_action >= 0 and state.last_move_delta is not None:
            delta = state.last_move_delta
            new_pos = state.agent_pos + delta
            if _in_bounds(new_pos, config.grid_size, obs):
                state.agent_pos = new_pos
        # else: no-op (first action or no prior movement)
        state.prev_action = action_idx
        return state

    elif effect_name == "temporal_3":
        # Inverse: apply opposite of prev action's movement
        if state.prev_action >= 0 and state.last_move_delta is not None:
            prev_d = state.last_move_delta
            delta = (-prev_d[0], -prev_d[1])
            new_pos = state.agent_pos + delta
            if _in_bounds(new_pos, config.grid_size, obs):
                state.agent_pos = new_pos
        state.prev_action = action_idx
        return state

    # temporal_0 or temporal_1: direction depends on same/diff
    if is_same:
        dir_idx = TYPE_F_SAME_DIR[effect_name]
    else:
        dir_idx = TYPE_F_DIFF_DIR[effect_name]

    if dir_idx >= 0:
        delta = DIRECTION_DELTAS[dir_idx]
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
        state.last_move_delta = delta
    state.prev_action = action_idx
    return state


# ── Action Dispatch ────────────────────────────────────────────────

ACTION_HANDLERS: dict[str, Callable] = {
    "A": apply_type_a,
    "B": apply_type_b,
    "C": apply_type_c,
    "D": apply_type_d,
    "E": apply_type_e,
    "F": apply_type_f,
}


def apply_action(state: GridState, config: EnvConfig, action_idx: int) -> GridState:
    """Main dispatch: apply action to state based on config's action_type.

    For combo types (e.g., "A+D"), the first 2 actions use type A
    and the last 2 use type D.
    """
    if action_idx < 0 or action_idx >= config.n_actions:
        raise ValueError(f"Invalid action {action_idx}, must be 0-{config.n_actions-1}")

    action_type = config.action_type
    if "+" in action_type:
        # Combo: e.g. "A+D" → first half Type A, second half Type D
        types = action_type.split("+")
        split = config.n_actions // len(types)
        type_idx = min(action_idx // split, len(types) - 1)
        handler = ACTION_HANDLERS[types[type_idx]]
    else:
        handler = ACTION_HANDLERS[action_type]

    return handler(state, config, action_idx)


# ── Helpers ────────────────────────────────────────────────────────

def _get_obstacles(config: EnvConfig) -> set:
    """Extract obstacle set from config (cached-friendly)."""
    if hasattr(config, 'obstacles') and config.obstacles:
        return {(r, c) for r, c in config.obstacles}
    return set()


def _in_bounds(pos: Position, grid_size: int, obstacles: set | None = None) -> bool:
    if not (0 <= pos.row < grid_size and 0 <= pos.col < grid_size):
        return False
    if obstacles and (pos.row, pos.col) in obstacles:
        return False
    return True


def _find_nearest_by_color(
    from_pos: Position,
    config: EnvConfig,
    same: bool,
    ref_color: int,
) -> Position | None:
    """Find nearest cell with same/different color (Manhattan distance, BFS order)."""
    best_pos = None
    best_dist = float("inf")
    for r in range(config.grid_size):
        for c in range(config.grid_size):
            if r == from_pos.row and c == from_pos.col:
                continue
            cell_color = config.cell_colors[r][c]
            color_match = (cell_color == ref_color) if same else (cell_color != ref_color)
            if color_match:
                dist = abs(r - from_pos.row) + abs(c - from_pos.col)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = Position(r, c)
    return best_pos


def get_all_effect_names(action_type: str) -> set[str]:
    """Get all possible effect names for a given action type."""
    mapping = {
        "A": set(TYPE_A_EFFECTS),
        "B": set(TYPE_B_EFFECTS.keys()),
        "C": set(TYPE_C_EFFECTS),
        "D": set(TYPE_D_EFFECTS),
        "E": set(TYPE_E_EFFECTS),
        "F": set(TYPE_F_EFFECTS),
    }
    if "+" in action_type:
        result = set()
        for t in action_type.split("+"):
            result |= mapping[t]
        return result
    return mapping[action_type]
