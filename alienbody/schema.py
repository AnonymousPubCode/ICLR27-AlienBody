"""FMB Schema: structured action-effect representation + simulator.

Design (b2): JSON schema with effect_type + params.
The simulator interprets the schema to predict next state — this is the
executable forward model T-hat that the planner uses.

Effect types cover all 6 AlienBody action families:
  translate  — cardinal movement (F1, part of F2-F3)
  rotate     — direction change (F2)
  state_change — color modification (F3)
  relational — color-relative movement (F4)
  composite  — two primitives chained (F5)
  temporal   — prev-action dependent (F6)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from enum import StrEnum

from alienbody.env.grid import (
    GridState, EnvConfig, Position,
    DIRECTION_DELTAS, DIRECTION_NAMES,
)


class EffectType(StrEnum):
    TRANSLATE = "translate"
    ROTATE = "rotate"
    STATE_CHANGE = "state_change"
    RELATIONAL = "relational"
    COMPOSITE = "composite"
    TEMPORAL = "temporal"
    NOOP = "noop"


# ── Schema Definition ──────────────────────────────────────────────

@dataclass
class ActionSchema:
    """Structured description of what one action does."""
    effect_type: EffectType
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"effect_type": str(self.effect_type), "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSchema":
        return cls(
            effect_type=EffectType(d["effect_type"]),
            params=d.get("params", {}),
        )


@dataclass
class ForwardModel:
    """Complete executable forward model: {action_id: ActionSchema}."""
    actions: dict[int, ActionSchema] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {str(k): v.to_dict() for k, v in self.actions.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ForwardModel":
        return cls(actions={int(k): ActionSchema.from_dict(v) for k, v in d.items()})

    def is_complete(self, n_actions: int) -> bool:
        return len(self.actions) == n_actions and all(
            a.effect_type != EffectType.NOOP for a in self.actions.values()
        )


# ── Simulator ──────────────────────────────────────────────────────

class ForwardModelSimulator:
    """Simulate state transitions using a ForwardModel (T-hat).

    Takes a GridState + action → predicts next GridState.
    Used by the BFS planner instead of the true transition function.
    """

    def __init__(self, model: ForwardModel, config: EnvConfig):
        self.model = model
        self.config = config
        self._obstacles = self._get_obstacles()

    def _get_obstacles(self) -> set:
        if hasattr(self.config, 'obstacles') and self.config.obstacles:
            return {(r, c) for r, c in self.config.obstacles}
        return set()

    def predict(self, state: GridState, action: int) -> GridState:
        """Predict next state given current state and action."""
        if action not in self.model.actions:
            return state  # unknown action → no change

        schema = self.model.actions[action]
        handler = self._HANDLERS.get(schema.effect_type)
        if handler:
            self._cur_action = action  # temporal same/diff needs current action
            return handler(self, state, schema.params)
        return state

    # ── Effect Handlers ──────────────────────────────────────────

    def _handle_translate(self, state: GridState, params: dict) -> GridState:
        """Cardinal direction movement."""
        direction = params.get("direction", "up")
        if direction in ("forward", "backward"):
            # Ego-centric: relative to facing direction
            dir_map = {"forward": 0, "right": 1, "backward": 2, "left": 3}
            facing = state.agent_dir
            if direction == "forward":
                delta = DIRECTION_DELTAS[facing]
            elif direction == "backward":
                delta = DIRECTION_DELTAS[(facing + 2) % 4]
            else:
                delta = DIRECTION_DELTAS[dir_map.get(direction, facing)]
        else:
            # Absolute direction: up/down/left/right
            abs_map = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
            delta = abs_map.get(direction, (0, 0))

        new_pos = state.agent_pos + delta
        if self._in_bounds(new_pos):
            state.agent_pos = new_pos
        return state

    def _handle_rotate(self, state: GridState, params: dict) -> GridState:
        """Rotation: cw or ccw."""
        rotation = params.get("rotation", "cw")
        if rotation == "cw":
            state.agent_dir = (state.agent_dir + 1) % 4
        elif rotation == "ccw":
            state.agent_dir = (state.agent_dir - 1) % 4
        return state

    def _handle_state_change(self, state: GridState, params: dict) -> GridState:
        """Color modification."""
        change = params.get("change", "inc")
        if change == "inc":
            state.agent_color = (state.agent_color + 1) % 4
        elif change == "dec":
            state.agent_color = (state.agent_color - 1) % 4
        elif change == "color_move":
            # Direction determined by color
            color_dir = {0: 0, 1: 1, 2: 2, 3: 3}  # color→direction
            move_dir = color_dir.get(state.agent_color, 0)
            delta = DIRECTION_DELTAS[move_dir]
            new_pos = state.agent_pos + delta
            if self._in_bounds(new_pos):
                state.agent_pos = new_pos
        elif change == "color_interact":
            # If color matches cell → forward; else rotate CW
            cell_color = self.config.cell_colors[state.agent_pos.row][state.agent_pos.col]
            if state.agent_color == (cell_color % 4):
                delta = DIRECTION_DELTAS[state.agent_dir]
                new_pos = state.agent_pos + delta
                if self._in_bounds(new_pos):
                    state.agent_pos = new_pos
            else:
                state.agent_dir = (state.agent_dir + 1) % 4
        return state

    def _handle_relational(self, state: GridState, params: dict) -> GridState:
        """Movement relative to colored cells."""
        relation = params.get("relation", "nearest_diff")
        pos = state.agent_pos
        # Use CELL color at agent position (not agent's internal color state).
        # This matches apply_type_c's behavior: F4 relations depend on the
        # grid cell color, not the agent's mutable color attribute.
        agent_cell_color = self.config.cell_colors[pos.row][pos.col]

        if relation == "nearest_diff":
            target = self._find_nearest(pos, same=False, ref_color=agent_cell_color)
            if target:
                state.agent_pos = target
        elif relation == "nearest_same":
            target = self._find_nearest(pos, same=True, ref_color=agent_cell_color)
            if target:
                state.agent_pos = target
        elif relation == "toward_brightest":
            best_pos, best_color = None, -1
            for d in range(4):
                delta = DIRECTION_DELTAS[d]
                nb = pos + delta
                if self._in_bounds(nb):
                    nc = self.config.cell_colors[nb.row][nb.col]
                    if nc > best_color:
                        best_color = nc
                        best_pos = nb
            if best_pos and best_color > 0:
                state.agent_pos = best_pos
        elif relation == "flee_same":
            nearest = self._find_nearest(pos, same=True, ref_color=agent_cell_color)
            if nearest:
                dr = pos.row - nearest.row
                dc = pos.col - nearest.col
                step = (1 if dr > 0 else -1, 0) if abs(dr) >= abs(dc) else (0, 1 if dc > 0 else -1)
                new_pos = pos + step
                if self._in_bounds(new_pos):
                    state.agent_pos = new_pos
        elif relation == "farthest_diff":
            best_pos, best_dist = None, -1
            for r in range(self.config.grid_size):
                for c in range(self.config.grid_size):
                    if (r, c) == (pos.row, pos.col):
                        continue
                    if self.config.cell_colors[r][c] != agent_cell_color:
                        dist = abs(r - pos.row) + abs(c - pos.col)
                        if dist > best_dist:
                            best_dist = dist
                            best_pos = Position(r, c)
            if best_pos:
                state.agent_pos = best_pos
        elif relation == "flee_nearest_diff":
            nearest = self._find_nearest(pos, same=False, ref_color=agent_cell_color)
            if nearest:
                dr = pos.row - nearest.row
                dc = pos.col - nearest.col
                step = (1 if dr > 0 else -1, 0) if abs(dr) >= abs(dc) else (0, 1 if dc > 0 else -1)
                new_pos = pos + step
                if self._in_bounds(new_pos):
                    state.agent_pos = new_pos
        elif relation == "toward_darkest":
            best_pos, best_color = None, float("inf")
            for d in range(4):
                delta = DIRECTION_DELTAS[d]
                nb = pos + delta
                if self._in_bounds(nb):
                    nc = self.config.cell_colors[nb.row][nb.col]
                    if nc < best_color:
                        best_color = nc
                        best_pos = nb
            if best_pos is not None:
                state.agent_pos = best_pos
        elif relation == "away_darkest":
            best_pos, best_color = None, float("inf")
            for d in range(4):
                delta = DIRECTION_DELTAS[d]
                nb = pos + delta
                if self._in_bounds(nb):
                    nc = self.config.cell_colors[nb.row][nb.col]
                    if nc < best_color:
                        best_color = nc
                        best_pos = nb
            if best_pos is not None:
                dr = pos.row - best_pos.row
                dc = pos.col - best_pos.col
                new_pos = pos + (dr, dc)
                if self._in_bounds(new_pos):
                    state.agent_pos = new_pos
        elif relation == "nearest_brighter":
            best_pos, best_dist = None, float("inf")
            for r in range(self.config.grid_size):
                for c in range(self.config.grid_size):
                    if (r, c) == (pos.row, pos.col):
                        continue
                    if self.config.cell_colors[r][c] > agent_cell_color:
                        dist = abs(r - pos.row) + abs(c - pos.col)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = Position(r, c)
            if best_pos:
                state.agent_pos = best_pos
        elif relation == "nearest_darker":
            best_pos, best_dist = None, float("inf")
            for r in range(self.config.grid_size):
                for c in range(self.config.grid_size):
                    if (r, c) == (pos.row, pos.col):
                        continue
                    if self.config.cell_colors[r][c] < agent_cell_color:
                        dist = abs(r - pos.row) + abs(c - pos.col)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = Position(r, c)
            if best_pos:
                state.agent_pos = best_pos
        return state

    def _handle_composite(self, state: GridState, params: dict) -> GridState:
        """Two primitives chained."""
        primitives = params.get("primitives", [])
        for prim in primitives:
            ptype = prim.get("type", "translate")
            pparams = prim.get("params", {})
            handler = self._HANDLERS.get(EffectType(ptype))
            if handler:
                state = handler(self, state, pparams)
        return state

    def _handle_temporal(self, state: GridState, params: dict) -> GridState:
        """Prev-action dependent: use last_move_delta (mirrors apply_type_f)."""
        is_same = (self._cur_action == state.prev_action)
        same_dir = params.get("same_dir") or 0  # handle None
        diff_dir = params.get("diff_dir") or 0
        repeat = params.get("repeat", False)
        inverse = params.get("inverse", False)

        if repeat and state.prev_action >= 0 and state.last_move_delta is not None:
            new_pos = state.agent_pos + state.last_move_delta
            if self._in_bounds(new_pos):
                state.agent_pos = new_pos
        elif inverse and state.prev_action >= 0 and state.last_move_delta is not None:
            delta = (-state.last_move_delta[0], -state.last_move_delta[1])
            new_pos = state.agent_pos + delta
            if self._in_bounds(new_pos):
                state.agent_pos = new_pos
        else:
            dir_idx = same_dir if is_same else diff_dir
            if dir_idx >= 0:
                delta = DIRECTION_DELTAS.get(dir_idx, (0, 0))
                new_pos = state.agent_pos + delta
                if self._in_bounds(new_pos):
                    state.agent_pos = new_pos
                state.last_move_delta = delta
        return state

    # ── Helpers ──────────────────────────────────────────────────

    def _in_bounds(self, pos: Position) -> bool:
        if not (0 <= pos.row < self.config.grid_size and 0 <= pos.col < self.config.grid_size):
            return False
        if (pos.row, pos.col) in self._obstacles:
            return False
        return True

    def _find_nearest(self, from_pos: Position, same: bool, ref_color: int) -> Optional[Position]:
        best_pos, best_dist = None, float("inf")
        for r in range(self.config.grid_size):
            for c in range(self.config.grid_size):
                if r == from_pos.row and c == from_pos.col:
                    continue
                cell_color = self.config.cell_colors[r][c]
                match = (cell_color == ref_color) if same else (cell_color != ref_color)
                if match:
                    dist = abs(r - from_pos.row) + abs(c - from_pos.col)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = Position(r, c)
        return best_pos

    _HANDLERS = {
        EffectType.TRANSLATE: _handle_translate,
        EffectType.ROTATE: _handle_rotate,
        EffectType.STATE_CHANGE: _handle_state_change,
        EffectType.RELATIONAL: _handle_relational,
        EffectType.COMPOSITE: _handle_composite,
        EffectType.TEMPORAL: _handle_temporal,
    }
