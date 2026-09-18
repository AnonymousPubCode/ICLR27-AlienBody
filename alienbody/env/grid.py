"""Immutable grid state and environment configuration.

Design: Craftax-style immutable dataclasses for reproducibility.
Serialization: ARC-AGI-style JSON for environment storage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Optional

# ARC-AGI standard 10-color palette
ARC_COLORS = {
    0: (0, 0, 0),        # black
    1: (0, 116, 217),    # blue
    2: (255, 65, 54),    # red
    3: (46, 204, 64),    # green
    4: (255, 220, 0),    # yellow
    5: (170, 170, 170),  # grey
    6: (240, 18, 190),   # magenta
    7: (255, 133, 27),   # orange
    8: (127, 219, 255),  # light blue
    9: (135, 12, 37),    # maroon
}

DIRECTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}
DIRECTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


class Phase(IntEnum):
    CALIBRATION = 1  # Phase 1: free exploration
    EXECUTION = 2    # Phase 2: navigate to target


@dataclass(frozen=True)
class Position:
    row: int
    col: int

    def manhattan_distance(self, other: Position) -> int:
        return abs(self.row - other.row) + abs(self.col - other.col)

    def __add__(self, delta: tuple[int, int]) -> Position:
        return Position(self.row + delta[0], self.col + delta[1])


@dataclass
class GridState:
    """Mutable game state for a single episode.

    Separated from EnvConfig so the same config can be replayed.
    """
    agent_pos: Position
    agent_color: int          # agent's color index (0-9), used by Type D
    agent_dir: int            # facing direction (0=up, 1=right, 2=down, 3=left)
    phase: Phase
    step_count: int           # total steps taken
    phase1_steps: int         # steps taken in Phase 1
    phase2_steps: int         # steps taken in Phase 2
    action_history: list[dict] = field(default_factory=list)
    done: bool = False
    success: bool = False
    prev_action: int = -1  # last action taken (-1 = none yet); used by Type F
    last_move_delta: Optional[tuple[int, int]] = None  # last positional delta; used by Type F repeat/inverse

    def clone(self) -> GridState:
        return GridState(
            agent_pos=self.agent_pos,
            agent_color=self.agent_color,
            agent_dir=self.agent_dir,
            phase=self.phase,
            step_count=self.step_count,
            phase1_steps=self.phase1_steps,
            phase2_steps=self.phase2_steps,
            action_history=list(self.action_history),
            done=self.done,
            success=self.success,
            prev_action=self.prev_action,
            last_move_delta=self.last_move_delta,
        )


@dataclass(frozen=True)
class EnvConfig:
    """Immutable environment specification. JSON-serializable.

    Fully determines an episode: same config + same agent = same trajectory.
    """
    # Identity
    env_id: str                        # e.g. "family1_train_042"
    family: int                        # 1-4
    seed: int

    # Grid layout
    grid_size: int                     # 16 (default)
    cell_colors: tuple[tuple[int, ...], ...]  # grid_size x grid_size color matrix
    agent_start: tuple[int, int]       # (row, col)
    agent_start_color: int             # initial agent color
    agent_start_dir: int               # initial facing direction (0-3)
    target_pos: tuple[int, int]        # (row, col)

    # Action configuration
    action_type: str                   # "A", "B", "C", "D", or combo like "A+D"
    action_mapping: tuple[str, ...]    # ground truth: ("rotate_cw", "forward", ...)
    n_actions: int = 4

    # Budget
    phase1_budget: int = 20
    max_total_steps: int = 50

    # Obstacles (optional)
    obstacles: tuple[tuple[int, int], ...] = ()  # wall positions

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, s: str) -> EnvConfig:
        d = json.loads(s)
        # Convert lists back to tuples for frozen dataclass
        d["cell_colors"] = tuple(tuple(row) for row in d["cell_colors"])
        d["agent_start"] = tuple(d["agent_start"])
        d["target_pos"] = tuple(d["target_pos"])
        d["action_mapping"] = tuple(d["action_mapping"])
        d["obstacles"] = tuple(tuple(o) for o in d.get("obstacles", []))
        return cls(**d)

    @classmethod
    def from_file(cls, path: str) -> EnvConfig:
        with open(path) as f:
            return cls.from_json(f.read())

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    def make_initial_state(self) -> GridState:
        return GridState(
            agent_pos=Position(*self.agent_start),
            agent_color=self.agent_start_color,
            agent_dir=self.agent_start_dir,
            phase=Phase.CALIBRATION,
            step_count=0,
            phase1_steps=0,
            phase2_steps=0,
        )
