"""Open effect DSL for AlienBody program identification (Method v3).

Design goals
------------
1. Express all six CogType families without exposing env-specific effect names.
2. Make blind enumeration expensive (program space >> n!), so a VLM proposer
   can win on sample efficiency when it works.
3. Keep programs executable: each AST node compiles to a pure
   (GridState, EnvConfig) -> GridState transform for verification + BFS.

Grammar (JSON AST)
------------------
  translate(dir in {N,E,S,W,F,B})     # F/B = forward/back w.r.t. facing
  rotate(k in {-1,+1,+2})
  step_toward(sel) / step_away(sel)   # 1-step along 4-neigh selector
  teleport(sel)                       # jump to selected cell
  set_color(delta in {-1,+1})
  color_move()                        # move by agent_color -> dir map
  cond(pred, then, else)
  seq(e1, e2, ...)
  noop()

Selectors ``sel``:
  {"metric": "manhattan"|"color", "agg": "min"|"max",
   "pred": "same"|"diff"|"any"|"brighter"|"darker",
   "scope": "grid"|"neighbors"}

Predicates for cond:
  {"on": "prev_same"|"prev_diff"|"color_match_cell"}
"""
from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from typing import Any, Iterator

from alienbody.env.grid import (
    DIRECTION_DELTAS, EnvConfig, GridState, Position,
)
from alienbody.env.actions import (
    COLOR_DIR_MAP, N_COLORS_TYPE_D, _get_obstacles, _in_bounds,
)

DIRS_CARDINAL = {"N": 0, "E": 1, "S": 2, "W": 3}


@dataclass(frozen=True)
class Prog:
    """Thin wrapper so programs are hashable for caches."""
    ast: tuple  # JSON-serializable nested tuples/strs/ints

    def to_json(self) -> Any:
        return _from_tuple(self.ast)

    @staticmethod
    def from_json(obj: Any) -> "Prog":
        return Prog(_to_tuple(obj))


def _to_tuple(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ("dict", tuple((k, _to_tuple(v)) for k, v in sorted(obj.items())))
    if isinstance(obj, list):
        return ("list", tuple(_to_tuple(x) for x in obj))
    return obj


def _from_tuple(obj: Any) -> Any:
    if isinstance(obj, tuple) and obj and obj[0] == "dict":
        return {k: _from_tuple(v) for k, v in obj[1]}
    if isinstance(obj, tuple) and obj and obj[0] == "list":
        return [_from_tuple(x) for x in obj[1]]
    return obj


# ── Interpreter ───────────────────────────────────────────────────

def normalize_prog_json(obj: Any) -> dict:
    """Accept both canonical {op:...} and LLM shorthand {translate: \"N\"}."""
    if not isinstance(obj, dict):
        raise ValueError(f"prog must be dict, got {type(obj)}")
    if "op" in obj:
        # Recurse into children
        out = dict(obj)
        if out["op"] == "seq":
            out["parts"] = [normalize_prog_json(p) for p in out.get("parts", [])]
        elif out["op"] == "cond":
            out["then"] = normalize_prog_json(out.get("then", {"op": "noop"}))
            out["else"] = normalize_prog_json(out.get("else", {"op": "noop"}))
        return out

    # Shorthand: single-key dicts
    if len(obj) == 1:
        key, val = next(iter(obj.items()))
        if key == "noop":
            return {"op": "noop"}
        if key == "translate":
            return {"op": "translate", "dir": val}
        if key == "rotate":
            return {"op": "rotate", "k": int(val) if not isinstance(val, dict) else int(val.get("k", 1))}
        if key in ("step_toward", "step_away", "teleport"):
            return {"op": key, "sel": val if isinstance(val, dict) else {}}
        if key == "set_color":
            return {"op": "set_color", "delta": int(val) if not isinstance(val, dict) else int(val.get("delta", 1))}
        if key == "color_move":
            return {"op": "color_move"}
        if key in ("repeat_last", "inverse_last"):
            return {"op": key}
        if key == "seq":
            parts = val if isinstance(val, list) else []
            return {"op": "seq", "parts": [normalize_prog_json(p) for p in parts]}
        if key == "cond":
            if not isinstance(val, dict):
                raise ValueError("cond shorthand needs dict")
            return {
                "op": "cond",
                "pred": val.get("pred", {"on": "prev_same"}),
                "then": normalize_prog_json(val.get("then", {"op": "noop"})),
                "else": normalize_prog_json(val.get("else", {"op": "noop"})),
            }

    # Multi-key without op: treat known fields
    if "dir" in obj and "translate" not in obj:
        return {"op": "translate", "dir": obj["dir"]}
    raise ValueError(f"Cannot normalize prog JSON: {obj}")


def interpret(prog: Prog | dict | list, state: GridState, config: EnvConfig) -> GridState:
    ast = prog.to_json() if isinstance(prog, Prog) else prog
    ast = normalize_prog_json(ast)
    return _eval(ast, state, config)


def _eval(node: Any, state: GridState, config: EnvConfig) -> GridState:
    if not isinstance(node, dict):
        raise ValueError(f"Program node must be dict, got {type(node)}")
    op = node.get("op")
    obs = _get_obstacles(config)

    if op == "noop":
        return state

    if op == "translate":
        d = node["dir"]
        if d in DIRS_CARDINAL:
            delta = DIRECTION_DELTAS[DIRS_CARDINAL[d]]
        elif d == "F":
            delta = DIRECTION_DELTAS[state.agent_dir]
        elif d == "B":
            delta = DIRECTION_DELTAS[(state.agent_dir + 2) % 4]
        else:
            raise ValueError(f"bad dir {d}")
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
            state.last_move_delta = delta
        return state

    if op == "rotate":
        k = int(node.get("k", 1))
        state.agent_dir = (state.agent_dir + k) % 4
        return state

    if op == "set_color":
        delta = int(node.get("delta", 1))
        state.agent_color = (state.agent_color + delta) % N_COLORS_TYPE_D
        return state

    if op == "color_move":
        move_dir = COLOR_DIR_MAP.get(state.agent_color, 0)
        delta = DIRECTION_DELTAS[move_dir]
        new_pos = state.agent_pos + delta
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
            state.last_move_delta = delta
        return state

    if op in ("step_toward", "step_away", "teleport"):
        target = _select_cell(node.get("sel", {}), state, config)
        if target is None:
            return state
        if op == "teleport":
            state.agent_pos = target
            return state
        # unit step toward / away
        dr = target.row - state.agent_pos.row
        dc = target.col - state.agent_pos.col
        if abs(dr) >= abs(dc):
            step = (1 if dr > 0 else -1, 0) if dr != 0 else (0, 1 if dc > 0 else -1)
        else:
            step = (0, 1 if dc > 0 else -1)
        if op == "step_away":
            step = (-step[0], -step[1])
        new_pos = state.agent_pos + step
        if _in_bounds(new_pos, config.grid_size, obs):
            state.agent_pos = new_pos
            state.last_move_delta = step
        return state

    if op == "cond":
        pred = node.get("pred", {})
        then_n = node.get("then", {"op": "noop"})
        else_n = node.get("else", {"op": "noop"})
        if _eval_pred(pred, state, config):
            return _eval(then_n, state, config)
        return _eval(else_n, state, config)

    if op == "seq":
        for part in node.get("parts", []):
            state = _eval(part, state, config)
        return state

    if op == "repeat_last":
        if state.last_move_delta is not None:
            new_pos = state.agent_pos + state.last_move_delta
            if _in_bounds(new_pos, config.grid_size, obs):
                state.agent_pos = new_pos
        return state

    if op == "inverse_last":
        if state.last_move_delta is not None:
            d = state.last_move_delta
            delta = (-d[0], -d[1])
            new_pos = state.agent_pos + delta
            if _in_bounds(new_pos, config.grid_size, obs):
                state.agent_pos = new_pos
                state.last_move_delta = delta
        return state

    raise ValueError(f"Unknown op: {op}")


def _eval_pred(pred: dict, state: GridState, config: EnvConfig) -> bool:
    on = pred.get("on", "prev_same")
    # prev_action comparison needs caller to set state.prev_action vs current;
    # we store the candidate action index on state via a transient attribute.
    cur = getattr(state, "_cond_action", -2)
    if on == "prev_same":
        return state.prev_action >= 0 and cur == state.prev_action
    if on == "prev_diff":
        return state.prev_action < 0 or cur != state.prev_action
    if on == "color_match_cell":
        cell = config.cell_colors[state.agent_pos.row][state.agent_pos.col]
        return state.agent_color == (cell % N_COLORS_TYPE_D)
    return False


def _select_cell(sel: dict, state: GridState, config: EnvConfig) -> Position | None:
    metric = sel.get("metric", "manhattan")
    agg = sel.get("agg", "min")
    pred = sel.get("pred", "any")
    scope = sel.get("scope", "grid")
    agent_color = config.cell_colors[state.agent_pos.row][state.agent_pos.col]
    obs = _get_obstacles(config)

    candidates: list[Position] = []
    if scope == "neighbors":
        for d in range(4):
            nb = state.agent_pos + DIRECTION_DELTAS[d]
            if _in_bounds(nb, config.grid_size, obs):
                candidates.append(nb)
    else:
        for r in range(config.grid_size):
            for c in range(config.grid_size):
                if r == state.agent_pos.row and c == state.agent_pos.col:
                    continue
                candidates.append(Position(r, c))

    filtered: list[tuple[Position, int]] = []  # (pos, discovery_index)
    for i, p in enumerate(candidates):
        cc = config.cell_colors[p.row][p.col]
        if pred == "any":
            ok = True
        elif pred == "same":
            ok = cc == agent_color
        elif pred == "diff":
            ok = cc != agent_color
        elif pred == "brighter":
            ok = cc > agent_color
        elif pred == "darker":
            ok = cc < agent_color
        else:
            ok = True
        # match env move_toward_brightest: ignore zero-color neighbors
        if scope == "neighbors" and metric == "color" and agg == "max" and cc <= 0:
            ok = False
        if ok:
            filtered.append((p, i))
    if not filtered:
        return None

    def score(p: Position) -> float:
        if metric == "color":
            return float(config.cell_colors[p.row][p.col])
        return float(abs(p.row - state.agent_pos.row) + abs(p.col - state.agent_pos.col))

    # Ties: neighbors preserve N,E,S,W discovery order (matches apply_type_c);
    # grid uses row-major.
    if scope == "neighbors":
        filtered.sort(key=lambda pi: (
            score(pi[0]) if agg == "min" else -score(pi[0]), pi[1]
        ))
    else:
        filtered.sort(key=lambda pi: (
            score(pi[0]) if agg == "min" else -score(pi[0]),
            pi[0].row, pi[0].col,
        ))
    return filtered[0][0]


# ── Apply a full action mapping (list of Progs) ───────────────────

def apply_prog_mapping(
    state: GridState,
    config: EnvConfig,
    mapping: list[Prog],
    action_idx: int,
) -> GridState:
    """Apply mapping[action_idx], updating prev_action / last_move_delta."""
    s = state.clone()
    s._cond_action = action_idx  # type: ignore[attr-defined]
    before = s.agent_pos
    s = interpret(mapping[action_idx], s, config)
    if s.agent_pos != before and s.last_move_delta is None:
        s.last_move_delta = (
            s.agent_pos.row - before.row,
            s.agent_pos.col - before.col,
        )
    s.prev_action = action_idx
    return s


# ── Ground-truth compilers (env effect name → Prog) ───────────────

def compile_effect_name(name: str) -> Prog:
    """Compile a known AlienBody effect name into the open DSL (audit/tests)."""
    table = {
        "up": {"op": "translate", "dir": "N"},
        "down": {"op": "translate", "dir": "S"},
        "left": {"op": "translate", "dir": "W"},
        "right": {"op": "translate", "dir": "E"},
        "forward": {"op": "translate", "dir": "F"},
        "backward": {"op": "translate", "dir": "B"},
        "rotate_cw": {"op": "rotate", "k": 1},
        "rotate_ccw": {"op": "rotate", "k": -1},
        "move_to_nearest_diff_color": {
            "op": "teleport",
            "sel": {"metric": "manhattan", "agg": "min", "pred": "diff", "scope": "grid"},
        },
        "move_to_nearest_same_color": {
            "op": "teleport",
            "sel": {"metric": "manhattan", "agg": "min", "pred": "same", "scope": "grid"},
        },
        "move_toward_brightest": {
            "op": "step_toward",
            "sel": {"metric": "color", "agg": "max", "pred": "any", "scope": "neighbors"},
        },
        "flee_same_color": {
            "op": "step_away",
            "sel": {"metric": "manhattan", "agg": "min", "pred": "same", "scope": "grid"},
        },
        "move_to_farthest_same_color": {
            "op": "teleport",
            "sel": {"metric": "manhattan", "agg": "max", "pred": "same", "scope": "grid"},
        },
        "move_away_from_brightest": {
            "op": "step_away",
            "sel": {"metric": "color", "agg": "max", "pred": "any", "scope": "neighbors"},
        },
        "move_to_farthest_diff_color": {
            "op": "teleport",
            "sel": {"metric": "manhattan", "agg": "max", "pred": "diff", "scope": "grid"},
        },
        "flee_nearest_diff_color": {
            "op": "step_away",
            "sel": {"metric": "manhattan", "agg": "min", "pred": "diff", "scope": "grid"},
        },
        "move_toward_darkest": {
            "op": "step_toward",
            "sel": {"metric": "color", "agg": "min", "pred": "any", "scope": "neighbors"},
        },
        "move_away_from_darkest": {
            "op": "step_away",
            "sel": {"metric": "color", "agg": "min", "pred": "any", "scope": "neighbors"},
        },
        "move_to_nearest_brighter": {
            "op": "teleport",
            "sel": {"metric": "manhattan", "agg": "min", "pred": "brighter", "scope": "grid"},
        },
        "move_to_nearest_darker": {
            "op": "teleport",
            "sel": {"metric": "manhattan", "agg": "min", "pred": "darker", "scope": "grid"},
        },
        "color_inc": {"op": "set_color", "delta": 1},
        "color_dec": {"op": "set_color", "delta": -1},
        "color_move": {"op": "color_move"},
        "temporal_2": {"op": "repeat_last"},
        "temporal_3": {"op": "inverse_last"},
        "move_and_rotate_cw": {
            "op": "seq",
            "parts": [{"op": "translate", "dir": "F"}, {"op": "rotate", "k": 1}],
        },
        "rotate_and_strafe": {
            "op": "seq",
            "parts": [{"op": "rotate", "k": -1}, {"op": "translate", "dir": "F"}],
        },
        "double_forward": {
            "op": "seq",
            "parts": [{"op": "translate", "dir": "F"}, {"op": "translate", "dir": "F"}],
        },
        "retreat_and_spin": {
            "op": "seq",
            "parts": [
                {"op": "translate", "dir": "B"},
                {"op": "rotate", "k": 1},
                {"op": "rotate", "k": 1},
            ],
        },
    }
    # temporal_0 / temporal_1 need cond
    if name == "temporal_0":
        return Prog.from_json({
            "op": "cond",
            "pred": {"on": "prev_same"},
            "then": {"op": "translate", "dir": "N"},
            "else": {"op": "translate", "dir": "E"},
        })
    if name == "temporal_1":
        return Prog.from_json({
            "op": "cond",
            "pred": {"on": "prev_same"},
            "then": {"op": "translate", "dir": "S"},
            "else": {"op": "translate", "dir": "W"},
        })
    if name == "color_interact":
        return Prog.from_json({
            "op": "cond",
            "pred": {"on": "color_match_cell"},
            "then": {"op": "translate", "dir": "F"},
            "else": {"op": "rotate", "k": 1},
        })
    if name not in table:
        raise KeyError(f"No DSL compile rule for effect {name}")
    return Prog.from_json(table[name])


def compile_mapping(effect_names: list[str]) -> list[Prog]:
    return [compile_effect_name(n) for n in effect_names]


# ── Blind program generator (budgeted) ────────────────────────────

_ATOMIC_TEMPLATES: list[dict] = [
    {"op": "translate", "dir": d} for d in ("N", "E", "S", "W", "F", "B")
] + [
    {"op": "rotate", "k": k} for k in (-1, 1, 2)
] + [
    {"op": "teleport", "sel": {
        "metric": "manhattan", "agg": agg, "pred": pred, "scope": "grid",
    }}
    for agg in ("min", "max")
    for pred in ("same", "diff", "brighter", "darker")
] + [
    {"op": "step_toward", "sel": {
        "metric": "color", "agg": agg, "pred": "any", "scope": "neighbors",
    }}
    for agg in ("min", "max")
] + [
    {"op": "step_away", "sel": {
        "metric": "manhattan", "agg": "min", "pred": pred, "scope": "grid",
    }}
    for pred in ("same", "diff")
] + [
    {"op": "step_away", "sel": {
        "metric": "color", "agg": agg, "pred": "any", "scope": "neighbors",
    }}
    for agg in ("min", "max")
] + [
    {"op": "set_color", "delta": 1},
    {"op": "set_color", "delta": -1},
    {"op": "color_move"},
    {"op": "repeat_last"},
    {"op": "inverse_last"},
    {"op": "noop"},
]


def iter_atomic_programs() -> Iterator[Prog]:
    for t in _ATOMIC_TEMPLATES:
        yield Prog.from_json(t)


def iter_depth2_programs(limit: int = 5000) -> Iterator[Prog]:
    """seq of two atomics — capped."""
    atomics = list(iter_atomic_programs())
    n = 0
    for a, b in itertools.product(atomics, atomics):
        yield Prog.from_json({"op": "seq", "parts": [a.to_json(), b.to_json()]})
        n += 1
        if n >= limit:
            return


def program_space_stats() -> dict:
    n_atomic = len(list(iter_atomic_programs()))
    return {
        "n_atomic": n_atomic,
        "n_depth2_uncapped": n_atomic * n_atomic,
        "note": "cond/temporal expand further; n=12 assignment is (n_atomic)^12",
    }


def dsl_prompt_spec() -> str:
    """Text shown to VLM proposers."""
    return (
        "Emit one JSON program per action using this DSL.\n"
        "Canonical form (preferred):\n"
        '  {"op":"translate","dir":"N"}  dir in {N,E,S,W,F,B}\n'
        '  {"op":"rotate","k":1}         k in {-1,1,2}\n'
        '  {"op":"teleport"|"step_toward"|"step_away",\n'
        '   "sel":{"metric":"manhattan"|"color","agg":"min"|"max",\n'
        '          "pred":"same"|"diff"|"any"|"brighter"|"darker",\n'
        '          "scope":"grid"|"neighbors"}}\n'
        '  {"op":"seq","parts":[...]}  {"op":"noop"}\n'
        '  {"op":"cond","pred":{"on":"prev_same"|"prev_diff"|"color_match_cell"},\n'
        '   "then":..., "else":...}\n'
        "Shorthand also accepted: {\"translate\":\"N\"}, {\"seq\":[...]}, {\"noop\":null}.\n"
        "For Relational grids, prefer teleport/step_* with color predicates over plain translate.\n"
        "Return JSON: {\"0\": <prog>, \"1\": <prog>, ...}."
    )
