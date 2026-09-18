"""Environment generator + solvability validator.

Generates EnvConfig instances for each family with controlled randomization.
Validates every generated environment is solvable within the action budget
using an oracle BFS solver.
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Iterator

from alienbody.env.grid import EnvConfig, Position, GridState, Phase, DIRECTION_DELTAS
from alienbody.env.actions import (
    apply_action, get_all_effect_names,
    TYPE_B_EFFECTS, TYPE_A_EFFECTS, TYPE_C_EFFECTS, TYPE_D_EFFECTS,
    TYPE_E_EFFECTS, TYPE_F_EFFECTS,
)

# ── Family Configurations ──────────────────────────────────────────

FAMILY_CONFIGS = {
    1: {
        "name": "Remapped Actions",
        "action_type": "B",
        "effects": list(TYPE_B_EFFECTS.keys()),  # up, down, left, right
        "description": "Remapped cardinal directions. Baseline difficulty.",
    },
    2: {
        "name": "Directional",
        "action_type": "A",
        "effects": list(TYPE_A_EFFECTS),
        "description": "Rotation-based movement. Requires egocentric reasoning.",
    },
    3: {
        "name": "State-Dependent",
        "action_type": "A+D",
        # For combo: effects are assigned per-type, not mixed.
        # Generator will pick 2 from Type A + 2 from Type D.
        "effects_A": ["rotate_cw", "forward"],           # always these 2 for Type A half
        "effects_D": ["color_inc", "color_move"],         # always these 2 for Type D half
        "effects": None,  # built dynamically in generate_env
        "description": "Actions depend on agent state. Medium difficulty.",
    },
    4: {
        "name": "Relational",
        "action_type": "C",
        # Tier-M vocabulary: the original 4 relation types. Tier-L (n=6)
        # extends to the full TYPE_C_EFFECTS list via effects_override.
        "effects": list(TYPE_C_EFFECTS[:4]),
        "description": "Actions depend on object relationships. Highest difficulty.",
    },
    5: {
        "name": "Composite",
        "action_type": "E",
        "effects": list(TYPE_E_EFFECTS),
        "description": "Two-step composite actions. Requires decomposition.",
    },
    6: {
        "name": "Temporal",
        "action_type": "F",
        "effects": list(TYPE_F_EFFECTS),
        "description": "Prev-action dependent effects. Requires history tracking.",
    },
}

SPLIT_SIZES = {"train": 50, "dev": 20, "test": 50, "secret": 30}  # per family


# ── Generator ──────────────────────────────────────────────────────

def generate_env(
    family: int,
    idx: int,
    split: str,
    seed: int | None = None,
    grid_size: int = 16,
    min_distance: int = 5,
    phase1_budget: int = 20,
    max_total_steps: int = 50,
    n_actions: int = 4,
    n_obstacles_range: tuple[int, int] = (3, 6),
    tier_tag: str = "",
    effects_override: list[str] | None = None,
) -> EnvConfig:
    """Generate a single environment config.

    Args:
        family: 1-4
        idx: index within split
        split: "train", "dev", "test", "secret"
        seed: random seed (default: deterministic from family+split+idx)
        grid_size: grid dimension
        min_distance: minimum Manhattan distance between agent and target
        phase1_budget: max Phase 1 actions
        max_total_steps: max total actions
        n_actions: number of actions (must not exceed family's effect count)
        n_obstacles_range: (min, max) obstacles per env
        tier_tag: env_id suffix (e.g. "l" for Tier-L) to avoid colliding
            with Tier-M environment ids
    """
    if seed is None:
        seed = family * 100000 + hash(split) % 1000 * 100 + idx

    rng = random.Random(seed)
    fconfig = FAMILY_CONFIGS[family]

    # Generate grid colors
    if family == 4:
        # Relational: need diverse colored cells for Type C actions
        n_special_colors = rng.randint(3, 5)
        palette = rng.sample(range(10), n_special_colors)
        cell_colors = _generate_colored_grid(rng, grid_size, palette)
    elif family == 5:
        # Composite: sparse colored (same as rotation-based, agent needs orientation)
        cell_colors = _generate_sparse_grid(rng, grid_size, n_colored=rng.randint(4, 10))
    elif family == 6:
        # Temporal: sparse colored, similar to remapped cardinal
        cell_colors = _generate_sparse_grid(rng, grid_size, n_colored=rng.randint(4, 10))
    else:
        # Other families: mostly black (0) with sparse colored cells
        cell_colors = _generate_sparse_grid(rng, grid_size, n_colored=rng.randint(4, 10))

    # Agent and target placement
    while True:
        agent_r, agent_c = rng.randint(1, grid_size - 2), rng.randint(1, grid_size - 2)
        target_r, target_c = rng.randint(1, grid_size - 2), rng.randint(1, grid_size - 2)
        dist = abs(agent_r - target_r) + abs(agent_c - target_c)
        if dist >= min_distance:
            break

    # Randomize action mapping
    if "+" in fconfig["action_type"]:
        # Combo type: shuffle within each type's effects independently
        effects_a = list(fconfig["effects_A"])
        effects_d = list(fconfig["effects_D"])
        rng.shuffle(effects_a)
        rng.shuffle(effects_d)
        action_mapping = tuple(effects_a + effects_d)
    else:
        effects = list(fconfig["effects"]) if effects_override is None else list(effects_override)
        if n_actions > len(effects):
            raise ValueError(
                f"family {family} has {len(effects)} effects, cannot generate "
                f"n_actions={n_actions}"
            )
        rng.shuffle(effects)
        action_mapping = tuple(effects[:n_actions])

    # Agent initial state
    agent_start_dir = rng.randint(0, 3)
    agent_start_color = rng.randint(0, 3) if "D" in fconfig["action_type"] else 0

    # Generate obstacles (n_obstacles_range per env, not on agent or target)
    obstacles = []
    n_obstacles = rng.randint(*n_obstacles_range)
    for _ in range(n_obstacles * 3):  # over-generate, filter
        or_, oc = rng.randint(1, grid_size - 2), rng.randint(1, grid_size - 2)
        if (or_, oc) != (agent_r, agent_c) and (or_, oc) != (target_r, target_c):
            if (or_, oc) not in obstacles:
                obstacles.append((or_, oc))
                if len(obstacles) >= n_obstacles:
                    break

    env_id = f"family{family}{tier_tag}_{split}_{idx:03d}"

    return EnvConfig(
        env_id=env_id,
        family=family,
        seed=seed,
        grid_size=grid_size,
        cell_colors=tuple(tuple(row) for row in cell_colors),
        agent_start=(agent_r, agent_c),
        agent_start_color=agent_start_color,
        agent_start_dir=agent_start_dir,
        target_pos=(target_r, target_c),
        action_type=fconfig["action_type"],
        action_mapping=action_mapping,
        n_actions=n_actions,
        phase1_budget=phase1_budget,
        max_total_steps=max_total_steps,
        obstacles=tuple(tuple(o) for o in obstacles),
    )


def _generate_sparse_grid(rng: random.Random, size: int, n_colored: int) -> list[list[int]]:
    """Grid with mostly black (0) cells and some colored cells."""
    grid = [[0] * size for _ in range(size)]
    for _ in range(n_colored):
        r, c = rng.randint(0, size - 1), rng.randint(0, size - 1)
        grid[r][c] = rng.randint(1, 9)
    return grid


def _generate_colored_grid(rng: random.Random, size: int, palette: list[int]) -> list[list[int]]:
    """Grid with diverse colored regions for Type C (relational) actions."""
    grid = [[0] * size for _ in range(size)]
    # Create color clusters
    n_clusters = rng.randint(8, 15)
    for _ in range(n_clusters):
        cr, cc = rng.randint(0, size - 1), rng.randint(0, size - 1)
        color = rng.choice(palette)
        # Fill a small region
        spread = rng.randint(1, 3)
        for dr in range(-spread, spread + 1):
            for dc in range(-spread, spread + 1):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if rng.random() < 0.6:
                        grid[nr][nc] = color
    return grid


# ── Validator (Oracle BFS Solver) ──────────────────────────────────

def validate_solvable(config: EnvConfig, max_steps: int | None = None) -> dict:
    """Check if environment is solvable using BFS with known action mapping.

    Returns dict with:
        solvable: bool
        optimal_steps: int (Phase 2 only, after perfect calibration)
        optimal_path: list of action indices
    """
    if max_steps is None:
        max_steps = config.max_total_steps - config.phase1_budget

    target = Position(*config.target_pos)
    obstacles = set()
    if hasattr(config, 'obstacles') and config.obstacles:
        obstacles = {(r, c) for r, c in config.obstacles}

    # BFS over (position, direction, color, prev_action) state space
    initial = config.make_initial_state()
    start_state = (initial.agent_pos.row, initial.agent_pos.col,
                   initial.agent_dir, initial.agent_color, initial.prev_action)

    queue = deque([(start_state, [])])
    visited = {start_state}

    while queue:
        (r, c, d, color, prev_act), path = queue.popleft()
        if len(path) >= max_steps:
            continue

        for action_idx in range(config.n_actions):
            # Simulate action
            state = GridState(
                agent_pos=Position(r, c),
                agent_color=color,
                agent_dir=d,
                phase=Phase.EXECUTION,
                step_count=0, phase1_steps=0, phase2_steps=0,
                prev_action=prev_act,
            )
            state = apply_action(state, config, action_idx)
            new_key = (state.agent_pos.row, state.agent_pos.col,
                       state.agent_dir, state.agent_color, state.prev_action)

            if state.agent_pos == target:
                return {
                    "solvable": True,
                    "optimal_steps": len(path) + 1,
                    "optimal_path": path + [action_idx],
                }

            if new_key not in visited:
                visited.add(new_key)
                queue.append((new_key, path + [action_idx]))

    return {"solvable": False, "optimal_steps": -1, "optimal_path": []}


# ── Batch Generation ───────────────────────────────────────────────

def generate_family(
    family: int,
    output_dir: str | Path,
    max_retries: int = 100,
) -> dict:
    """Generate all environments for a family, with solvability validation.

    Returns summary stats.
    """
    output_dir = Path(output_dir)
    stats = {"family": family, "generated": 0, "rejected": 0}

    for split, count in SPLIT_SIZES.items():
        split_dir = output_dir / f"family{family}" / split
        split_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        attempts = 0
        while generated < count and attempts < count * max_retries:
            seed = family * 1000000 + hash(split) % 10000 * 1000 + attempts
            config = generate_env(family, generated, split, seed=seed)

            # Validate solvability
            result = validate_solvable(config)
            if result["solvable"] and result["optimal_steps"] >= 3:
                config.save(str(split_dir / f"env_{generated:03d}.json"))
                generated += 1
                stats["generated"] += 1
            else:
                stats["rejected"] += 1
            attempts += 1

    return stats


def generate_all(output_dir: str | Path) -> list[dict]:
    """Generate all 900 environments (6 families × 150 each)."""
    results = []
    for family in range(1, 7):
        stats = generate_family(family, output_dir)
        results.append(stats)
        print(f"Family {family}: generated={stats['generated']}, "
              f"rejected={stats['rejected']}")
    return results
