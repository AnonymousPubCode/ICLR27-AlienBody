"""Trajectory persistence — JSONL save/load for experiment results.

Each trajectory is one JSON line in a .jsonl file.
Supports streaming writes (crash-safe) and filtered reads.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator


class TrajectoryWriter:
    """Append trajectories to a JSONL file. Use as context manager.

    Usage:
        with TrajectoryWriter("results/gpt4o/family1_test.jsonl") as writer:
            for env in environments:
                trajectory = run_episode(env, agent)
                writer.write(trajectory, agent_name="gpt4o", modality="image")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def write(self, trajectory: dict, **extra_fields) -> None:
        """Write a trajectory with optional extra metadata."""
        record = {
            "timestamp": datetime.now().isoformat(),
            **trajectory,
            **extra_fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()  # flush each record for crash safety
        else:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def load_trajectories(path: str | Path) -> list[dict]:
    """Load all trajectories from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_trajectories_dir(
    dir_path: str | Path,
    filter_fn: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Load trajectories from all .jsonl files in a directory."""
    dir_path = Path(dir_path)
    records = []
    for jsonl_file in sorted(dir_path.glob("**/*.jsonl")):
        for record in load_trajectories(jsonl_file):
            if filter_fn is None or filter_fn(record):
                records.append(record)
    return records


def get_completed_env_ids(path: str | Path) -> set[str]:
    """Get set of env_ids already completed (for resume support)."""
    records = load_trajectories(path)
    return {r["env_id"] for r in records if "env_id" in r}


def trajectory_summary(trajectories: list[dict]) -> dict:
    """Quick summary stats for a list of trajectories."""
    if not trajectories:
        return {"count": 0}

    n = len(trajectories)
    successes = sum(1 for t in trajectories if t.get("success", False))

    return {
        "count": n,
        "success_count": successes,
        "success_rate": successes / n * 100,
        "families": sorted(set(t.get("family", 0) for t in trajectories)),
        "agents": sorted(set(t.get("agent_name", "unknown") for t in trajectories)),
    }
