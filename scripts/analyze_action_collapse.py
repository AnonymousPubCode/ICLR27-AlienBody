#!/usr/bin/env python3
"""Analyze phase-2 action collapse by family (P0.A, life-or-death evidence).

Prove that F4's 0-5% SR is a relational-specific failure, not a generic
model collapse: for the SAME model at the SAME L3 setting, F1-F3 phase-2
should show diverse, purposeful actions while F4 collapses into repeated
single actions with no progress toward the target.

Usage:
  python3 scripts/analyze_action_collapse.py \
      --input results/qwen397b_l3/vllm__project_model_Qwen3.5-397B-A17B-GPTQ-Int4/all_test.jsonl \
      [--input ...] \
      --env-root data/envs \
      --output results/action_collapse/qwen397b_l3.json
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _env_config_path(env_id: str, env_root: str) -> str | None:
    """Map trajectory env_id (e.g. family1_test_000) to its config file."""
    m = re.match(r"(family\d+)_(\w+)_(\d+)", env_id)
    if not m:
        return None
    fam, split, idx = m.groups()
    p = Path(env_root) / fam / split / f"env_{idx}.json"
    return str(p) if p.exists() else None


def _target_pos(env_id: str, env_root: str) -> tuple[int, int] | None:
    p = _env_config_path(env_id, env_root)
    if not p:
        return None
    try:
        cfg = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None
    tp = cfg.get("target_pos")
    if isinstance(tp, (list, tuple)) and len(tp) == 2:
        return (int(tp[0]), int(tp[1]))
    return None


def _dist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class OraclePlanner:
    """Per-env OracleAgent BFS policy with a per-state cache.

    At L3 the agent is given the perfect action mapping, so the oracle's
    optimal move from any state is well-defined. Agreement between the
    agent's actual action and the oracle move measures true planning
    quality, not just "is the agent moving around".
    """

    def __init__(self, env_root: str):
        from alienbody.env.grid import EnvConfig
        from alienbody.agents import OracleAgent

        self._env_root = env_root
        self._EnvConfig = EnvConfig
        self._OracleAgent = OracleAgent
        self._agents: dict[str, OracleAgent] = {}
        self._cache: dict[tuple, int | None] = {}

    def _agent(self, env_id: str):
        if env_id not in self._agents:
            p = _env_config_path(env_id, self._env_root)
            cfg = self._EnvConfig.from_file(p)
            self._agents[env_id] = self._OracleAgent(cfg)
        return self._agents[env_id]

    def oracle_move(self, env_id: str, pos, direction, color, prev_action) -> int | None:
        key = (env_id, tuple(pos), direction, color, prev_action)
        if key in self._cache:
            return self._cache[key]
        try:
            agent = self._agent(env_id)
            plan = agent._bfs_plan({
                "agent_pos": pos,
                "agent_dir": direction,
                "agent_color": color,
                "prev_action": prev_action,
            })
            move = plan[0] if plan else None
        except Exception:
            move = None
        self._cache[key] = move
        return move


def analyze_trajectory(traj: dict, env_root: str, planner: OraclePlanner | None) -> dict:
    metrics = traj.get("metrics", {})
    family = metrics.get("family", traj.get("family"))
    success = bool(traj.get("success", metrics.get("success")))
    env_id = traj.get("env_id", metrics.get("env_id", ""))
    history = traj.get("history", [])

    phase1 = [h for h in history if h.get("phase") == 1]
    phase2 = [h for h in history if h.get("phase") == 2]

    # Phase-1 exploration coverage: did the model even try all actions?
    p1_actions = [h["action"] for h in phase1]
    unique_p1 = len(set(p1_actions))

    # Phase-2 behavior
    p2_actions = [h["action"] for h in phase2]
    n_p2 = len(p2_actions)
    unique_p2 = len(set(p2_actions)) if n_p2 else 0

    # Shannon entropy of the phase-2 action distribution, normalized to [0,1]
    entropy = 0.0
    if n_p2 and unique_p2 > 1:
        counts = Counter(p2_actions)
        n_actions = metrics.get("n_actions", traj.get("n_actions", 4))
        entropy = -sum((c / n_p2) * math.log(c / n_p2) for c in counts.values())
        entropy /= math.log(max(n_actions, 2))

    # Consecutive repetition rate (action == previous action)
    rep_rate = 0.0
    if n_p2 >= 2:
        rep_rate = sum(1 for i in range(1, n_p2) if p2_actions[i] == p2_actions[i - 1]) / (n_p2 - 1)

    # Longest run of the same action
    max_run = 0
    run = 0
    prev = None
    for a in p2_actions:
        run = run + 1 if a == prev else 1
        prev = a
        max_run = max(max_run, run)

    # Purposefulness: does each phase-2 step move toward the target?
    target = _target_pos(env_id, env_root)
    toward = away = still = 0
    max_progress = 0.0  # max reduction in distance-to-target in any single step
    if target is not None:
        for h in phase2:
            pv = h.get("prev_pos")
            nv = h.get("new_pos")
            if not pv or not nv:
                continue
            d0 = _dist(pv, target)
            d1 = _dist(nv, target)
            if d1 < d0:
                toward += 1
            elif d1 > d0:
                away += 1
            else:
                still += 1
            max_progress = max(max_progress, float(d0 - d1))

    n_valid = toward + away + still
    toward_rate = toward / n_valid if n_valid else 0.0
    still_rate = still / n_valid if n_valid else 0.0

    # Toward-rate by episode half: degeneration over time is the collapse
    # signature (starts purposeful, degenerates into repetition).
    def _toward_rate(steps) -> float:
        if target is None or not steps:
            return 0.0
        t = 0
        for h in steps:
            pv, nv = h.get("prev_pos"), h.get("new_pos")
            if not pv or not nv:
                continue
            if _dist(nv, target) < _dist(pv, target):
                t += 1
        return t / len(steps)

    half = max(1, n_p2 // 2)
    toward_first_half = _toward_rate(phase2[:half])
    toward_second_half = _toward_rate(phase2[half:])

    # Oracle agreement: fraction of steps where the agent's action equals
    # the oracle BFS optimal move from the same state (with perfect mapping).
    oracle_agree = oracle_agree_valid = 0
    if planner is not None:
        for i, h in enumerate(phase2):
            pv = h.get("prev_pos")
            if not pv:
                continue
            prev_action = phase2[i - 1]["action"] if i > 0 else -1
            move = planner.oracle_move(
                env_id, pv,
                h.get("prev_dir", 0),
                h.get("prev_color", 0),
                prev_action,
            )
            if move is None:
                continue
            oracle_agree_valid += 1
            if move == h.get("action"):
                oracle_agree += 1
    oracle_rate = oracle_agree / oracle_agree_valid if oracle_agree_valid else 0.0

    return {
        "env_id": env_id,
        "family": family,
        "success": success,
        "n_p1": len(phase1),
        "n_p2": n_p2,
        "unique_p1": unique_p1,
        "unique_p2": unique_p2,
        "entropy_p2": entropy,
        "rep_rate_p2": rep_rate,
        "max_run_p2": max_run,
        "toward_p2": toward,
        "away_p2": away,
        "still_p2": still,
        "toward_rate_p2": toward_rate,
        "still_rate_p2": still_rate,
        "toward_first_half": toward_first_half,
        "toward_second_half": toward_second_half,
        "max_progress_p2": max_progress,
        "oracle_agree_valid": oracle_agree_valid,
        "oracle_agree_rate": oracle_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    help="trajectory jsonl (repeatable)")
    ap.add_argument("--env-root", default="data/envs")
    ap.add_argument("--output", required=True, help="output JSON path")
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip oracle-agreement computation (faster)")
    args = ap.parse_args()

    planner = None if args.no_oracle else OraclePlanner(args.env_root)

    rows = []
    for path in args.input:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    traj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(analyze_trajectory(traj, args.env_root, planner))

    if not rows:
        print("no trajectories found")
        return

    # Aggregate per family
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["family"]]["rows"].append(r)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {}
    print("family   n   SR%  uniqP1 uniqP2 entropy repRate maxRun "
          "toward% still% tow1st tow2nd  oracleAgree%")
    for fam in sorted(agg.keys()):
        rs = agg[fam]["rows"]
        n = len(rs)
        sr = mean([r["success"] for r in rs]) * 100
        u1 = mean([r["unique_p1"] for r in rs])
        u2 = mean([r["unique_p2"] for r in rs])
        ent = mean([r["entropy_p2"] for r in rs])
        rep = mean([r["rep_rate_p2"] for r in rs])
        mr = mean([r["max_run_p2"] for r in rs])
        tw = mean([r["toward_rate_p2"] for r in rs]) * 100
        st = mean([r["still_rate_p2"] for r in rs]) * 100
        t1 = mean([r["toward_first_half"] for r in rs]) * 100
        t2 = mean([r["toward_second_half"] for r in rs]) * 100
        oa = mean([r["oracle_agree_rate"] for r in rs]) * 100
        c1 = mean([r["unique_p2"] <= 1 for r in rs]) * 100
        c2 = mean([r["unique_p2"] <= 2 for r in rs]) * 100
        summary[f"F{fam}"] = {
            "n": n, "sr_pct": sr, "unique_p1_mean": u1, "unique_p2_mean": u2,
            "entropy_p2_mean": ent, "rep_rate_p2_mean": rep,
            "max_run_p2_mean": mr, "toward_rate_p2_mean_pct": tw,
            "still_rate_p2_mean_pct": st,
            "toward_first_half_pct": t1, "toward_second_half_pct": t2,
            "oracle_agree_rate_pct": oa,
            "collapse_le1_pct": c1, "collapse_le2_pct": c2,
        }
        print(f"F{fam:<6} {n:>3} {sr:>5.1f} {u1:>6.2f} {u2:>6.2f} {ent:>7.2f} "
              f"{rep:>7.2f} {mr:>6.1f} {tw:>7.1f} {st:>6.1f} {t1:>6.1f} "
              f"{t2:>6.1f} {oa:>12.1f}")

    out = {
        "inputs": args.input,
        "n_trajectories": len(rows),
        "per_family": summary,
        "trajectories": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
