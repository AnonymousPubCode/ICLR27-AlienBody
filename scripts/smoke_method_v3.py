#!/usr/bin/env python3
"""Smoke tests for Method v3 scaffolding (no API required).

Checks:
  1. Open DSL compiles all Tier-M F4 effect names and matches apply_type_c
     on a sample of positions (expressibility audit).
  2. Sandboxed BFS reference plan solves Oracle-solvable F4 envs.
  3. DSLBlindAgent runs end-to-end on a few F4 envs (may have low SR —
     that's the baseline, not a failure of the smoke test).
  4. CodeToolAgent sandbox executes a hand-written BFS and returns PLAN.

Usage:
  python scripts/smoke_method_v3.py
  python scripts/smoke_method_v3.py --n 10
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from alienbody.agents import run_episode
from alienbody.agents.code_tool_agent import (
    CodeToolAgent, _run_sandbox, reference_bfs_plan,
)
from alienbody.agents.dsl_agent import DSLBlindAgent
from alienbody.effect_dsl import (
    compile_effect_name, compile_mapping, interpret, program_space_stats,
)
from alienbody.env.actions import apply_type_c
from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig, GridState, Phase, Position


def load_f4(data_dir: Path, n: int) -> list[EnvConfig]:
    d = data_dir / "family4" / "test"
    configs = []
    for p in sorted(d.glob("env_*.json"))[:n]:
        configs.append(EnvConfig.from_file(str(p)))
    return configs


def audit_dsl_expressibility(configs: list[EnvConfig], probes: int = 20) -> dict:
    """Compare DSL compile(effect) vs apply_type_c on random positions."""
    import random
    rng = random.Random(0)
    n_ok = n_tot = 0
    mismatches = []
    for cfg in configs:
        mapping = list(cfg.action_mapping)
        progs = compile_mapping(mapping)
        gs = cfg.grid_size
        for _ in range(probes):
            r, c = rng.randrange(gs), rng.randrange(gs)
            for a, (name, prog) in enumerate(zip(mapping, progs)):
                # env truth
                temp = copy.deepcopy(cfg)
                object.__setattr__(temp, "action_mapping", tuple(mapping))
                st = GridState(
                    agent_pos=Position(r, c), agent_dir=0, agent_color=0,
                    phase=Phase.CALIBRATION, step_count=0, phase1_steps=0, phase2_steps=0,
                )
                truth = apply_type_c(copy.deepcopy(st), temp, a)
                # dsl
                st2 = GridState(
                    agent_pos=Position(r, c), agent_dir=0, agent_color=0,
                    phase=Phase.CALIBRATION, step_count=0, phase1_steps=0, phase2_steps=0,
                )
                pred = interpret(prog, st2, cfg)
                n_tot += 1
                if (pred.agent_pos.row, pred.agent_pos.col) == (
                    truth.agent_pos.row, truth.agent_pos.col
                ):
                    n_ok += 1
                else:
                    if len(mismatches) < 8:
                        mismatches.append({
                            "env": cfg.env_id, "pos": (r, c), "a": a, "name": name,
                            "pred": (pred.agent_pos.row, pred.agent_pos.col),
                            "truth": (truth.agent_pos.row, truth.agent_pos.col),
                        })
    return {"match_rate": n_ok / max(n_tot, 1), "n": n_tot, "mismatches": mismatches}


def test_reference_bfs(configs: list[EnvConfig]) -> dict:
    ok = 0
    for cfg in configs:
        plan = reference_bfs_plan(cfg)
        if not plan:
            continue
        # simulate
        pos = tuple(cfg.agent_start)
        for a in plan:
            st = GridState(
                agent_pos=Position(*pos), agent_dir=0, agent_color=0,
                phase=Phase.EXECUTION, step_count=0, phase1_steps=0, phase2_steps=0,
            )
            from alienbody.env.actions import apply_action
            st = apply_action(st, cfg, a)
            pos = (st.agent_pos.row, st.agent_pos.col)
        if pos == tuple(cfg.target_pos):
            ok += 1
    return {"solved": ok, "n": len(configs)}


def test_sandbox_bfs(cfg: EnvConfig) -> dict:
    """Hand-written BFS inside the CodeTool sandbox."""
    from alienbody.agents.code_tool_agent import CodeToolAgent
    # Fake client unused — we call sandbox directly
    class _Dummy:
        name = "dummy"
        def complete(self, *a, **k):
            return ""

    agent = CodeToolAgent(cfg, _Dummy())
    start = tuple(cfg.agent_start)
    code = """
from collections import deque
q = deque([(START, [])])
seen = {START}
PLAN = []
while q:
    (r, c), path = q.popleft()
    if (r, c) == TARGET:
        PLAN = path
        break
    if len(path) >= 40:
        continue
    for a in range(N_ACTIONS):
        nr, nc = next_state(r, c, a)
        if (nr, nc) not in seen:
            seen.add((nr, nc))
            q.append(((nr, nc), path + [a]))
"""
    # sandbox forbids import — rewrite without import (deque is preload)
    code = """
q = deque([(START, [])])
seen = {START}
PLAN = []
while q:
    (r, c), path = q.popleft()
    if (r, c) == TARGET:
        PLAN = path
        break
    if len(path) >= 40:
        continue
    for a in range(N_ACTIONS):
        nr, nc = next_state(r, c, a)
        if (nr, nc) not in seen:
            seen.add((nr, nc))
            q.append(((nr, nc), path + [a]))
"""
    result = _run_sandbox(code, agent._sandbox_env(start))
    plan = result.get("PLAN") or []
    pos = start
    for a in plan:
        pos = agent._next_state_fn(pos[0], pos[1], a)
    return {
        "plan_len": len(plan),
        "reached": pos == tuple(cfg.target_pos),
        "sim_calls": agent._sim_calls,
    }


def test_dsl_blind(configs: list[EnvConfig]) -> dict:
    succ = 0
    for cfg in configs:
        env = AlienBodyEnv(cfg, render_mode="text")
        agent = DSLBlindAgent(cfg, max_candidates=800, phase1_sweeps=3)
        traj = run_episode(env, agent)
        if traj.get("success"):
            succ += 1
    return {"sr": 100.0 * succ / max(len(configs), 1), "n": len(configs), "success": succ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/envs")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    configs = load_f4(root / args.data_dir, args.n)
    print(f"Loaded {len(configs)} F4 envs")
    print(f"DSL space: {program_space_stats()}")

    print("\n[1] DSL expressibility audit (compile vs apply_type_c)")
    audit = audit_dsl_expressibility(configs)
    print(f"  match_rate={audit['match_rate']:.3f} over {audit['n']} probes")
    if audit["mismatches"]:
        print(f"  sample mismatches: {audit['mismatches'][:3]}")

    print("\n[2] Reference BFS (code-tool semantics)")
    bfs = test_reference_bfs(configs)
    print(f"  solved {bfs['solved']}/{bfs['n']}")

    print("\n[3] Sandbox hand-written BFS")
    sb = test_sandbox_bfs(configs[0])
    print(f"  env={configs[0].env_id} reached={sb['reached']} "
          f"plan_len={sb['plan_len']} sim_calls={sb['sim_calls']}")

    print("\n[4] DSLBlindAgent end-to-end (baseline; SR may be low)")
    blind = test_dsl_blind(configs)
    print(f"  SR={blind['sr']:.1f}% ({blind['success']}/{blind['n']})")

    print("\nDone.")
    # Soft gate: sandbox BFS must work; DSL expressibility should be high
    if not sb["reached"]:
        sys.exit("FAIL: sandbox BFS did not reach target")
    if audit["match_rate"] < 0.85:
        print("WARN: DSL expressibility < 85% — check compile rules / tie-breaks")
        sys.exit(2)


if __name__ == "__main__":
    main()
