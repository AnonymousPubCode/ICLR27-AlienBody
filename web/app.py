#!/usr/bin/env python3
"""AlienBody Web — Game-like interface for the action calibration benchmark.

Usage:
    cd papers/alienbody/code
    python web/app.py
    # Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig, Phase
from alienbody.env.generator import generate_env
from alienbody.agents import (
    RandomAgent, OracleAgent, SystematicExplorer, DONE_EXPLORING,
)
from alienbody.agents.algorithmic_agents import (
    BayesianExplorer, MemoryAugmentedExplorer,
)

app = FastAPI(title="AlienBody")
STATIC = Path(__file__).parent / "static"

FAMILY_NAMES = {
    1: "Remapped",
    2: "Directional",
    3: "State-Dependent",
    4: "Relational",
    5: "Composite",
    6: "Temporal",
}


def serialize_state(
    env: AlienBodyEnv, config: EnvConfig, feedback: str = "", **extra
) -> dict:
    s = env.state
    d = {
        "type": "state",
        "grid": [list(row) for row in config.cell_colors],
        "gridSize": config.grid_size,
        "agent": {
            "row": s.agent_pos.row,
            "col": s.agent_pos.col,
            "dir": s.agent_dir,
            "color": s.agent_color,
        },
        "target": list(config.target_pos),
        "obstacles": [list(o) for o in config.obstacles],
        "phase": int(s.phase),
        "step": s.step_count,
        "phase1Steps": s.phase1_steps,
        "phase2Steps": s.phase2_steps,
        "nActions": config.n_actions,
        "phase1Budget": config.phase1_budget,
        "maxSteps": config.max_total_steps,
        "family": config.family,
        "familyName": FAMILY_NAMES.get(config.family, ""),
        "actionType": config.action_type,
        "feedback": feedback,
        "done": s.done,
        "success": s.success,
    }
    d.update(extra)
    return d


async def run_ai_episode(
    ws: WebSocket, config: EnvConfig, agent_type: str, speed: int
) -> Optional[dict]:
    """Run an AI agent, streaming each step. Returns interrupting message or None."""
    env = AlienBodyEnv(config, render_mode="text")
    obs, info = env.reset()
    await ws.send_json(
        serialize_state(env, config, f"{agent_type.title()} agent starting…", aiMode=True)
    )

    if agent_type == "oracle":
        agent = OracleAgent(config)
    elif agent_type == "systematic":
        agent = SystematicExplorer(n_actions=config.n_actions)
    elif agent_type == "bayesian":
        agent = BayesianExplorer(n_actions=config.n_actions)
    elif agent_type == "memory":
        agent = MemoryAugmentedExplorer(n_actions=config.n_actions)
    else:
        agent = RandomAgent(n_actions=config.n_actions, seed=random.randint(0, 99999))
    agent.reset()

    while not env.done:
        await asyncio.sleep(speed / 1000.0)

        # Non-blocking check for cancellation
        try:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=0.02)
            if msg.get("type") in ("stop_ai", "new_game", "watch_ai"):
                return msg
        except asyncio.TimeoutError:
            pass

        prev_pos = (env.state.agent_pos.row, env.state.agent_pos.col)
        action = agent.act(obs, info)

        if action == DONE_EXPLORING:
            if env.phase == Phase.CALIBRATION:
                obs = env.start_phase2()
                info = env._get_info()
                if hasattr(agent, "set_target"):
                    agent.set_target(config.target_pos)
                await ws.send_json(
                    serialize_state(env, config, "AI ended exploration.", aiMode=True)
                )
                continue
            else:
                action = 0

        obs, reward, term, trunc, info = env.step(action)

        # Record observations for agents that need it
        if hasattr(agent, "record_step"):
            agent.record_step(action, prev_pos, info["agent_pos"])
        if info["phase"] == 2 and hasattr(agent, "set_target"):
            if hasattr(agent, "_target") and agent._target is None:
                agent.set_target(config.target_pos)

        feedback = obs.get("feedback", "")
        await ws.send_json(
            serialize_state(
                env, config, feedback,
                aiMode=True, aiAction=action, aiAgent=agent_type,
            )
        )

    return None


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def websocket_game(ws: WebSocket):
    await ws.accept()
    env: Optional[AlienBodyEnv] = None
    config: Optional[EnvConfig] = None
    pending: Optional[dict] = None

    try:
        while True:
            if pending:
                data = pending
                pending = None
            else:
                data = await ws.receive_json()
            t = data.get("type")

            if t == "new_game":
                family = data.get("family", 1)
                grid_size = data.get("gridSize", 8)
                seed = data.get("seed") or random.randint(0, 999999)
                config = generate_env(
                    family=family, idx=0, split="play",
                    seed=seed, grid_size=grid_size,
                )
                env = AlienBodyEnv(config, render_mode="text")
                env.reset()
                msg = serialize_state(env, config, seed=seed)
                await ws.send_json(msg)

            elif t == "action" and env and config:
                if env.done:
                    continue
                action = data.get("action", 0)
                obs, reward, term, trunc, info = env.step(action)
                feedback = obs.get("feedback", "")
                await ws.send_json(serialize_state(env, config, feedback))

            elif t == "done_exploring" and env and config:
                if env.state.phase == Phase.CALIBRATION:
                    env.start_phase2()
                    await ws.send_json(
                        serialize_state(env, config, "Phase 2 started! Navigate to the target.")
                    )

            elif t == "watch_ai" and config:
                agent_type = data.get("agent", "random")
                speed = data.get("speed", 400)
                pending = await run_ai_episode(ws, config, agent_type, speed)

            elif t == "stop_ai":
                pass

    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
