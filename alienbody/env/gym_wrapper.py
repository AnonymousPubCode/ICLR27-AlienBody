"""Gymnasium-compatible wrapper for AlienBody.

Enables training RL agents (PPO, ICM) with stable-baselines3.
Registers as gymnasium environment: AlienBody-v0.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("pip install gymnasium")

from alienbody.env.core import AlienBodyEnv
from alienbody.env.grid import EnvConfig, Phase
from alienbody.env.generator import generate_env


class AlienBodyGymEnv(gym.Env):
    """Gymnasium wrapper for AlienBody benchmark.

    Observation: RGB image (grid_size*cell_px, grid_size*cell_px, 3)
    Action: Discrete(n_actions + 1) where last action = DONE_EXPLORING
    Reward: +1 for reaching target, 0 otherwise
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        family: int = 1,
        split: str = "train",
        cell_px: int = 8,  # smaller for RL (saves memory)
        render_mode: str = "rgb_array",
        randomize_env: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.family = family
        self.split = split
        self.cell_px = cell_px
        self.render_mode = render_mode
        self.randomize_env = randomize_env
        self._seed = seed
        self._episode_count = 0

        # Spaces
        gs = 16  # default grid size
        img_size = gs * cell_px
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(img_size, img_size, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(5)  # 0-3 actions + 4 = DONE_EXPLORING

        self._env: Optional[AlienBodyEnv] = None
        self._config: Optional[EnvConfig] = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Generate a new environment for each episode (if randomize)
        if self.randomize_env:
            env_seed = (self._seed + self._episode_count) * 31337
        else:
            env_seed = self._seed

        self._config = generate_env(
            family=self.family, idx=self._episode_count % 50,
            split=self.split, seed=env_seed
        )
        self._env = AlienBodyEnv(self._config, render_mode="image")
        obs, info = self._env.reset()
        self._episode_count += 1

        img = self._resize_obs(obs["image"])
        return img, info

    def step(self, action: int):
        if action == 4:
            # DONE_EXPLORING signal
            if self._env.phase == Phase.CALIBRATION:
                obs = self._env.start_phase2()
                info = self._env._get_info()
                img = self._resize_obs(obs["image"])
                return img, 0.0, False, False, info

        # Clamp action to valid range
        action = min(action, self._config.n_actions - 1)
        obs, reward, terminated, truncated, info = self._env.step(action)
        img = self._resize_obs(obs["image"])
        return img, reward, terminated, truncated, info

    def render(self):
        if self._env and self._env.state:
            from alienbody.env.renderer import render_image
            return render_image(self._env.state, self._config,
                                show_target=self._env.phase == Phase.EXECUTION,
                                cell_px=self.cell_px)
        return None

    def _resize_obs(self, img: np.ndarray) -> np.ndarray:
        """Resize image if cell_px differs from renderer default."""
        expected = 16 * self.cell_px
        if img.shape[0] != expected:
            from PIL import Image
            pil = Image.fromarray(img).resize((expected, expected), Image.NEAREST)
            return np.array(pil)
        return img


# Register with gymnasium
try:
    gym.register(
        id="AlienBody-v0",
        entry_point="alienbody.env.gym_wrapper:AlienBodyGymEnv",
        kwargs={"family": 1},
    )
except gym.error.Error:
    pass  # already registered
