"""RL agent wrapper for evaluation — loads trained SB3/custom models.

Supports: PPO, DQN (via stable-baselines3), and Meta-RL (custom LSTM).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from alienbody.agents import Agent, DONE_EXPLORING


class SB3Agent(Agent):
    """Wraps a trained stable-baselines3 model for AlienBody evaluation.

    Converts AlienBody observations to the format expected by the trained
    model (RGB image → CNN input), then uses model.predict() for actions.
    """

    def __init__(self, model_path: str, n_actions: int = 4, cell_px: int = 8):
        self.model_path = model_path
        self.n_actions = n_actions
        self.cell_px = cell_px
        self._model = None
        self._algo_name = Path(model_path).stem.split("_")[0]

    @property
    def name(self) -> str:
        return f"RL/{self._algo_name}"

    def reset(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        try:
            from stable_baselines3 import PPO, DQN
        except ImportError:
            raise ImportError("pip install stable-baselines3 torch")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        algo_name = self._algo_name.lower()
        if "dqn" in algo_name:
            self._model = DQN.load(str(path))
        else:
            self._model = PPO.load(str(path))

    def act(self, observation: dict, info: dict | None = None) -> int:
        if self._model is None:
            self._load_model()

        img = observation.get("image")
        if img is None:
            return 0

        # Resize to match training resolution
        img = self._resize(img)

        action, _ = self._model.predict(img, deterministic=True)
        action = int(action)

        # Action 4 = DONE_EXPLORING in the gym wrapper
        if action == self.n_actions:
            return DONE_EXPLORING

        return min(action, self.n_actions - 1)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize observation to match training image size."""
        expected = 16 * self.cell_px
        if img.shape[0] != expected or img.shape[1] != expected:
            from PIL import Image
            pil = Image.fromarray(img).resize((expected, expected), Image.NEAREST)
            return np.array(pil)
        return img


class MetaRLAgent(Agent):
    """Meta-RL (RL²) agent with LSTM policy.

    The LSTM maintains a hidden state across steps within an episode,
    allowing the agent to "learn to learn" — adapting its behavior based
    on observed action effects without explicit re-training.

    Architecture:
        obs (image) → CNN encoder → [LSTM hidden state] → action logits

    The LSTM hidden state serves as implicit working memory of the
    action-effect model learned during the episode.
    """

    def __init__(self, model_path: str, n_actions: int = 4, cell_px: int = 8):
        self.model_path = model_path
        self.n_actions = n_actions
        self.cell_px = cell_px
        self._model = None
        self._hidden: Optional[tuple] = None

    @property
    def name(self) -> str:
        return "MetaRL/RL2"

    def reset(self):
        if self._model is None:
            self._load_model()
        self._hidden = None

    def _load_model(self):
        try:
            import torch
        except ImportError:
            raise ImportError("pip install torch")

        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Meta-RL model not found: {path}")

        self._model = torch.load(str(path), map_location="cpu", weights_only=False)
        self._model.eval()

    def act(self, observation: dict, info: dict | None = None) -> int:
        import torch

        img = observation.get("image")
        if img is None or self._model is None:
            return 0

        img = self._resize(img)
        img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        phase = observation.get("phase", 1)
        step = observation.get("step", 0)
        extra = torch.tensor([[phase, step / 50.0]], dtype=torch.float32)

        with torch.no_grad():
            logits, self._hidden = self._model(img_t, extra, self._hidden)
            action = logits.argmax(dim=-1).item()

        if action == self.n_actions:
            return DONE_EXPLORING
        return min(action, self.n_actions - 1)

    def _resize(self, img: np.ndarray) -> np.ndarray:
        expected = 16 * self.cell_px
        if img.shape[0] != expected or img.shape[1] != expected:
            from PIL import Image
            pil = Image.fromarray(img).resize((expected, expected), Image.NEAREST)
            return np.array(pil)
        return img
