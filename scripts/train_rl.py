#!/usr/bin/env python3
"""Train RL agents on AlienBody environments.

Algorithms:
  ppo       — PPO with CNN policy (vanilla baseline)
  ppo-icm   — PPO + count-based curiosity bonus
  ppo-rnd   — PPO + Random Network Distillation exploration
  dqn       — DQN with CNN encoder + replay buffer
  meta-rl   — RL² (LSTM policy trained across episodes, learns to learn)

Usage:
    python scripts/train_rl.py --algo ppo --family 1 --steps 200000
    python scripts/train_rl.py --algo dqn --family all --steps 500000
    python scripts/train_rl.py --algo ppo-rnd --family 2 --steps 300000
    python scripts/train_rl.py --algo meta-rl --family 1 --steps 500000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Environment Utilities ─────────────────────────────────────────

def make_vec_env(family: int, n_envs: int = 4, seed: int = 0, wrapper=None):
    """Create vectorized AlienBody environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    from alienbody.env.gym_wrapper import AlienBodyGymEnv

    def make_env(env_seed):
        def _init():
            env = AlienBodyGymEnv(family=family, cell_px=8, seed=env_seed)
            if wrapper:
                env = wrapper(env)
            return env
        return _init

    return DummyVecEnv([make_env(seed + i) for i in range(n_envs)])


# ── PPO (Vanilla) ─────────────────────────────────────────────────

def train_ppo(family: int, total_steps: int, output_dir: Path):
    from stable_baselines3 import PPO

    env = make_vec_env(family, n_envs=4)
    model = PPO(
        "CnnPolicy", env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    print(f"[PPO] Training on Family {family} for {total_steps} steps")
    model.learn(total_timesteps=total_steps)

    path = output_dir / f"ppo_family{family}.zip"
    model.save(str(path))
    print(f"  Saved: {path}")
    return model


# ── PPO + ICM (Count-Based Curiosity) ─────────────────────────────

import gymnasium

class ICMWrapper(gymnasium.Wrapper):
    """Count-based curiosity reward (lightweight ICM approximation)."""

    def __init__(self, env, beta: float = 0.2):
        super().__init__(env)
        self._counts = {}
        self._beta = beta

    def reset(self, **kw):
        self._counts = {}
        return self.env.reset(**kw)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        key = obs.tobytes()[:128]
        count = self._counts.get(key, 0)
        intrinsic = self._beta / (count + 1) ** 0.5
        self._counts[key] = count + 1
        return obs, reward + intrinsic, term, trunc, info

    def render(self): return self.env.render()
    def close(self): return self.env.close()


def train_ppo_icm(family: int, total_steps: int, output_dir: Path):
    from stable_baselines3 import PPO

    env = make_vec_env(family, n_envs=4, wrapper=ICMWrapper)
    model = PPO(
        "CnnPolicy", env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    print(f"[PPO+ICM] Training on Family {family} for {total_steps} steps")
    model.learn(total_timesteps=total_steps)

    path = output_dir / f"icm_family{family}.zip"
    model.save(str(path))
    print(f"  Saved: {path}")
    return model


# ── PPO + RND (Random Network Distillation) ───────────────────────

class RNDWrapper(gymnasium.Wrapper):
    """Random Network Distillation exploration bonus.

    Two networks: fixed random target f(s) and trainable predictor f̂(s).
    Intrinsic reward = ||f(s) - f̂(s)||² — high for novel states.
    """

    def __init__(self, env, feature_dim: int = 64, lr: float = 1e-3, beta: float = 0.3):
        super().__init__(env)
        import torch
        import torch.nn as nn

        self._beta = beta

        obs_shape = env.observation_space.shape
        flat_dim = np.prod(obs_shape[:2]) // 64

        # Random target network (fixed)
        self._target = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), feature_dim, bias=False),
        )
        for p in self._target.parameters():
            p.requires_grad = False

        # Predictor network (trainable)
        self._predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )
        self._opt = torch.optim.Adam(self._predictor.parameters(), lr=lr)
        self._torch = torch

    def reset(self, **kw):
        return self.env.reset(**kw)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        # Compute RND intrinsic reward
        with self._torch.no_grad():
            obs_t = self._torch.from_numpy(obs.flatten().astype(np.float32)).unsqueeze(0) / 255.0
            target_feat = self._target(obs_t)

        pred_feat = self._predictor(obs_t)
        rnd_loss = ((target_feat - pred_feat) ** 2).mean()
        intrinsic = rnd_loss.item() * self._beta

        # Train predictor
        self._opt.zero_grad()
        rnd_loss.backward()
        self._opt.step()

        return obs, reward + intrinsic, term, trunc, info

    def render(self): return self.env.render()
    def close(self): return self.env.close()


def train_ppo_rnd(family: int, total_steps: int, output_dir: Path):
    from stable_baselines3 import PPO

    env = make_vec_env(family, n_envs=4, wrapper=RNDWrapper)
    model = PPO(
        "CnnPolicy", env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    print(f"[PPO+RND] Training on Family {family} for {total_steps} steps")
    model.learn(total_timesteps=total_steps)

    path = output_dir / f"rnd_family{family}.zip"
    model.save(str(path))
    print(f"  Saved: {path}")
    return model


# ── DQN ───────────────────────────────────────────────────────────

def train_dqn(family: int, total_steps: int, output_dir: Path):
    from stable_baselines3 import DQN

    env = make_vec_env(family, n_envs=1)  # DQN uses single env
    model = DQN(
        "CnnPolicy", env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    print(f"[DQN] Training on Family {family} for {total_steps} steps")
    model.learn(total_timesteps=total_steps)

    path = output_dir / f"dqn_family{family}.zip"
    model.save(str(path))
    print(f"  Saved: {path}")
    return model


# ── Meta-RL (RL²) ─────────────────────────────────────────────────

def train_meta_rl(family: int, total_steps: int, output_dir: Path):
    """Train an RL² agent with LSTM policy.

    Key idea: the LSTM hidden state serves as implicit memory of what
    the agent has learned about action effects within the current episode.
    By training across many randomized episodes, the LSTM learns a general
    strategy for action calibration.

    Architecture:
        obs_image → CNN → features (128)
        [features, phase, step_norm, prev_action, prev_reward] → LSTM(256) → action logits
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    from alienbody.env.gym_wrapper import AlienBodyGymEnv

    class MetaRLPolicy(nn.Module):
        def __init__(self, obs_shape, n_actions: int, hidden_dim: int = 256):
            super().__init__()
            self.n_actions = n_actions

            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(32 * 16, 128), nn.ReLU(),
            )

            # LSTM input: CNN features + (phase, step_norm)
            self.lstm = nn.LSTM(128 + 2, hidden_dim, batch_first=True)
            self.actor = nn.Linear(hidden_dim, n_actions + 1)  # +1 for DONE
            self.critic = nn.Linear(hidden_dim, 1)

        def forward(self, img, extra, hidden=None):
            """
            img: (B, 3, H, W)
            extra: (B, 2) — [phase, step_norm]
            hidden: (h, c) LSTM state
            """
            feat = self.cnn(img)  # (B, 128)
            x = torch.cat([feat, extra], dim=-1).unsqueeze(1)  # (B, 1, 130)

            if hidden is None:
                out, hidden = self.lstm(x)
            else:
                out, hidden = self.lstm(x, hidden)

            out = out.squeeze(1)  # (B, 256)
            logits = self.actor(out)
            value = self.critic(out)
            return logits, value, hidden

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_shape = (3, 128, 128)
    n_actions = 4

    policy = MetaRLPolicy(obs_shape, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    n_episodes = total_steps // 50  # approx episodes
    gamma = 0.99
    eps_clip = 0.2

    print(f"[Meta-RL] Training on Family {family} for ~{n_episodes} episodes")

    for ep in range(n_episodes):
        env = AlienBodyGymEnv(family=family, cell_px=8, seed=ep)
        obs, info = env.reset()

        log_probs = []
        values = []
        rewards = []
        hidden = None

        done = False
        while not done:
            obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            phase = info.get("phase", 1)
            step = info.get("step_count", 0)
            extra = torch.tensor([[phase, step / 50.0]], dtype=torch.float32, device=device)

            logits, value, hidden = policy(obs_t, extra, hidden)
            # Detach hidden to prevent BPTT across full episode
            hidden = (hidden[0].detach(), hidden[1].detach())

            dist = Categorical(logits=logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze())

            action_int = action.item()
            if action_int == n_actions:
                from alienbody.env.grid import Phase
                if hasattr(env, '_env') and env._env.phase == Phase.CALIBRATION:
                    step_result = env.step(action_int)
                    obs, info = step_result[0], step_result[-1]
                    rewards.append(0.0)
                    continue
                action_int = 0

            obs, reward, terminated, truncated, info = env.step(action_int)
            rewards.append(reward)
            done = terminated or truncated

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        if not log_probs:
            continue

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO-style loss
        policy_loss = -(log_probs * advantages).mean()
        value_loss = ((values - returns) ** 2).mean()
        entropy_loss = -Categorical(logits=logits).entropy().mean()

        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"reward={sum(rewards):.2f} steps={len(rewards)} "
                  f"loss={loss.item():.4f}")

    # Save
    path = output_dir / f"meta_rl_family{family}.pt"
    torch.save(policy, str(path))
    print(f"  Saved: {path}")
    return policy


# ── Main ──────────────────────────────────────────────────────────

ALGO_MAP = {
    "ppo": train_ppo,
    "ppo-icm": train_ppo_icm,
    "ppo-rnd": train_ppo_rnd,
    "dqn": train_dqn,
    "meta-rl": train_meta_rl,
}


def main():
    parser = argparse.ArgumentParser(description="Train RL agents on AlienBody")
    parser.add_argument("--algo", choices=list(ALGO_MAP.keys()), default="ppo",
                        help="Algorithm to train")
    parser.add_argument("--family", type=str, default="1",
                        help="Family (1-6 or 'all')")
    parser.add_argument("--steps", type=int, default=200000,
                        help="Total training steps/timesteps")
    parser.add_argument("--output", type=str, default="models",
                        help="Output directory for saved models")
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    families = list(range(1, 7)) if args.family == "all" else [int(args.family)]
    train_fn = ALGO_MAP[args.algo]

    for family in families:
        print(f"\n{'='*50}")
        train_fn(family, args.steps, output_dir)
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
