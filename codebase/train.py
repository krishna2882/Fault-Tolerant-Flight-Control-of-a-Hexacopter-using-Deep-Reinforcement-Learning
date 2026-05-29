"""
train.py
========
Trains a PPO agent on the HexacopterEnv with Dynamic Curriculum Learning.
Uses Stable-Baselines3 (SB3) for the PPO implementation.

Usage:
    python train.py

Outputs:
    models/ppo_hexacopter_final.zip   – trained model
    models/ppo_hexacopter_best/       – best checkpoint
    logs/                              – TensorBoard logs
"""

import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from hexacopter_env import HexacopterEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────
TOTAL_TIMESTEPS   = 1_500_000
N_ENVS            = 6          # parallel environments
EVAL_FREQ         = 20_000     # evaluate every N steps (per env)
N_EVAL_EPISODES   = 20

# PPO hyperparameters (tuned for continuous drone control)
PPO_KWARGS = dict(
    policy          = "MlpPolicy",
    n_steps         = 2048,
    batch_size      = 512,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.005,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    learning_rate   = 3e-4,
    policy_kwargs   = dict(net_arch=[256, 256]),
    verbose         = 1,
    tensorboard_log = "logs/",
    device          = "cpu",
)

# Curriculum: probability of [0,1,2] failures
CURRICULUM_PROBS  = [0.33, 0.34, 0.33]


class CurriculumCallback(BaseCallback):
    """
    Adjusts curriculum probabilities during training.
    Phase 1 (0-33%):  mostly easy (0 failures)
    Phase 2 (33-66%): balanced (0/1 failures)
    Phase 3 (66-100%): hard  (1/2 failures)
    """

    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        if progress < 0.33:
            probs = [0.60, 0.35, 0.05]
        elif progress < 0.66:
            probs = [0.33, 0.40, 0.27]
        else:
            probs = [0.20, 0.45, 0.35]

        # Update all parallel environments (works for both DummyVecEnv and SubprocVecEnv)
        try:
            for env in self.training_env.envs:
                env.env.curriculum_probs = probs   # unwrap Monitor → HexacopterEnv
        except AttributeError:
            pass   # SubprocVecEnv - curriculum update skipped (not critical)

        return True


def make_env(seed=0, num_failed=None, probs=None):
    def _init():
        env = HexacopterEnv(
            num_failed_motors=num_failed,
            max_steps=500,
            curriculum_probs=probs or CURRICULUM_PROBS,
            seed=seed,
        )
        env = Monitor(env)
        return env
    return _init


def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    print("=" * 60)
    print("  Hexacopter FTC — PPO Training")
    print("  Total timesteps : {:,}".format(TOTAL_TIMESTEPS))
    print("  Parallel envs   : {}".format(N_ENVS))
    print("=" * 60)

    # ── Build vectorised training environment ─────────────────────────────
    train_env = DummyVecEnv(
        [make_env(seed=i) for i in range(N_ENVS)]
    )

    # ── Build deterministic eval environments (one per failure mode) ──────
    eval_env_1 = DummyVecEnv([lambda: Monitor(HexacopterEnv(num_failed_motors=1, max_steps=500))])

    # ── Callbacks ─────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq       = 100_000 // N_ENVS,
        save_path       = "models/checkpoints/",
        name_prefix     = "ppo_hex",
    )
    eval_cb = EvalCallback(
        eval_env_1,                        # evaluate on 1-failure scenario
        best_model_save_path = "models/ppo_hexacopter_best/",
        log_path             = "logs/eval/",
        eval_freq            = EVAL_FREQ // N_ENVS,
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        render               = False,
    )
    curriculum_cb = CurriculumCallback(TOTAL_TIMESTEPS)

    # ── Build model ───────────────────────────────────────────────────────
    model = PPO(env=train_env, **PPO_KWARGS)

    print("\nStarting training …\n")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, eval_cb, curriculum_cb],
        progress_bar    = True,
    )

    # ── Save final model ──────────────────────────────────────────────────
    model.save("models/ppo_hexacopter_final")
    print("\n✓ Model saved → models/ppo_hexacopter_final.zip")

    train_env.close()
    return model


if __name__ == "__main__":
    train()