"""CLI entrypoint to train a MaskablePPO agent on SC2FightEnv.

Usage:
    python -m sc2rl.training.train --config configs/default.yaml
    python -m sc2rl.training.train --config configs/default.yaml --resume-from checkpoints/sc2rl_50000_steps.zip
"""

from __future__ import annotations

import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..config import Config
from ..env.sc2_env_wrapper import SC2FightEnv
from .callbacks import build_callbacks


def _make_env(config: Config):
    def _init():
        return Monitor(SC2FightEnv(config.env))

    return _init


def train(config: Config, resume_from: str | None = None) -> None:
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.tensorboard_log, exist_ok=True)

    vec_env = DummyVecEnv([_make_env(config)])
    ppo = config.training.ppo

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model = MaskablePPO.load(resume_from, env=vec_env, tensorboard_log=config.training.tensorboard_log)
        reset_num_timesteps = False
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=ppo.learning_rate,
            n_steps=ppo.n_steps,
            batch_size=ppo.batch_size,
            n_epochs=ppo.n_epochs,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            clip_range=ppo.clip_range,
            ent_coef=ppo.ent_coef,
            seed=ppo.seed,
            tensorboard_log=config.training.tensorboard_log,
            verbose=1,
        )
        reset_num_timesteps = True

    # NOTE: with reset_num_timesteps=False (i.e. --resume-from was given),
    # SB3 treats total_timesteps as an ADDITIONAL budget on top of the
    # checkpoint's own progress, not an absolute target -- so this always
    # trains for ppo.total_timesteps more steps from wherever the checkpoint
    # left off, however many times you resume.
    model.learn(
        total_timesteps=ppo.total_timesteps,
        callback=build_callbacks(config),
        reset_num_timesteps=reset_num_timesteps,
    )

    final_path = os.path.join(config.training.checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a MaskablePPO StarCraft II fight-mission agent")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a saved .zip checkpoint to resume training from (policy/optimizer state and "
             "hyperparameters are restored; trains for another ppo.total_timesteps steps from there)",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
