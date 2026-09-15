"""CLI entrypoint to train a MaskablePPO agent on SC2FightEnv.

Usage:
    python -m sc2rl.training.train --config configs/default.yaml
    python -m sc2rl.training.train --config configs/default.yaml --resume-from checkpoints/sc2rl_50000_steps.zip
    python -m sc2rl.training.train --config configs/default.yaml --bc-dataset demonstrations.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ..config import Config
from ..env.sc2_env_wrapper import SC2FightEnv
from .behavior_cloning import pretrain_with_behavior_cloning
from .callbacks import build_callbacks


def _make_env(config: Config):
    def _init():
        return Monitor(SC2FightEnv(config.env))

    return _init


def train(
    config: Config,
    resume_from: str | None = None,
    bc_dataset: str | None = None,
    bc_epochs: int = 10,
) -> None:
    if resume_from and bc_dataset:
        raise ValueError(
            "--resume-from and --bc-dataset are mutually exclusive: a resumed checkpoint already has "
            "trained weights, so there is nothing meaningful for BC pretraining to initialize."
        )

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

        if bc_dataset:
            print(f"Pretraining via behavior cloning on {bc_dataset} ...")
            data = np.load(bc_dataset)
            expected_obs_dim = vec_env.observation_space.shape[0]
            dataset_obs_dim = data["observations"].shape[1]
            if dataset_obs_dim != expected_obs_dim:
                raise ValueError(
                    f"BC dataset {bc_dataset} has {dataset_obs_dim}-dim observations but the current "
                    f"environment produces {expected_obs_dim}-dim observations -- the observation "
                    "features or grid size changed since it was collected. Re-collect it with "
                    "`python -m sc2rl.training.collect_demonstrations` against the current config."
                )
            pretrain_with_behavior_cloning(
                model, data["observations"], data["actions"], data["masks"], epochs=bc_epochs,
            )

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
    parser.add_argument(
        "--bc-dataset",
        default=None,
        help="Path to a .npz dataset (from `python -m sc2rl.training.collect_demonstrations`) to "
             "pretrain the policy on via behavior cloning before RL fine-tuning starts. Mutually "
             "exclusive with --resume-from.",
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=10, help="Number of behavior-cloning epochs over --bc-dataset",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    train(config, resume_from=args.resume_from, bc_dataset=args.bc_dataset, bc_epochs=args.bc_epochs)


if __name__ == "__main__":
    main()
