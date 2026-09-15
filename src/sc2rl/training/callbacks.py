"""Training callbacks: periodic checkpointing only for v1. A hold-out
MaskableEvalCallback is deliberately not wired up here -- it would require a
second live SC2 client running concurrently with the training env, which is
significant overhead for a single-machine setup. Add one later by pointing
`sb3_contrib.common.maskable.callbacks.MaskableEvalCallback` at a second
SC2FightEnv instance if periodic hold-out evaluation becomes worth the cost.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from ..config import Config


def build_callbacks(config: Config) -> CallbackList:
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.checkpoint_freq,
        save_path=config.training.checkpoint_dir,
        name_prefix="sc2rl",
    )
    return CallbackList([checkpoint_callback])
