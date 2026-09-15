"""CLI to run the scripted policy (env/scripted_policy.py) live and save an
(observation, action, action_mask) dataset for behavior-cloning pretraining.

Usage:
    python -m sc2rl.training.collect_demonstrations --config configs/default.yaml --episodes 20 --out demonstrations.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import Config
from ..env.sc2_env_wrapper import SC2FightEnv
from ..env.scripted_policy import ScriptedPolicy, ScriptedPolicyConfig


def collect(config: Config, episodes: int, env_factory=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = SC2FightEnv(config.env) if env_factory is None else SC2FightEnv(config.env, env_factory=env_factory)
    policy = ScriptedPolicy(ScriptedPolicyConfig())
    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []

    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            policy.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                mask = env.action_masks()
                action = policy.action(env.state, env.action_spec, env.masking_config, env.orientation)
                observations.append(obs)
                actions.append(action)
                masks.append(mask)
                obs, _reward, terminated, truncated, _info = env.step(action)
            print(f"episode {episode + 1}/{episodes} collected -- dataset size so far: {len(observations)}")
    finally:
        env.close()

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(masks, dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect scripted-policy demonstrations for BC pretraining")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", default="demonstrations.npz")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    observations, actions, masks = collect(config, args.episodes)
    np.savez_compressed(args.out, observations=observations, actions=actions, masks=masks)
    print(f"Saved {len(observations)} (obs, action, mask) triples to {args.out}")


if __name__ == "__main__":
    main()
