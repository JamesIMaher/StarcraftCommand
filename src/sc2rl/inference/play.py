"""CLI entrypoint to run a trained MaskablePPO checkpoint against a live game.

Usage:
    python -m sc2rl.inference.play --checkpoint checkpoints/final_model --episodes 5
"""

from __future__ import annotations

import argparse

from sb3_contrib import MaskablePPO

from ..config import Config
from ..env.sc2_env_wrapper import SC2FightEnv


def play(config: Config, checkpoint: str, episodes: int, deterministic: bool = True) -> None:
    env = SC2FightEnv(config.env)
    model = MaskablePPO.load(checkpoint)

    wins = 0
    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
        if total_reward > 0:
            wins += 1
        print(f"Episode {episode + 1}/{episodes}: reward={total_reward:.3f}")

    env.close()
    print(f"Finished {episodes} episodes, {wins} won.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained MaskablePPO checkpoint against StarCraft II")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved MaskablePPO .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", help="Render the game window")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of taking the argmax")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.visualize:
        config.env.visualize = True

    play(config, args.checkpoint, args.episodes, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
