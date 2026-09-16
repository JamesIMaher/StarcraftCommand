"""CLI entrypoint to run a trained MaskablePPO checkpoint against a live game
with a local web command console: type or speak an instruction (e.g. "return
the marines to base") and it interrupts autonomous play for one action, then
hands control straight back to the trained policy. See play.py for plain
(non-interactive) inference -- that entrypoint is untouched by this one.

Usage:
    python -m sc2rl.inference.interactive_play --checkpoint checkpoints/final_model

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import os
import queue

import anthropic
from sb3_contrib import MaskablePPO

from ..command.interpreter import interpret_command
from ..command.server import PendingCommand, run_server
from ..config import Config
from ..env.sc2_env_wrapper import SC2FightEnv

DEFAULT_PORT = 8765


def play(config: Config, checkpoint: str, episodes: int, port: int, deterministic: bool = True) -> None:
    env = SC2FightEnv(config.env)
    model = MaskablePPO.load(checkpoint)
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    command_queue: "queue.Queue[PendingCommand]" = queue.Queue()
    run_server(command_queue, port)
    print(f"Command console: http://127.0.0.1:{port}")

    wins = 0
    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action_masks = env.action_masks()

            try:
                pending = command_queue.get_nowait()
            except queue.Empty:
                pending = None

            if pending is not None:
                # Nothing calls env.step() while this resolves, so the game
                # does not advance underneath the command -- the bot-mode
                # client simply waits for the next step request.
                result = interpret_command(
                    client, pending.text, env.state, env.action_spec, action_masks,
                    env.orientation, env.mobilized, env.config.garrison_size,
                )
                print(f"[command] {pending.text!r} -> {result.action_name}: {result.message}")
                pending.result = {
                    "action_index": result.action_index,
                    "action_name": result.action_name,
                    "message": result.message,
                }
                pending.done.set()
                if result.action_index is None:
                    continue  # declined -- doesn't consume a game step
                action = result.action_index
            else:
                predicted, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
                action = int(predicted)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        if total_reward > 0:
            wins += 1
        print(f"Episode {episode + 1}/{episodes}: reward={total_reward:.3f}")

    env.close()
    print(f"Finished {episodes} episodes, {wins} won.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained MaskablePPO checkpoint with a live command console")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved MaskablePPO .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", help="Render the game window")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of taking the argmax")
    parser.add_argument("--command-port", type=int, default=DEFAULT_PORT, help="Local port for the command console")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set -- the command console needs it to reach Claude.")

    config = Config.from_yaml(args.config)
    if args.visualize:
        config.env.visualize = True

    play(config, args.checkpoint, args.episodes, args.command_port, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
