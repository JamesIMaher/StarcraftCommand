"""CLI entrypoint to run a trained MaskablePPO checkpoint against a live game
with a local web command console: type or speak an instruction (e.g. "return
the marines to base") and it interrupts autonomous play, then hands control
straight back to the trained policy. See play.py for plain (non-interactive)
inference -- that entrypoint is untouched by this one.

Beyond one-shot actions, the console also supports:
- dispatching specific marines to a sector under persistent PLAYER control,
  excluded from the autonomous policy's own moves until released (voice/text,
  or the console's own per-unit button);
- standing directives that ban (or re-allow) the AUTONOMOUS policy from
  choosing a specific action for the rest of the game, without ever
  restricting the player's own explicit one-off commands.

Usage:
    python -m sc2rl.inference.interactive_play --checkpoint checkpoints/final_model

Requires ANTHROPIC_API_KEY, either already in the environment or in a local
.env file (see .env.example) -- load_dotenv() below picks up the latter.

With --kamiwaza-url, the console is instead the SC2 Command Console app on a
Kamiwaza instance (kamiwaza-app/ in this repo), and commands are interpreted
by the platform's own LLM (gpt-oss-120b) through that app's /v1 proxy -- no
Anthropic key needed:

    python -m sc2rl.inference.interactive_play --checkpoint checkpoints/final_model \
        --kamiwaza-url https://<kamiwaza-host>/runtime/apps/<app-name>

Requires KAMIWAZA_API_KEY (a Kamiwaza personal access token) in the
environment or .env. Set KAMIWAZA_VERIFY_SSL=false for an instance with a
self-signed certificate.
"""

from __future__ import annotations

import argparse
import os
import queue

import anthropic
import httpx
import openai
from dotenv import load_dotenv
from sb3_contrib import MaskablePPO

from ..command.interpreter import (
    ActionResult,
    DeclineResult,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    DirectiveResult,
    DispatchResult,
    ReleaseResult,
    interpret_command,
)
from ..command.relay_client import RelayClient
from ..command.server import EventLog, PendingCommand, PendingRelease, run_server
from ..config import Config
from ..env.action_space import FixedAction
from ..env.sc2_env_wrapper import SC2FightEnv

DEFAULT_PORT = 8765

# Comfortably under server.py's _COMMAND_TIMEOUT_SECONDS (30s). The
# anthropic SDK's own default (600s read timeout, 2 retries) let one slow
# or hung API call freeze the WHOLE game loop for minutes -- confirmed
# live: a command timed out in the browser (server.py's own 30s wait gave
# up), but the main loop was still blocked inside client.messages.create(),
# so no further env.step() calls happened at all and the autonomous policy
# stopped issuing any new orders (SCVs kept mining only because that's an
# already-issued SC2 order, not something our loop re-triggers each step).
# max_retries=0 too: a local single-user tool should fail fast into
# interpret_command's own exception handler (a DeclineResult) rather than
# silently retrying for up to another _CLAUDE_TIMEOUT_SECONDS on top.
_CLAUDE_TIMEOUT_SECONDS = 12.0


def _build_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(timeout=_CLAUDE_TIMEOUT_SECONDS, max_retries=0)


def _build_openai_client(base_url: str, api_key: str, verify: bool = True) -> openai.OpenAI:
    """Same fail-fast timeout/no-retry policy as _build_client, for an
    OpenAI-compatible endpoint (the Kamiwaza console app's /v1 proxy)."""
    return openai.OpenAI(
        base_url=base_url, api_key=api_key, timeout=_CLAUDE_TIMEOUT_SECONDS, max_retries=0,
        http_client=httpx.Client(verify=verify),
    )


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _apply_command_result(env: SC2FightEnv, result, pending: PendingCommand, events: EventLog):
    """Route one interpreted command to its effect on the environment.
    Returns (action, extra_calls) for the caller to pass to env.step(), or
    (None, None) when nothing should be stepped this turn -- a decline, or a
    release/directive, both pure bookkeeping with no raw game calls."""
    if isinstance(result, ActionResult):
        pending.result = {"kind": "action", "action_name": result.action_name, "message": result.message}
        return result.action_index, None

    if isinstance(result, DispatchResult):
        calls = env.assign_to_player_control(target_sector=result.target_sector, count=result.unit_count)
        pending.result = {"kind": "dispatch", "message": result.message}
        events.append("dispatch", result.message)
        return int(FixedAction.NO_OP), calls

    if isinstance(result, ReleaseResult):
        released = env.release_from_player_control(sector=result.sector, most_recent=result.most_recent)
        pending.result = {"kind": "release", "released": sorted(released), "message": result.message}
        events.append("release", result.message)
        return None, None

    if isinstance(result, DirectiveResult):
        (env.allow_action if result.allowed else env.ban_action)(result.action_name)
        pending.result = {
            "kind": "directive", "action_name": result.action_name, "allowed": result.allowed,
            "message": result.message,
        }
        events.append("directive", result.message)
        return None, None

    if isinstance(result, DeclineResult):
        pending.result = {"kind": "decline", "message": result.message}
        return None, None

    raise AssertionError(f"unhandled command result: {result!r}")


def _autonomous_action(env: SC2FightEnv, model, obs, action_masks, deterministic: bool, events: EventLog) -> int:
    """The autonomous policy's own choice for this step. With no directive
    active this is a single, ordinary predict() call -- zero overhead over
    the pre-directives behavior. With a directive active, predict TWICE:
    once against the full mask (what the policy would have chosen) and once
    against the directive-narrowed mask (what actually executes). Baking
    the ban into the sampled-from mask -- rather than letting the policy
    choose freely and vetoing the result after the fact -- is what avoids a
    real soft-lock risk: with deterministic selection, if the banned choice
    is genuinely the policy's top pick and the observation barely changes
    step to step, a post-hoc veto would just re-pick (and re-veto) the same
    banned action forever. Only log a denial when the two choices actually
    differ because of the ban -- the two forward passes are free (a 2x64
    MLP), and this only ever runs at all while a directive is active."""
    if not env.banned_actions:
        predicted, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
        return int(predicted)

    unconstrained, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
    constrained_mask = env.apply_bans(action_masks)
    constrained, _ = model.predict(obs, action_masks=constrained_mask, deterministic=deterministic)
    unconstrained, constrained = int(unconstrained), int(constrained)
    if unconstrained != constrained:
        name = env.action_spec.name(unconstrained)
        if name in env.banned_actions:
            events.append("denied", f"Wanted to {name}, but that's currently banned by a player directive.")
    return constrained


def play(
    config: Config, checkpoint: str, episodes: int, port: int,
    deterministic: bool = True, command_model: str = DEFAULT_MODEL,
    kamiwaza_url: str | None = None, kamiwaza_token: str | None = None, verify_ssl: bool = True,
) -> None:
    env = SC2FightEnv(config.env)
    model = MaskablePPO.load(checkpoint)
    if kamiwaza_url:
        client = _build_openai_client(f"{kamiwaza_url.rstrip('/')}/v1", kamiwaza_token, verify=verify_ssl)
    else:
        client = _build_client()  # reads ANTHROPIC_API_KEY from the environment

    command_queue: "queue.Queue[PendingCommand]" = queue.Queue()
    control_queue: "queue.Queue[PendingRelease]" = queue.Queue()
    events = EventLog()

    def state_snapshot() -> dict:
        return {
            "player_controlled_units": sorted(env.player_controlled_tags),
            "banned_actions": sorted(env.banned_actions),
        }

    if kamiwaza_url:
        RelayClient(
            kamiwaza_url, kamiwaza_token, command_queue, control_queue, events, state_snapshot,
            verify=verify_ssl,
        ).start()
        print(f"Command console: {kamiwaza_url.rstrip('/')}/")
    else:
        run_server(command_queue, control_queue, events, state_snapshot, port)
        print(f"Command console: http://127.0.0.1:{port}")

    wins = 0
    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action_masks = env.action_masks()

            # A UI button click never needs Claude -- handle it before
            # checking the Claude-bound queue, and without consuming a game
            # step (pure bookkeeping: releasing doesn't need a raw order,
            # the marine just stops being excluded from the next AI move).
            try:
                release = control_queue.get_nowait()
            except queue.Empty:
                release = None
            if release is not None:
                released = env.release_from_player_control(
                    tags=release.tags, sector=release.sector, most_recent=release.most_recent,
                )
                release.result = {"released": sorted(released)}
                release.done.set()
                if released:
                    events.append("release", f"Released {sorted(released)} to AI control.")
                continue

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
                    len(env.player_controlled_tags), tuple(env.banned_actions),
                    model=command_model,
                )
                print(f"[command] {pending.text!r} -> {type(result).__name__}: {result.message}")
                action, extra_calls = _apply_command_result(env, result, pending, events)
                pending.done.set()
                if action is None:
                    continue  # declined, or pure bookkeeping -- no step consumed
            else:
                action = _autonomous_action(env, model, obs, action_masks, deterministic, events)
                extra_calls = None

            obs, reward, terminated, truncated, _ = env.step(action, extra_calls=extra_calls)
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
    parser.add_argument(
        "--no-realtime", dest="realtime", action="store_false",
        help="Step the game as fast as the client can simulate instead of at true game speed. "
             "Defaults to realtime ON, since a human needs true pacing to type or speak a command in time.",
    )
    parser.set_defaults(realtime=True)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of taking the argmax")
    parser.add_argument("--command-port", type=int, default=DEFAULT_PORT, help="Local port for the command console")
    parser.add_argument(
        "--command-model", default=None,
        help=f"Model that interprets console commands (default: {DEFAULT_MODEL}, or "
             f"{DEFAULT_OPENAI_MODEL} with --kamiwaza-url -- where the app pins its own model anyway)",
    )
    parser.add_argument(
        "--kamiwaza-url", default=None,
        help="Use the SC2 Command Console app on Kamiwaza instead of the local console, e.g. "
             "https://<kamiwaza-host>/runtime/apps/<app-name>. Needs KAMIWAZA_API_KEY.",
    )
    args = parser.parse_args()

    load_dotenv()  # picks up a local .env if present; no-op (and harmless) if it isn't
    kamiwaza_token = os.environ.get("KAMIWAZA_API_KEY")
    if args.kamiwaza_url:
        if not kamiwaza_token:
            raise SystemExit(
                "KAMIWAZA_API_KEY is not set -- --kamiwaza-url needs a Kamiwaza personal access token. "
                "Put it in a .env file (see .env.example) or set it in your shell."
            )
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set -- the command console needs it to reach Claude. "
            "Put it in a .env file (see .env.example) or set it in your shell."
        )
    command_model = args.command_model or (DEFAULT_OPENAI_MODEL if args.kamiwaza_url else DEFAULT_MODEL)

    config = Config.from_yaml(args.config)
    if args.visualize:
        config.env.visualize = True
    config.env.realtime = args.realtime

    play(
        config, args.checkpoint, args.episodes, args.command_port,
        deterministic=not args.stochastic, command_model=command_model,
        kamiwaza_url=args.kamiwaza_url, kamiwaza_token=kamiwaza_token,
        verify_ssl=_env_flag("KAMIWAZA_VERIFY_SSL", default=True),
    )


if __name__ == "__main__":
    main()
