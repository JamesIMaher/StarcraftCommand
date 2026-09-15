"""Training callbacks: periodic checkpointing, plus a reward-component
breakdown printed to the console (and logged to TensorBoard) every episode.

A hold-out MaskableEvalCallback is deliberately not wired up here -- it
would require a second live SC2 client running concurrently with the
training env, which is significant overhead for a single-machine setup. Add
one later by pointing `sb3_contrib.common.maskable.callbacks.MaskableEvalCallback`
at a second SC2FightEnv instance if periodic hold-out evaluation becomes
worth the cost.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from ..config import Config

_REWARD_COMPONENTS = (
    "reward_terminal",
    "reward_economic",
    "reward_kill",
    "reward_home_defense",
    "reward_scouting",
    "reward_exploration",
    "reward_stale_search",
    "reward_time",
    "reward_approach",
)


class RewardBreakdownCallback(BaseCallback):
    """SC2FightEnv.step() returns each reward component separately in its
    info dict; SB3's own console/TensorBoard output only shows the single
    summed reward, so an imbalance between components (e.g. kill-value
    outweighing a loss) is otherwise invisible without re-deriving it from
    formulas. This accumulates each component into a per-episode sum and
    prints/logs it the moment an episode ends (detected via Monitor's own
    "episode" info key), one line per env per episode.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_totals: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._episode_totals = [dict.fromkeys(_REWARD_COMPONENTS, 0.0) for _ in range(n_envs)]

    def _on_step(self) -> bool:
        for env_idx, info in enumerate(self.locals.get("infos", [])):
            totals = self._episode_totals[env_idx]
            for key in _REWARD_COMPONENTS:
                if key in info:
                    totals[key] += info[key]

            if "episode" in info:
                ep_reward = info["episode"]["r"]
                breakdown = " ".join(f"{k.removeprefix('reward_')}={v:+.3f}" for k, v in totals.items())
                print(f"[episode end] total={ep_reward:+.3f}  {breakdown}")
                for key, value in totals.items():
                    self.logger.record(f"reward_breakdown/{key}", value)
                self._episode_totals[env_idx] = dict.fromkeys(_REWARD_COMPONENTS, 0.0)
        return True


def build_callbacks(config: Config) -> CallbackList:
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.checkpoint_freq,
        save_path=config.training.checkpoint_dir,
        name_prefix="sc2rl",
    )
    return CallbackList([checkpoint_callback, RewardBreakdownCallback()])
