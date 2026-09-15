"""Gymnasium Env wrapping pysc2.env.sc2_env.SC2Env.

This is the ONLY module in sc2rl that talks to a live StarCraft II client.
Everything it delegates to (observation featurization, action masking, action
translation) operates on plain GameState snapshots and is fully unit-testable
without a live game -- see tests/test_env_wrapper.py, which exercises this
class's step()/reset() contract against a stubbed SC2Env instead of a real one.
"""

from __future__ import annotations

import sys

# NOTE: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION is set in sc2rl/__init__.py,
# which runs before this module -- see that file for why. Anything importing
# pysc2 directly (bypassing `sc2rl`) must set it first itself.

import numpy as np
import gymnasium as gym
from absl import flags
from gymnasium import spaces
from pysc2.env import sc2_env
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

from ..config import EnvConfig
from .action_masking import MaskingConfig, compute_action_masks

# pysc2's run_configs module reads absl flags (e.g. --sc2_run_config) and
# raises if they were never parsed. Our CLI entrypoints use argparse, not
# absl.app.run(), so nothing parses them otherwise -- parse with just the
# program name (no extra args) so pysc2's internals are satisfied without
# absl trying to interpret our own argparse flags.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv[:1])
from .action_space import ActionSpaceSpec, FixedAction
from .action_translation import ActionTranslator
from .game_state import GameState
from .observation import featurize, observation_length
from .sector_grid import SectorGrid, SpawnOrientation

_RACE_MAP = {
    "terran": sc2_env.Race.terran,
    "protoss": sc2_env.Race.protoss,
    "zerg": sc2_env.Race.zerg,
    "random": sc2_env.Race.random,
}

_HOME_SECTOR = 0  # canonical sector nearest home after SpawnOrientation mirroring

_DIFFICULTY_MAP = {
    "very_easy": sc2_env.Difficulty.very_easy,
    "easy": sc2_env.Difficulty.easy,
    "medium": sc2_env.Difficulty.medium,
    "medium_hard": sc2_env.Difficulty.medium_hard,
    "hard": sc2_env.Difficulty.hard,
    "harder": sc2_env.Difficulty.harder,
    "very_hard": sc2_env.Difficulty.very_hard,
    "cheat_vision": sc2_env.Difficulty.cheat_vision,
    "cheat_money": sc2_env.Difficulty.cheat_money,
    "cheat_insane": sc2_env.Difficulty.cheat_insane,
}


def _build_sc2_env(config: EnvConfig) -> sc2_env.SC2Env:
    return sc2_env.SC2Env(
        map_name=config.map_name,
        players=[
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(_RACE_MAP[config.opponent_race], _DIFFICULTY_MAP[config.difficulty]),
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=config.map_size, minimap=config.map_size),
            action_space=sc2_actions.ActionSpace.RAW,
            use_raw_units=True,
            use_feature_units=False,
            raw_resolution=config.map_size,
        ),
        step_mul=config.step_mul,
        game_steps_per_episode=0,
        visualize=config.visualize,
    )


class SC2FightEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig, env_factory=_build_sc2_env):
        super().__init__()
        self.config = config
        self._env_factory = env_factory

        self.grid = SectorGrid(map_size=config.map_size, cols=config.grid.cols, rows=config.grid.rows)
        self.action_spec = ActionSpaceSpec(grid=self.grid)
        self.masking_config = MaskingConfig(
            max_supply_depots=config.masking.max_supply_depots,
            max_barracks=config.masking.max_barracks,
            supply_depot_minerals=config.masking.supply_depot_minerals,
            barracks_minerals=config.masking.barracks_minerals,
            marine_minerals=config.masking.marine_minerals,
            min_marines_to_move=config.masking.min_marines_to_move,
        )
        self._translator = ActionTranslator(self.action_spec)
        self._sc2_env = None
        self._state: GameState | None = None
        self._cooldowns = np.zeros(self.action_spec.num_actions, dtype=np.int32)
        self._prev_total_value_units = 0
        self._prev_kill_value = 0
        self._seen_enemy_sectors: set[int] = set()
        self._orientation = SpawnOrientation(map_size=config.map_size, mirror_x=False, mirror_y=False)

        self.action_space = spaces.Discrete(self.action_spec.num_actions)
        obs_len = observation_length(self.grid)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    def _ensure_env(self):
        if self._sc2_env is None:
            self._sc2_env = self._env_factory(self.config)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_env()
        timesteps = self._sc2_env.reset()
        self._cooldowns[:] = 0
        self._state = GameState.from_observation(timesteps[0])
        self._orientation = self._compute_orientation(self._state)
        self._prev_total_value_units = self._state.total_value_units
        self._prev_kill_value = self._state.killed_value_units + self._state.killed_value_structures
        self._seen_enemy_sectors = set()
        obs = featurize(self._state, self.grid, self.config.max_game_loop_norm, self._orientation)
        return obs, {}

    def _sector_of(self, x: float, y: float) -> int:
        cx, cy = self._orientation.to_canonical(x, y)
        return self.grid.sector_of(cx, cy)

    def _compute_orientation(self, state: GameState) -> SpawnOrientation:
        home = state.command_center_pos
        if home is None:
            # Shouldn't happen at reset (you always start with a command
            # center), but fall back to an identity mapping rather than crash.
            return SpawnOrientation(map_size=self.config.map_size, mirror_x=False, mirror_y=False)
        return SpawnOrientation.from_home_position(self.config.map_size, *home)

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("step() called before reset()")

        legal = self.action_masks()
        if not legal[action]:
            action = int(FixedAction.NO_OP)

        calls = self._translator.translate(action, self._state, self._orientation)
        timesteps = self._sc2_env.step([calls])
        ts = timesteps[0]

        self._state = GameState.from_observation(ts)
        self._tick_cooldowns(action)

        obs = featurize(self._state, self.grid, self.config.max_game_loop_norm, self._orientation)
        reward = self._compute_reward(ts)
        terminated = bool(ts.last())
        truncated = False
        return obs, reward, terminated, truncated, {}

    def action_masks(self) -> np.ndarray:
        if self._state is None:
            return np.zeros(self.action_spec.num_actions, dtype=bool)
        hard_mask = compute_action_masks(self._state, self.action_spec, self.masking_config)
        cooldown_mask = self._cooldowns <= 0
        mask = hard_mask & cooldown_mask
        mask[FixedAction.NO_OP] = True
        return mask

    def _tick_cooldowns(self, taken_action: int) -> None:
        self._cooldowns[self._cooldowns > 0] -= 1
        if taken_action in (FixedAction.BUILD_SUPPLY_DEPOT, FixedAction.BUILD_BARRACKS):
            self._cooldowns[taken_action] = self.config.build_cooldown_steps

    def _compute_reward(self, ts) -> float:
        reward = float(ts.reward)
        if self.config.reward.shaping_enabled and not ts.last():
            reward += self._combat_shaping_reward()
            reward += self._home_defense_penalty()
            reward += self._scouting_bonus()
        return reward

    def _combat_shaping_reward(self) -> float:
        """Army-value growth is always rewarded/penalized in full; the
        killed-value portion is scaled by min(1, marine_count /
        concentration_threshold) -- Mass / Concentration of Force -- so a
        kill landed with a large army earns full credit while one landed
        with a tiny, exposed squad earns much less."""
        cfg = self.config.reward
        total_value = self._state.total_value_units
        kill_value = self._state.killed_value_units + self._state.killed_value_structures

        army_delta = total_value - self._prev_total_value_units
        kill_delta = kill_value - self._prev_kill_value
        concentration_factor = min(1.0, len(self._state.marines) / max(cfg.concentration_threshold, 1))

        self._prev_total_value_units = total_value
        self._prev_kill_value = kill_value

        return cfg.shaping_coefficient * (army_delta + concentration_factor * kill_delta)

    def _home_defense_penalty(self) -> float:
        """Economy of Force / Security: per-step penalty while the home
        sector has enemy units present and no friendly marines there to
        respond."""
        enemy_at_home = any(self._sector_of(u.x, u.y) == _HOME_SECTOR for u in self._state.enemies)
        if not enemy_at_home:
            return 0.0
        friendly_at_home = any(self._sector_of(u.x, u.y) == _HOME_SECTOR for u in self._state.marines)
        return 0.0 if friendly_at_home else -self.config.reward.home_defense_penalty

    def _scouting_bonus(self) -> float:
        """OODA loop (Observe): one-time reward the first time an enemy unit
        is seen in a given sector during this episode."""
        seen_this_step = {self._sector_of(u.x, u.y) for u in self._state.enemies}
        newly_seen = seen_this_step - self._seen_enemy_sectors
        if not newly_seen:
            return 0.0
        self._seen_enemy_sectors |= newly_seen
        return self.config.reward.scouting_bonus * len(newly_seen)

    def close(self):
        if self._sc2_env is not None:
            self._sc2_env.close()
            self._sc2_env = None
