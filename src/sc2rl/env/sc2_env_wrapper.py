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
            min_marines_to_advance=config.masking.min_marines_to_advance,
        )
        self._translator = ActionTranslator(self.action_spec)
        self._sc2_env = None
        self._state: GameState | None = None
        self._cooldowns = np.zeros(self.action_spec.num_actions, dtype=np.int32)
        self._prev_total_value = 0
        self._prev_kill_value = 0
        self._seen_enemy_sectors: set[int] = set()
        self._visited_sectors: set[int] = set()
        self._newly_visited_this_step: set[int] = set()
        self._steps_since_new_sector = 0
        self._home_defense_total = 0.0
        self._stale_search_total = 0.0
        self._last_reward_breakdown: dict[str, float] = {}
        self._orientation = SpawnOrientation(map_size=config.map_size, mirror_x=False, mirror_y=False)

        self.action_space = spaces.Discrete(self.action_spec.num_actions)
        obs_len = observation_length(self.grid)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    @property
    def state(self) -> GameState | None:
        """Current GameState snapshot -- public so external callers (e.g.
        the scripted-policy demonstration collector) can drive their own
        logic off it without reaching into a private attribute."""
        return self._state

    @property
    def orientation(self) -> SpawnOrientation:
        """Current episode's home-relative coordinate mapping -- see
        SpawnOrientation. Public for the same reason as `state` above."""
        return self._orientation

    def _ensure_env(self):
        if self._sc2_env is None:
            self._sc2_env = self._env_factory(self.config)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_env()
        timesteps = self._sc2_env.reset()
        self._translator.playable_area = self._read_playable_area()
        self._cooldowns[:] = 0
        self._state = GameState.from_observation(timesteps[0])
        self._orientation = self._compute_orientation(self._state)
        self._prev_total_value = self._state.total_value_units + self._state.total_value_structures
        self._prev_kill_value = self._state.killed_value_units + self._state.killed_value_structures
        self._seen_enemy_sectors = set()
        # Home sector counts as already "visited" at spawn -- marines start
        # there, so it shouldn't pay an exploration bonus (or read as
        # unexplored in the observation) the first time it's checked.
        self._visited_sectors = {_HOME_SECTOR}
        self._newly_visited_this_step: set[int] = set()
        self._steps_since_new_sector = 0
        self._home_defense_total = 0.0
        self._stale_search_total = 0.0
        return self._featurize(), {}

    def _featurize(self) -> np.ndarray:
        return featurize(
            self._state, self.grid, self.config.max_game_loop_norm, self._orientation,
            explored_sectors=self._visited_sectors,
        )

    def _record_visited_sectors(self) -> None:
        """Episode memory of which sectors a marine has been in. Updated on
        every step regardless of reward shaping, because the observation
        reads it too (the `explored` per-sector feature) -- not just the
        exploration bonus."""
        present_this_step = {self._sector_of(u.x, u.y) for u in self._state.marines}
        self._newly_visited_this_step = present_this_step - self._visited_sectors
        self._visited_sectors |= self._newly_visited_this_step

    def _read_playable_area(self) -> tuple[float, float, float, float] | None:
        """The game's own start_raw.playable_area, via SC2Env.game_info --
        see ActionTranslator.playable_area for why attack targets must stay
        inside it. None (full-map fallback) when the underlying env doesn't
        expose it, e.g. the stubbed env in tests.

        game_info is in world coordinates, but everything this env works in
        -- raw_units positions and the "world" argument of raw actions -- is
        pysc2's raw_resolution frame: world coordinates scaled by
        raw_resolution / max(map width, height) and with the y axis FLIPPED
        (pysc2 features.Features.init_camera's _world_to_minimap_px chain:
        world -> top-left origin -> scale). The playable area must go
        through that same transform or the clamp lands in the wrong place
        entirely -- as it did at first: fed in untransformed, it pushed
        targets toward an edge rather than away from one.
        """
        game_info = getattr(self._sc2_env, "game_info", None)
        if not game_info:
            return None
        start_raw = game_info[0].start_raw
        area, world = start_raw.playable_area, start_raw.map_size
        scale = self.config.map_size / max(world.x, world.y)
        raw = (
            area.p0.x * scale,
            (world.y - area.p1.y) * scale,  # y flips, so p1.y becomes the minimum
            area.p1.x * scale,
            (world.y - area.p0.y) * scale,
        )
        if self._translator.playable_area is None:
            print(
                f"[env] map world size {world.x}x{world.y}, playable area world "
                f"({area.p0.x},{area.p0.y})-({area.p1.x},{area.p1.y}) -> raw frame "
                f"x {raw[0]:.1f}..{raw[2]:.1f}, y {raw[1]:.1f}..{raw[3]:.1f}"
            )
        return raw

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
        self._record_visited_sectors()

        obs = self._featurize()
        reward = self._compute_reward(ts)
        terminated = bool(ts.last())
        truncated = False
        return obs, reward, terminated, truncated, dict(self._last_reward_breakdown)

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
        # Shaping applies on every step, INCLUDING the terminal one. A loss
        # typically means the base/army gets wiped out right at the end --
        # skipping shaping on ts.last() meant that collapse (a large negative
        # economic-value delta) was never charged against the reward the
        # agent had already banked from building an economy earlier in the
        # game, so a losing episode's total reward could still come out
        # strongly positive. Computing it here too means the delta correctly
        # reflects whatever the actual final state is.
        reward_terminal = float(ts.reward) * self.config.reward.terminal_reward_scale
        reward_economic = 0.0
        reward_kill = 0.0
        reward_home_defense = 0.0
        reward_scouting = 0.0
        reward_exploration = 0.0
        reward_stale_search = 0.0
        if self.config.reward.shaping_enabled:
            reward_economic, reward_kill = self._combat_shaping_reward()
            reward_home_defense = self._home_defense_penalty()
            reward_scouting = self._scouting_bonus()
            reward_exploration = self._exploration_bonus()
            reward_stale_search = self._stale_search_penalty(made_progress=bool(self._newly_visited_this_step))

        self._last_reward_breakdown = {
            "reward_terminal": reward_terminal,
            "reward_economic": reward_economic,
            "reward_kill": reward_kill,
            "reward_home_defense": reward_home_defense,
            "reward_scouting": reward_scouting,
            "reward_exploration": reward_exploration,
            "reward_stale_search": reward_stale_search,
        }
        return (
            reward_terminal + reward_economic + reward_kill + reward_home_defense
            + reward_scouting + reward_exploration + reward_stale_search
        )

    def _combat_shaping_reward(self) -> tuple[float, float]:
        """Returns (economic_reward, kill_reward) separately -- see
        step()'s _last_reward_breakdown -- rather than a single combined
        value, so each component's magnitude is directly inspectable instead
        of needing to be reasoned about from formulas after the fact.

        Economic-value growth (units AND structures -- training a marine or
        completing a supply depot/barracks both count) is always
        rewarded/penalized in full. Including total_value_structures matters:
        without it, building a supply depot or barracks earned zero
        immediate shaped reward (only the eventual marines trained from it
        did), which is a weak, indirect signal for "build infrastructure
        early" -- observed in practice as the policy learning to delay
        barracks construction.

        The killed-value portion is scaled by min(1, marine_count /
        concentration_threshold) -- Mass / Concentration of Force -- so a
        kill landed with a large army earns full credit while one landed
        with a tiny, exposed squad earns much less, AND by the separate,
        smaller kill_value_scale, AND capped at kill_value_cap -- since
        killed_value only ever increases (kills aren't offset by an eventual
        loss the way economic value is), all three matter together: observed
        in practice, a losing episode's ep_rew_mean went UP because kills
        traded during a losing fight outweighed the terminal penalty and the
        (comparatively small) economic-collapse penalty.
        """
        cfg = self.config.reward
        total_value = self._state.total_value_units + self._state.total_value_structures
        kill_value = self._state.killed_value_units + self._state.killed_value_structures

        # Structures are already bounded by max_supply_depots/max_barracks,
        # but marine count has no upper limit in the masking -- so
        # total_value_units can climb indefinitely as long as minerals keep
        # flowing, with ZERO requirement to ever risk those marines in
        # combat. Confirmed live: a losing episode earned +2.85 in economic
        # reward alone (total_value grew by ~2850 raw), because marines that
        # never engage also never die. Capping the value used for the delta
        # means growing the army past a generously large-but-bounded ceiling
        # earns no further reward -- removes the incentive to hoard
        # indefinitely while still fully rewarding building a real fighting
        # force. _prev_total_value itself stays uncapped/raw so the delta
        # math stays correct across the boundary.
        capped_current = min(total_value, cfg.economic_value_cap)
        capped_prev = min(self._prev_total_value, cfg.economic_value_cap)
        army_delta = capped_current - capped_prev

        # kill_value only ever increases (kills aren't "undone"), so like
        # total_value it needs a cap or a long grindy fight against a
        # continuously-spawning bot has no natural ceiling.
        capped_kill_current = min(kill_value, cfg.kill_value_cap)
        capped_kill_prev = min(self._prev_kill_value, cfg.kill_value_cap)
        kill_delta = capped_kill_current - capped_kill_prev
        concentration_factor = min(1.0, len(self._state.marines) / max(cfg.concentration_threshold, 1))

        self._prev_total_value = total_value
        self._prev_kill_value = kill_value

        economic_reward = cfg.shaping_coefficient * army_delta
        kill_reward = cfg.shaping_coefficient * cfg.kill_value_scale * concentration_factor * kill_delta
        return economic_reward, kill_reward

    def _home_defense_penalty(self) -> float:
        """Economy of Force / Security: per-step penalty while the home
        sector has enemy units present and no friendly marines there to
        respond. Capped per episode (home_defense_penalty_cap) -- episodes
        have no step limit (only PySC2's own game-end conditions), so an
        uncapped per-step penalty can accumulate for hundreds or thousands
        of steps in an unusually long episode, dwarfing terminal_reward_scale
        despite the caps already in place on the positive-side terms.
        Confirmed live: an early, untrained episode's ep_rew_mean reached
        -46.4 after only 18,000 timesteps -- far below anything the "any win
        beats any loss" analysis accounted for, because that analysis only
        ever reasoned about a single-step transition, never an
        episode-length accumulation of an uncapped per-step penalty."""
        cfg = self.config.reward
        enemy_at_home = any(self._sector_of(u.x, u.y) == _HOME_SECTOR for u in self._state.enemies)
        if not enemy_at_home:
            return 0.0
        friendly_at_home = any(self._sector_of(u.x, u.y) == _HOME_SECTOR for u in self._state.marines)
        if friendly_at_home:
            return 0.0
        if self._home_defense_total >= cfg.home_defense_penalty_cap:
            return 0.0
        penalty = min(cfg.home_defense_penalty, cfg.home_defense_penalty_cap - self._home_defense_total)
        self._home_defense_total += penalty
        return -penalty

    def _scouting_bonus(self) -> float:
        """OODA loop (Observe): one-time reward the first time an enemy unit
        is seen in a given sector during this episode."""
        seen_this_step = {self._sector_of(u.x, u.y) for u in self._state.enemies}
        newly_seen = seen_this_step - self._seen_enemy_sectors
        if not newly_seen:
            return 0.0
        self._seen_enemy_sectors |= newly_seen
        return self.config.reward.scouting_bonus * len(newly_seen)

    def _exploration_bonus(self) -> float:
        """Counterweight to home_defense_penalty -- see the comment on
        RewardConfig.exploration_bonus. One-time reward the first time a
        friendly marine is present in a given sector during this episode,
        independent of whether an enemy is there. The memory itself is
        maintained by _record_visited_sectors() in step()."""
        return self.config.reward.exploration_bonus * len(self._newly_visited_this_step)

    def _stale_search_penalty(self, made_progress: bool) -> float:
        """See RewardConfig.stale_search_penalty -- a flat per-step penalty
        once the army has gone too long without entering a new sector,
        counteracting the fact that exploration_bonus/scouting_bonus are
        both one-time and so eventually stop pulling the army onward.
        Capped per episode (stale_search_penalty_cap) for the same reason
        home_defense_penalty is -- see that method's docstring."""
        cfg = self.config.reward
        if made_progress:
            self._steps_since_new_sector = 0
            return 0.0
        self._steps_since_new_sector += 1
        if len(self._state.marines) < self.config.masking.min_marines_to_advance:
            # Not yet allowed to leave home, so holding position is the only
            # legal (and correct) thing to do. This used to gate on the lower
            # min_marines_to_move, which made marines 4..19 a penalty stream
            # the policy had no legal way to stop -- observed live as it
            # learning to build a few marines and then never any more.
            return 0.0
        if self._steps_since_new_sector <= cfg.stale_search_patience:
            return 0.0
        if self._stale_search_total >= cfg.stale_search_penalty_cap:
            return 0.0
        penalty = min(cfg.stale_search_penalty, cfg.stale_search_penalty_cap - self._stale_search_total)
        self._stale_search_total += penalty
        return -penalty

    def close(self):
        if self._sc2_env is not None:
            self._sc2_env.close()
            self._sc2_env = None
