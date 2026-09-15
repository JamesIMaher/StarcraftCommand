"""Gymnasium Env wrapping pysc2.env.sc2_env.SC2Env.

This is the ONLY module in sc2rl that talks to a live StarCraft II client.
Everything it delegates to (observation featurization, action masking, action
translation) operates on plain GameState snapshots and is fully unit-testable
without a live game -- see tests/test_env_wrapper.py, which exercises this
class's step()/reset() contract against a stubbed SC2Env instead of a real one.
"""

from __future__ import annotations

import math
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
from .action_masking import MaskingConfig, can_advance, compute_action_masks

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
from .pathing import PathingMap
from .sector_grid import SectorGrid, SpawnOrientation, home_sector, sectors_of

_RACE_MAP = {
    "terran": sc2_env.Race.terran,
    "protoss": sc2_env.Race.protoss,
    "zerg": sc2_env.Race.zerg,
    "random": sc2_env.Race.random,
}

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
            min_marines_to_continue=config.masking.min_marines_to_continue,
        )
        self._mobilized = False
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
        self._time_total = 0.0
        self._prev_approach_potential: float | None = None
        self._prev_known_structure_tags: frozenset[int] = frozenset()
        self._approach_pos_total = 0.0
        self._approach_neg_total = 0.0
        self._pathing: PathingMap | None = None
        self._raw_pathing: PathingMap | None = None
        self._unreachable_sectors: set[int] = set()
        self._reported_unreachable = False
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

    @property
    def mobilized(self) -> bool:
        """Episode memory: the army has reached masking.min_marines_to_advance
        at some point and hasn't since fallen below min_marines_to_continue
        -- the hysteresis input to the movement mask. Public so the
        demonstration collector can hand it to the scripted teacher."""
        return self._mobilized

    def _update_mobilized(self) -> None:
        count = len(self._state.marines)
        if count >= self.masking_config.min_marines_to_advance:
            self._mobilized = True
        elif count < self.masking_config.min_marines_to_continue:
            self._mobilized = False

    @property
    def unreachable_sectors(self) -> frozenset[int]:
        """Sectors with no pathable ground this episode (see pathing.py) --
        never legal move targets. Public so the demonstration collector can
        hand them to the scripted teacher, which computes its own mask."""
        return frozenset(self._unreachable_sectors)

    def _ensure_env(self):
        if self._sc2_env is None:
            self._sc2_env = self._env_factory(self.config)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._ensure_env()
        timesteps = self._sc2_env.reset()
        self._apply_playable_area(self._read_playable_area())
        self._cooldowns[:] = 0
        self._state = GameState.from_observation(timesteps[0])
        self._mobilized = False
        self._update_mobilized()
        self._orientation = self._compute_orientation(self._state)
        self._pathing = self._raw_pathing = self._read_pathing(timesteps[0])
        if self._pathing is not None and self._state.command_center_pos is not None:
            # Only ground actually connected to the base counts -- see pathing.py.
            self._pathing = self._pathing.reachable_from(*self._state.command_center_pos)
        self._translator.pathing = self._pathing
        self._compute_sector_targets()
        self._prev_total_value = self._state.total_value_units + self._state.total_value_structures
        self._prev_kill_value = self._state.killed_value_units + self._state.killed_value_structures
        self._seen_enemy_sectors = set()
        # The base's sectors count as already "visited" at spawn -- marines
        # start there, so they shouldn't pay an exploration bonus (or read as
        # unexplored in the observation) the first time they're checked.
        # Unreachable sectors likewise: there is nothing there to explore,
        # and leaving them "unexplored" would have the exploration/stale
        # incentives pulling the army toward ground it can never stand on.
        self._visited_sectors = set(self._base_sectors()) | set(self._unreachable_sectors)
        self._newly_visited_this_step: set[int] = set()
        self._steps_since_new_sector = 0
        self._home_defense_total = 0.0
        self._stale_search_total = 0.0
        self._time_total = 0.0
        self._prev_approach_potential = None
        self._prev_known_structure_tags = frozenset()
        self._approach_pos_total = 0.0
        self._approach_neg_total = 0.0
        return self._featurize(), {}

    @staticmethod
    def _read_pathing(ts) -> PathingMap | None:
        """The minimap `pathable` layer, which at minimap size == raw_resolution
        is in exactly the raw frame unit positions use. None when the
        observation has no feature layers (stubbed env): everything pathable."""
        feature_minimap = getattr(ts.observation, "feature_minimap", None)
        if feature_minimap is None:
            return None
        pathable = np.asarray(feature_minimap[features.MINIMAP_FEATURES.pathable.index]) > 0
        try:
            height = np.asarray(feature_minimap[features.MINIMAP_FEATURES.height_map.index])
        except (KeyError, IndexError, TypeError):
            height = None
        return PathingMap(pathable, height)

    def _compute_sector_targets(self) -> None:
        """Per-sector attack target = the pathable point nearest the sector's
        center (world coords), and the set of sectors with no pathable
        ground at all. Recomputed every reset because the orientation (which
        world rectangle each canonical sector covers) changes with the spawn
        corner."""
        self._unreachable_sectors = set()
        if self._pathing is None:
            self._translator.sector_targets = None
            if not self._reported_unreachable:
                print("[env] WARNING: observation has no pathable layer -- move targets fall back to sector centers")
                self._reported_unreachable = True
            return
        grid, orientation = self.grid, self._orientation
        targets: list[tuple[float, float] | None] = []
        for sector in range(grid.num_sectors):
            col, row = grid.sector_coords(sector)
            cx0, cy0 = grid.min_x + col * grid.cell_width, grid.min_y + row * grid.cell_height
            (ax, ay), (bx, by) = orientation.to_world(cx0, cy0), orientation.to_world(
                cx0 + grid.cell_width, cy0 + grid.cell_height,
            )
            rect = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
            # Interior cells only when there are any: a cliff-edge cell is
            # ambiguous about which level it's on and sends the army to
            # stand at the foot of a cliff.
            target = self._pathing.nearest_pathable(
                *orientation.to_world(*grid.sector_center(sector)), rect, prefer_interior=True,
            )
            targets.append(target)
            if target is None:
                self._unreachable_sectors.add(sector)
        self._translator.sector_targets = targets
        if not self._reported_unreachable:
            print(
                f"[env] pathing: {self._pathing.cell_count} raw-frame cells reachable from the command "
                f"center; sectors with no reachable ground (never move targets): "
                f"{sorted(self._unreachable_sectors) or 'none'}"
            )
            if self.config.pathing_debug_path:
                self._write_pathing_debug(self.config.pathing_debug_path, targets)
                print(f"[env] wrote pathing debug map to {self.config.pathing_debug_path}")
            self._reported_unreachable = True

    def _write_pathing_debug(self, path: str, targets: list[tuple[float, float] | None]) -> None:
        """ASCII map in the raw frame, row 0 at the top (y = 0), one character
        per cell: '#' unpathable, '~' pathable but not reachable from the
        base, '.' reachable, 'C' command center, 'm' marine, 'T' a sector's
        attack target, 'E' enemy structure. See EnvConfig.pathing_debug_path."""
        h, w = self._raw_pathing.shape
        canvas = [["#" if not self._raw_pathing.grid[y, x] else ("." if self._pathing.grid[y, x] else "~")
                   for x in range(w)] for y in range(h)]

        def stamp(x: float, y: float, ch: str) -> None:
            ix, iy = int(x), int(y)
            if 0 <= ix < w and 0 <= iy < h:
                canvas[iy][ix] = ch

        for t in targets:
            if t is not None:
                stamp(t[0], t[1], "T")
        for u in self._state.enemy_structures:
            stamp(u.x, u.y, "E")
        for u in self._state.marines:
            stamp(u.x, u.y, "m")
        if self._state.command_center_pos is not None:
            stamp(*self._state.command_center_pos, "C")

        lines = [
            f"raw frame {w}x{h}; grid bounds (min_x, min_y, max_x, max_y) = {self.grid.bounds}",
            f"orientation mirror_x={self._orientation.mirror_x} mirror_y={self._orientation.mirror_y}",
            f"command center raw = {self._state.command_center_pos}; home sector = {self._home_sector()}",
            f"reachable cells = {self._pathing.cell_count} of {self._raw_pathing.cell_count} pathable",
            f"unreachable sectors = {sorted(self._unreachable_sectors)}",
            "sector -> canonical center -> world target:",
        ]
        for s, t in enumerate(targets):
            cx, cy = self.grid.sector_center(s)
            wx, wy = self._orientation.to_world(cx, cy)
            lines.append(f"  {s:2d}: canon ({cx:5.1f},{cy:5.1f}) world ({wx:5.1f},{wy:5.1f}) -> {t}")
        lines.append("")
        lines.append("    " + "".join(str(x % 10) for x in range(w)))
        lines.extend(f"{y:3d} " + "".join(row) for y, row in enumerate(canvas))
        if self._raw_pathing.height is not None:
            lines.append("")
            lines.append("height_map // 16 (one hex digit per cell; '#' unpathable):")
            height = self._raw_pathing.height
            lines.append("    " + "".join(str(x % 10) for x in range(w)))
            for y in range(h):
                row = "".join(
                    f"{min(int(height[y, x]) // 16, 15):x}" if self._raw_pathing.grid[y, x] else "#"
                    for x in range(w)
                )
                lines.append(f"{y:3d} {row}")
            for u in self._state.enemy_structures:
                lines.append(f"enemy structure {u.unit_type} at raw ({u.x},{u.y}) height {self._raw_pathing.height_at(u.x, u.y)}")
            if self._state.command_center_pos is not None:
                cc = self._state.command_center_pos
                lines.append(f"command center height {self._raw_pathing.height_at(*cc)}")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

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

    def _apply_playable_area(self, bounds: tuple[float, float, float, float] | None) -> None:
        """Lay the sector grid over the playable area (see SectorGrid.bounds
        for why). The number of sectors -- and so the action/observation
        space sizes -- never changes, only the geometry, so this is safe to
        do once the game is up. No-op when nothing is known (stubbed env)."""
        if bounds is None or bounds == self.grid.bounds:
            return
        print(f"[env] sector grid laid over playable area (raw frame) x {bounds[0]:.1f}..{bounds[2]:.1f}, "
              f"y {bounds[1]:.1f}..{bounds[3]:.1f}")
        self.grid = self.grid.with_bounds(bounds)
        self.action_spec = ActionSpaceSpec(grid=self.grid)
        self._translator.spec = self.action_spec

    def _read_playable_area(self) -> tuple[float, float, float, float] | None:
        """The game's own start_raw.playable_area, via SC2Env.game_info,
        converted into the raw frame -- the rectangle the sector grid is laid
        over and attack targets are clamped to. None (full-map fallback) when
        the underlying env doesn't expose it, e.g. the stubbed env in tests.

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
        return (
            area.p0.x * scale,
            (world.y - area.p1.y) * scale,  # y flips, so p1.y becomes the minimum
            area.p1.x * scale,
            (world.y - area.p0.y) * scale,
        )

    def _sector_of(self, x: float, y: float) -> int:
        cx, cy = self._orientation.to_canonical(x, y)
        return self.grid.sector_of(cx, cy)

    def _home_sector(self) -> int:
        return home_sector(self._state.command_center_pos, self.grid, self._orientation)

    def _base_sectors(self) -> set[int]:
        """Every sector with one of our buildings in it -- what "home" means
        for defense. With ~7-unit cells the base straddles several sectors,
        so a single hard-coded home sector (as this used to be) missed
        attacks on the buildings next door."""
        return sectors_of(self._state.structures, self.grid, self._orientation) or {self._home_sector()}

    def _compute_orientation(self, state: GameState) -> SpawnOrientation:
        home = state.command_center_pos
        if home is None:
            # Shouldn't happen at reset (you always start with a command
            # center), but fall back to an identity mapping rather than crash.
            return SpawnOrientation(
                map_size=self.config.map_size, mirror_x=False, mirror_y=False, bounds=self.grid.bounds,
            )
        return SpawnOrientation.from_home_position(self.config.map_size, *home, bounds=self.grid.bounds)

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
        self._update_mobilized()
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
        hard_mask = compute_action_masks(
            self._state, self.action_spec, self.masking_config,
            home_sector=self._home_sector(), unreachable_sectors=self._unreachable_sectors,
            mobilized=self._mobilized,
        )
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
        reward_time = 0.0
        reward_approach = 0.0
        if self.config.reward.shaping_enabled:
            reward_economic, reward_kill = self._combat_shaping_reward()
            reward_home_defense = self._home_defense_penalty()
            reward_scouting = self._scouting_bonus()
            reward_exploration = self._exploration_bonus()
            reward_stale_search = self._stale_search_penalty(made_progress=bool(self._newly_visited_this_step))
            reward_time = self._time_penalty()
            reward_approach = self._approach_reward()

        self._last_reward_breakdown = {
            "reward_terminal": reward_terminal,
            "reward_economic": reward_economic,
            "reward_kill": reward_kill,
            "reward_home_defense": reward_home_defense,
            "reward_scouting": reward_scouting,
            "reward_exploration": reward_exploration,
            "reward_stale_search": reward_stale_search,
            "reward_time": reward_time,
            "reward_approach": reward_approach,
        }
        return sum(self._last_reward_breakdown.values())

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
        base_sectors = self._base_sectors()
        threatened = {self._sector_of(u.x, u.y) for u in self._state.enemies} & base_sectors
        if not threatened:
            return 0.0
        defended = {self._sector_of(u.x, u.y) for u in self._state.marines}
        if threatened <= defended:
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
        if not can_advance(len(self._state.marines), self.masking_config, self._mobilized):
            # Not allowed to leave home, so holding position is the only
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

    def _time_penalty(self) -> float:
        """See RewardConfig.time_penalty_per_step: time costs something, so
        finishing sooner is worth more. Capped per episode."""
        cfg = self.config.reward
        remaining = cfg.time_penalty_cap - self._time_total
        if remaining <= 0:
            return 0.0
        penalty = min(cfg.time_penalty_per_step, remaining)
        self._time_total += penalty
        return -penalty

    def _approach_reward(self) -> float:
        """See RewardConfig.approach_reward_scale: potential-based reward for
        closing distance between the army's centroid and the nearest known
        enemy structure. No reward on a step where the set of known
        structures changed (discovery or kill), so destroying a building is
        never charged as "the nearest one just got farther away"."""
        cfg = self.config.reward
        structures, marines = self._state.enemy_structures, self._state.marines
        tags = frozenset(u.tag for u in structures)
        potential = None
        if structures and marines:
            army_x = sum(m.x for m in marines) / len(marines)
            army_y = sum(m.y for m in marines) / len(marines)
            distance = min(math.hypot(u.x - army_x, u.y - army_y) for u in structures)
            diagonal = math.hypot(self.grid.max_x - self.grid.min_x, self.grid.max_y - self.grid.min_y)
            potential = -cfg.approach_reward_scale * distance / max(diagonal, 1e-6)

        reward = 0.0
        comparable = potential is not None and self._prev_approach_potential is not None
        if comparable and tags == self._prev_known_structure_tags:
            delta = potential - self._prev_approach_potential
            if delta > 0:
                delta = min(delta, max(cfg.approach_reward_cap - self._approach_pos_total, 0.0))
                self._approach_pos_total += delta
            elif delta < 0:
                delta = max(delta, -max(cfg.approach_reward_cap - self._approach_neg_total, 0.0))
                self._approach_neg_total -= delta
            reward = delta
        self._prev_approach_potential = potential
        self._prev_known_structure_tags = tags
        return reward

    def close(self):
        if self._sc2_env is not None:
            self._sc2_env.close()
            self._sc2_env = None
