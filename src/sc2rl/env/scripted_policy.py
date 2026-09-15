"""A simple, hand-written policy used as a behavior-cloning teacher
(imitation-learning warm start) -- NOT intended to play well on its own,
just to give the RL policy a reasonable starting point instead of random
weights. It operates in the exact same action space as the trained policy
(see action_space.py), so its choices are directly usable as supervised
training targets: build a small base, keep training marines, then either
defend home (if under attack and undefended) or -- once mobilized to a real
attack force, not just the minimum to move at all -- search sector by
sector for the enemy and commit to destroying whatever is found. Same
spirit as AlphaStar's imitation-learning warm start from human replays --
imitating our own scripted teacher instead, since our action space is a
simplified custom abstraction that doesn't correspond to raw replay actions
the way AlphaStar's full-interface action space did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .action_masking import MaskingConfig, compute_action_masks
from .action_space import ActionSpaceSpec, FixedAction
from .game_state import GameState, UnitInfo
from .sector_grid import SpawnOrientation

_HOME_SECTOR = 0


@dataclass(frozen=True)
class ScriptedPolicyConfig:
    target_supply_depots: int = 4
    target_barracks: int = 2
    # Separate from masking.min_marines_to_move (which only gates whether
    # movement is legal at all): the army doesn't commit to a full
    # search-and-destroy offensive until it has this many marines. Below
    # this, movement stays legal (e.g. for home defense) but the teacher
    # deliberately holds position rather than advancing piecemeal.
    attack_threshold: int = 20
    # Give up on confirming "arrival" at the current search target after
    # this many steps and move on regardless. Maps often have irregular
    # playable terrain (cliffs, water) that doesn't perfectly fill our
    # coordinate grid, so a sector near the coordinate-space edge can
    # correspond to partially unreachable terrain -- without this, the
    # search can stall forever on a sector the army approaches but never
    # technically enters.
    search_timeout_steps: int = 25


class ScriptedPolicy:
    """Stateful: remembers which sectors it has already searched and found
    empty this episode, and which sector it's currently advancing toward,
    so it commits to clearing one area at a time instead of retargeting
    every step. Call reset() at the start of each episode.
    """

    def __init__(self, config: ScriptedPolicyConfig | None = None):
        self.config = config or ScriptedPolicyConfig()
        self._cleared_sectors: set[int] = set()
        self._current_search_target: int | None = None
        self._search_target_steps = 0

    def reset(self) -> None:
        self._cleared_sectors = set()
        self._current_search_target = None
        self._search_target_steps = 0

    def action(
        self,
        state: GameState,
        spec: ActionSpaceSpec,
        masking_config: MaskingConfig,
        orientation: SpawnOrientation,
    ) -> int:
        mask = compute_action_masks(state, spec, masking_config)

        if mask[FixedAction.BUILD_SUPPLY_DEPOT] and len(state.supply_depots) < self.config.target_supply_depots:
            return int(FixedAction.BUILD_SUPPLY_DEPOT)
        if mask[FixedAction.BUILD_BARRACKS] and len(state.barracks) < self.config.target_barracks:
            return int(FixedAction.BUILD_BARRACKS)
        if mask[FixedAction.TRAIN_MARINE]:
            return int(FixedAction.TRAIN_MARINE)

        can_move = any(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))
        if not can_move:
            return int(FixedAction.NO_OP)

        home_action = spec.move_action_for_sector(_HOME_SECTOR)
        enemy_sectors = _sectors_of(state.enemies, spec, orientation)
        if mask[home_action] and _HOME_SECTOR in enemy_sectors:
            self._clear_search_target()  # break off any in-progress search to defend
            return home_action

        if len(state.marines) < self.config.attack_threshold:
            # Not mobilized enough to commit to an offensive yet -- hold
            # position rather than advance piecemeal.
            return int(FixedAction.NO_OP)

        return self._search_and_destroy(state, spec, orientation, mask, enemy_sectors)

    def _search_and_destroy(self, state, spec, orientation, mask, enemy_sectors: set[int]) -> int:
        # Destroy takes priority over searching: attack the closest legal
        # sector with a currently-known enemy in it.
        for sector in sorted(enemy_sectors, reverse=True):
            action = spec.move_action_for_sector(sector)
            if mask[action]:
                self._clear_search_target()
                return action

        # No enemy currently visible -- keep searching. Mark the current
        # target cleared once either the army actually arrives there with
        # nothing found, OR search_timeout_steps elapses without confirmed
        # arrival (handles targets in terrain the army can approach but
        # never technically enter).
        if self._current_search_target is not None:
            marine_sectors = _sectors_of(state.marines, spec, orientation)
            self._search_target_steps += 1
            arrived = self._current_search_target in marine_sectors
            timed_out = self._search_target_steps >= self.config.search_timeout_steps
            if arrived or timed_out:
                self._cleared_sectors.add(self._current_search_target)
                self._clear_search_target()

        if self._current_search_target is None:
            candidates = [s for s in range(spec.grid.num_sectors) if s not in self._cleared_sectors]
            if not candidates:
                self._cleared_sectors.clear()  # searched everywhere and found nothing -- start over
                candidates = list(range(spec.grid.num_sectors))
            # Farthest-from-home first: home-adjacent sectors are already
            # covered by the defense check above, and the enemy is more
            # likely to be found away from our own base.
            self._current_search_target = max(candidates)

        return spec.move_action_for_sector(self._current_search_target)

    def _clear_search_target(self) -> None:
        self._current_search_target = None
        self._search_target_steps = 0


def _sectors_of(units: Iterable[UnitInfo], spec: ActionSpaceSpec, orientation: SpawnOrientation) -> set[int]:
    sectors = set()
    for unit in units:
        cx, cy = orientation.to_canonical(unit.x, unit.y)
        sectors.add(spec.grid.sector_of(cx, cy))
    return sectors
