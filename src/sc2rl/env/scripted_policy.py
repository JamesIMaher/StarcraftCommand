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
from typing import Collection

from .action_masking import MaskingConfig, compute_action_masks
from .action_space import ActionSpaceSpec, FixedAction
from .game_state import GameState
from .sector_grid import SpawnOrientation, home_sector, sectors_of


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
    # A new move/attack order is only (re)issued once at least this fraction
    # of marines are idle (UnitInfo.is_idle -- no active order, so neither
    # mid-fight nor still traveling). Below this fraction, the teacher
    # issues no_op instead, which does not interrupt units' current orders
    # -- confirmed live: reissuing move commands while marines were still
    # engaged or mid-approach was part of why they'd scatter instead of
    # staying grouped, and interrupting combat orders needlessly.
    idle_fraction_to_reorder: float = 0.5
    # Safety net only: give up waiting for the idle signal after this many
    # steps and move on regardless, in case marines get stuck fighting
    # unreachable/kited stragglers indefinitely without ever going idle.
    search_timeout_steps: int = 60


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
        unreachable_sectors: Collection[int] = (),
        mobilized: bool = False,
        garrison_size: int = 0,
    ) -> int:
        home = home_sector(state.command_center_pos, spec.grid, orientation)
        mask = compute_action_masks(
            state, spec, masking_config, home_sector=home, unreachable_sectors=unreachable_sectors,
            mobilized=mobilized,
        )

        if mask[FixedAction.BUILD_SUPPLY_DEPOT] and len(state.supply_depots) < self.config.target_supply_depots:
            return int(FixedAction.BUILD_SUPPLY_DEPOT)
        if mask[FixedAction.BUILD_BARRACKS] and len(state.barracks) < self.config.target_barracks:
            return int(FixedAction.BUILD_BARRACKS)
        if mask[FixedAction.TRAIN_MARINE]:
            return int(FixedAction.TRAIN_MARINE)

        can_move = any(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))
        if not can_move:
            return int(FixedAction.NO_OP)

        home_action = spec.move_action_for_sector(home)
        enemy_sectors = sectors_of(state.enemies, spec.grid, orientation)
        # "The base" is every sector with one of our buildings in it, not
        # just the command center's -- with ~7-unit cells the base straddles
        # several, and a counterattack on the barracks next door has to
        # register as an attack on home.
        base_sectors = sectors_of(state.structures, spec.grid, orientation) or {home}
        enemies_at_base = sum(
            1 for u in state.enemies
            if spec.grid.sector_of(*orientation.to_canonical(u.x, u.y)) in base_sectors
        )
        # The garrison (env.garrison_size marines permanently held at the
        # base -- see SC2FightEnv._update_garrison) can fend off a routine
        # small raid on its own. Recalling the WHOLE offensive force for any
        # enemy sighted near home, even a single stray unit the garrison
        # could handle alone, made the army destroy a few enemies elsewhere
        # and then abandon the search every single time -- confirmed live.
        # garrison_size defaults to 0, so a caller that doesn't know about
        # the garrison gets the original "any enemy at home recalls
        # everyone" behavior unchanged.
        if mask[home_action] and enemies_at_base > garrison_size:
            self._clear_search_target()  # break off any in-progress search to defend
            return home_action

        # attack_threshold launches an offensive; once launched (the env's
        # `mobilized`), the mask keeps advancing legal down to
        # min_marines_to_continue, and the teacher keeps pressing as long as
        # the mask allows it. Regrouping at home only when the mask no
        # longer lets the army go anywhere else -- otherwise an army that
        # launched at 20 and lost a few marines turned around and walked
        # away from the enemy's last buildings (observed live).
        offensive_allowed = any(
            mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors) if s != home
        )
        if len(state.marines) < self.config.attack_threshold and not (mobilized and offensive_allowed):
            return self._regroup_at_home(state, spec, mask, home, base_sectors, orientation)

        return self._search_and_destroy(state, spec, mask, enemy_sectors)

    def _regroup_at_home(self, state, spec, mask, home: int, base_sectors: set[int], orientation) -> int:
        """Below attack_threshold: not mobilized enough to commit to an
        offensive. Holding position is right AT home -- but if the army is
        already out in the field (it advanced at full strength and took
        losses), holding there means sitting in the middle of the map
        getting picked off while reinforcements pile up at home, until the
        count somehow climbs back over the threshold. Observed live exactly
        so. Regroup at the base instead."""
        home_action = spec.move_action_for_sector(home)
        marine_sectors = sectors_of(state.marines, spec.grid, orientation)
        if marine_sectors <= base_sectors or not mask[home_action]:
            return int(FixedAction.NO_OP)
        if self._current_search_target == home and not self._ready_for_new_order(state):
            self._search_target_steps += 1
            return int(FixedAction.NO_OP)  # already heading home -- don't re-issue every step
        self._current_search_target = home
        self._search_target_steps = 0
        return home_action

    def _search_and_destroy(self, state: GameState, spec, mask, enemy_sectors: set[int]) -> int:
        ready_for_new_order = self._ready_for_new_order(state)

        if enemy_sectors:
            # Destroy takes priority over searching: attack the closest
            # legal sector with a currently-known enemy in it.
            target_sector = max(s for s in enemy_sectors if mask[spec.move_action_for_sector(s)]) \
                if any(mask[spec.move_action_for_sector(s)] for s in enemy_sectors) else None
            if target_sector is not None:
                if target_sector == self._current_search_target and not ready_for_new_order:
                    return int(FixedAction.NO_OP)  # already engaging it -- don't interrupt
                self._current_search_target = target_sector
                self._search_target_steps = 0
                return spec.move_action_for_sector(target_sector)

        # No enemy currently visible.
        if not ready_for_new_order:
            # Still mid-approach or finishing something -- leave them be
            # rather than interrupt with a redundant order.
            self._search_target_steps += 1
            return int(FixedAction.NO_OP)

        # Idle (or timed out) with nothing to fight here -- this area is
        # clear, move on to the next one.
        if self._current_search_target is not None:
            self._cleared_sectors.add(self._current_search_target)

        # Only sectors the mask allows: an unreachable sector (no pathable
        # ground) is never legal, and picking one anyway meant the collector
        # substituted no_op for the illegal order every step -- the army
        # just sat there for the rest of the game.
        legal = [s for s in range(spec.grid.num_sectors) if mask[spec.move_action_for_sector(s)]]
        candidates = [s for s in legal if s not in self._cleared_sectors]
        if not candidates:
            self._cleared_sectors.clear()  # searched everywhere and found nothing -- start over
            candidates = legal
        if not candidates:
            return int(FixedAction.NO_OP)
        # Farthest-from-home first: home-adjacent sectors are already
        # covered by the defense check above, and the enemy is more likely
        # to be found away from our own base.
        self._current_search_target = max(candidates)
        self._search_target_steps = 0
        return spec.move_action_for_sector(self._current_search_target)

    def _ready_for_new_order(self, state: GameState) -> bool:
        if self._search_target_steps >= self.config.search_timeout_steps:
            return True
        if not state.marines:
            return True
        idle_count = sum(1 for m in state.marines if m.is_idle)
        return (idle_count / len(state.marines)) >= self.config.idle_fraction_to_reorder

    def _clear_search_target(self) -> None:
        self._current_search_target = None
        self._search_target_steps = 0
