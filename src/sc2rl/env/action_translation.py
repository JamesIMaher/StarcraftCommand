"""Translates a semantic action (index or name) into raw PySC2 FunctionCalls.

This is the seam a future generative-AI command layer is meant to call into as
well: `translate_named("build_barracks", state)` works exactly like the
trained policy calling `translate(FixedAction.BUILD_BARRACKS, state)`, so a
higher-level (human- or LLM-driven) command layer never needs to know raw
PySC2 function IDs.
"""

from __future__ import annotations

import random

from pysc2.lib import actions as sc2_actions

from .action_space import ActionSpaceSpec, FixedAction
from .game_state import GameState, UnitInfo
from .sector_grid import SpawnOrientation

_BUILD_OFFSET_RANGE = 6.0
# Small on purpose: this used to be 3.0, giving each marine an independent
# random offset from the shared target every time a move order fired --
# confirmed live as part of why the group would scatter instead of staying
# clustered (easier for the enemy to defeat piecemeal). A little jitter
# avoids every marine pathing to the exact same point (SC2's own collision
# avoidance handles spacing from there); it doesn't need to be large enough
# to meaningfully separate the group on its own.
_MOVE_VARIANCE = 0.75


# Keep clamped targets this far inside the playable-area boundary rather than
# exactly on it, so the point is somewhere a unit can actually stand.
_PLAYABLE_MARGIN = 1.0


class ActionTranslator:
    def __init__(self, spec: ActionSpaceSpec, rng: random.Random | None = None):
        self.spec = spec
        self._rng = rng or random.Random()
        self._barracks_cursor = 0
        # (min_x, min_y, max_x, max_y) in world coordinates, from the game's
        # own start_raw.playable_area. Set by the env once a game is running;
        # None falls back to the full map_size square. The map's playable
        # area is inset from its nominal size (Simple64's is well inside the
        # 64x64 square), so a target at the literal map corner is unpathable
        # -- an attack-move there never completes, and the whole army parks
        # at the nearest cliff edge forever. Confirmed live, twice: once in
        # the old repo and again here after edge-biased attack targets were
        # introduced to reach corner buildings on the coarser 4x4 grid.
        self.playable_area: tuple[float, float, float, float] | None = None

    def translate(self, action_index: int, state: GameState, orientation: SpawnOrientation) -> list:
        if action_index == FixedAction.NO_OP:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        if action_index == FixedAction.BUILD_SUPPLY_DEPOT:
            return self._build_supply_depot(state)
        if action_index == FixedAction.BUILD_BARRACKS:
            return self._build_barracks(state)
        if action_index == FixedAction.TRAIN_MARINE:
            return self._train_marine(state)
        if self.spec.is_move_action(action_index):
            sector = self.spec.sector_for_move_action(action_index)
            return self._move_army(state, sector, orientation)
        raise ValueError(f"unknown action index: {action_index}")

    def translate_named(self, action_name: str, state: GameState, orientation: SpawnOrientation) -> list:
        return self.translate(self.spec.index_for_name(action_name), state, orientation)

    def _pick_scv(self, state: GameState) -> UnitInfo | None:
        # Deliberately not filtered to idle_scvs: a build order interrupts
        # whatever a worker is doing (mining included), so requiring an idle
        # one -- order_length == 0 -- means almost never finding a candidate,
        # since an auto-mining SCV has a continuously active harvest order.
        if not state.scvs:
            return None
        return self._rng.choice(state.scvs)

    def _build_supply_depot(self, state: GameState) -> list:
        scv = self._pick_scv(state)
        if scv is None:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        target = self._offset_point(scv.x, scv.y)
        return [sc2_actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, target)]

    def _build_barracks(self, state: GameState) -> list:
        scv = self._pick_scv(state)
        if scv is None:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        target = self._offset_point(scv.x, scv.y)
        return [sc2_actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, target)]

    def _train_marine(self, state: GameState) -> list:
        complete = state.complete_barracks
        if not complete:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        self._barracks_cursor %= len(complete)
        barracks = complete[self._barracks_cursor]
        self._barracks_cursor = (self._barracks_cursor + 1) % len(complete)
        return [sc2_actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)]

    def _move_army(self, state: GameState, sector: int, orientation: SpawnOrientation) -> list:
        if not state.marines:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        target = self._known_structure_target(state, sector, orientation)
        if target is None:
            # sector_center() is in canonical (home-relative) space; convert
            # back to real map coordinates for the actual attack-move order.
            # At the default 6x6 grid a cell's center already sees the whole
            # cell (half-diagonal ~7.5 < marine sight ~9), so an empty
            # sector's center is the right place to sweep to.
            target = orientation.to_world(*self.spec.grid.sector_center(sector))
        wx, wy = target
        calls = []
        for marine in state.marines:
            tx, ty = self._clamp_to_playable(
                wx + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE),
                wy + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE),
            )
            calls.append(sc2_actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, (tx, ty)))
        return calls

    def _known_structure_target(
        self, state: GameState, sector: int, orientation: SpawnOrientation
    ) -> tuple[float, float] | None:
        """If the target sector holds a known enemy structure (visible, or a
        fog snapshot of one seen earlier), attack-move at the structure
        itself -- the one nearest the army -- instead of the sector's center.
        A building's own position is pathable by construction, which is what
        makes this the fix for the unreachable-corner problem: the policy now
        sees known structures per sector in its observation, so "go to the
        sector with the building" resolves to "go to the building"."""
        in_sector = [
            u for u in state.enemy_structures
            if self.spec.grid.sector_of(*orientation.to_canonical(u.x, u.y)) == sector
        ]
        if not in_sector:
            return None
        n = len(state.marines)
        army_x = sum(m.x for m in state.marines) / n
        army_y = sum(m.y for m in state.marines) / n
        nearest = min(in_sector, key=lambda u: (u.x - army_x) ** 2 + (u.y - army_y) ** 2)
        return nearest.x, nearest.y

    def _clamp_to_playable(self, x: float, y: float) -> tuple[float, float]:
        if self.playable_area is None:
            min_x = min_y = 0.0
            max_x = max_y = float(self.spec.grid.map_size)
        else:
            min_x, min_y, max_x, max_y = self.playable_area
        return (
            max(min_x + _PLAYABLE_MARGIN, min(max_x - _PLAYABLE_MARGIN, x)),
            max(min_y + _PLAYABLE_MARGIN, min(max_y - _PLAYABLE_MARGIN, y)),
        )

    def _offset_point(self, x: float, y: float) -> tuple[float, float]:
        dx = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        dy = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        return x + dx, y + dy
