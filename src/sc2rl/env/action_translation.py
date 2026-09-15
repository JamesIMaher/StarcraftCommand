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


class ActionTranslator:
    def __init__(self, spec: ActionSpaceSpec, rng: random.Random | None = None):
        self.spec = spec
        self._rng = rng or random.Random()
        self._barracks_cursor = 0

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
        # sector_attack_target() is in canonical (home-relative) space;
        # convert back to real map coordinates for the actual attack-move
        # order. Edge/corner sectors are biased to the true map boundary
        # (not just the cell center) so an attack order can actually reach a
        # building tucked into a corner -- see sector_attack_target()'s
        # docstring.
        cx, cy = self.spec.grid.sector_attack_target(sector)
        wx, wy = orientation.to_world(cx, cy)
        map_size = self.spec.grid.map_size
        calls = []
        for marine in state.marines:
            tx = self._clamp(wx + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE), map_size)
            ty = self._clamp(wy + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE), map_size)
            calls.append(sc2_actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, (tx, ty)))
        return calls

    @staticmethod
    def _clamp(value: float, map_size: int) -> float:
        return max(0.0, min(float(map_size), value))

    def _offset_point(self, x: float, y: float) -> tuple[float, float]:
        dx = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        dy = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        return x + dx, y + dy
