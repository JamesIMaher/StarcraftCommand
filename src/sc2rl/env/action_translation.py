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

_BUILD_OFFSET_RANGE = 6.0
_MOVE_VARIANCE = 3.0


class ActionTranslator:
    def __init__(self, spec: ActionSpaceSpec, rng: random.Random | None = None):
        self.spec = spec
        self._rng = rng or random.Random()
        self._barracks_cursor = 0

    def translate(self, action_index: int, state: GameState) -> list:
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
            return self._move_army(state, sector)
        raise ValueError(f"unknown action index: {action_index}")

    def translate_named(self, action_name: str, state: GameState) -> list:
        return self.translate(self.spec.index_for_name(action_name), state)

    def _pick_idle_scv(self, state: GameState) -> UnitInfo | None:
        idle = state.idle_scvs
        if not idle:
            return None
        return self._rng.choice(idle)

    def _build_supply_depot(self, state: GameState) -> list:
        scv = self._pick_idle_scv(state)
        if scv is None:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        target = self._offset_point(scv.x, scv.y)
        return [sc2_actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, target)]

    def _build_barracks(self, state: GameState) -> list:
        scv = self._pick_idle_scv(state)
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

    def _move_army(self, state: GameState, sector: int) -> list:
        if not state.marines:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        cx, cy = self.spec.grid.sector_center(sector)
        calls = []
        for marine in state.marines:
            tx = cx + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE)
            ty = cy + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE)
            calls.append(sc2_actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, (tx, ty)))
        return calls

    def _offset_point(self, x: float, y: float) -> tuple[float, float]:
        dx = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        dy = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        return x + dx, y + dy
