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
from .sector_grid import SpawnOrientation, home_sector

_BUILD_OFFSET_RANGE = 6.0
# Small on purpose: this used to be 3.0, giving each marine an independent
# random offset from the shared target every time a move order fired --
# confirmed live as part of why the group would scatter instead of staying
# clustered (easier for the enemy to defeat piecemeal). A little jitter
# avoids every marine pathing to the exact same point (SC2's own collision
# avoidance handles spacing from there); it doesn't need to be large enough
# to meaningfully separate the group on its own.
_MOVE_VARIANCE = 0.75


# Keep clamped targets this far inside the grid's bounds (the playable area,
# once the env knows it) rather than exactly on the boundary, so the point is
# somewhere a unit can actually stand. A target outside the playable area is
# unpathable -- an attack-move there never completes, and the whole army
# parks at the nearest cliff edge forever. Confirmed live, twice: once in the
# old repo and again here after edge-biased attack targets were introduced to
# reach corner buildings on the coarser 4x4 grid.
_PLAYABLE_MARGIN = 1.0


class ActionTranslator:
    def __init__(self, spec: ActionSpaceSpec, rng: random.Random | None = None):
        self.spec = spec
        self._rng = rng or random.Random()
        self._barracks_cursor = 0
        # Per-sector pathable target in world coordinates (None for a sector
        # with no pathable ground), computed by the env from the game's
        # pathing layer once a game is running -- see pathing.py. None here
        # means "unknown": fall back to sector centers.
        self.sector_targets: list[tuple[float, float] | None] | None = None
        # Ground reachable from the base this episode (see pathing.py), used
        # to snap a known enemy structure's position -- which is inside its
        # own unpathable footprint, and may be on high ground the army can
        # only reach via a ramp -- to the nearest reachable point.
        self.pathing = None

    # A structure is at most a few cells across; a reachable point this far
    # from its center is adjacent to it, i.e. in a marine's weapon range.
    _STRUCTURE_SNAP_RADIUS = 6.0

    def translate(
        self,
        action_index: int,
        state: GameState,
        orientation: SpawnOrientation,
        garrison_tags: frozenset[int] = frozenset(),
    ) -> list:
        """`garrison_tags` (see SC2FightEnv._update_garrison) are marines a
        move action must never touch -- they hold the base regardless of
        which sector is targeted, including a "recall home" order."""
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
            return self._move_army(state, sector, orientation, garrison_tags)
        raise ValueError(f"unknown action index: {action_index}")

    def translate_named(
        self,
        action_name: str,
        state: GameState,
        orientation: SpawnOrientation,
        garrison_tags: frozenset[int] = frozenset(),
    ) -> list:
        return self.translate(self.spec.index_for_name(action_name), state, orientation, garrison_tags)

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

    def _move_army(
        self, state: GameState, sector: int, orientation: SpawnOrientation, garrison_tags: frozenset[int]
    ) -> list:
        movers = [m for m in state.marines if m.tag not in garrison_tags]
        if not movers:
            return [sc2_actions.RAW_FUNCTIONS.no_op()]
        target = self._known_structure_target(state, sector, orientation, movers)
        cc = state.command_center_pos
        if target is None and cc is not None and sector == home_sector(cc, self.spec.grid, orientation):
            # "Recall/defend home" means the base itself, not the home
            # sector's geometric center -- the army should gather at the
            # command center, not at the cell's midpoint (which, with the old
            # full-map grid, was the corner cliff behind the base).
            target = cc
        if target is None and self.sector_targets is not None:
            target = self.sector_targets[sector]  # pathable point nearest the center, already world coords
        if target is None:
            # sector_center() is in canonical (home-relative) space; convert
            # back to real map coordinates for the actual attack-move order.
            # At the default 6x6 grid over Simple64's playable area a cell's
            # center already sees the whole cell (half-diagonal ~5 < marine
            # sight ~9), so an empty sector's center is the right place to
            # sweep to.
            target = orientation.to_world(*self.spec.grid.sector_center(sector))
        wx, wy = target
        calls = []
        for marine in movers:
            tx, ty = self._clamp_to_playable(
                wx + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE),
                wy + self._rng.uniform(-_MOVE_VARIANCE, _MOVE_VARIANCE),
            )
            calls.append(sc2_actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, (tx, ty)))
        return calls

    def _known_structure_target(
        self, state: GameState, sector: int, orientation: SpawnOrientation, movers: list[UnitInfo]
    ) -> tuple[float, float] | None:
        """If the target sector holds a known enemy structure (visible, or a
        fog snapshot of one seen earlier), attack-move at the structure
        itself -- the one nearest the moving army -- instead of the sector's
        center. A building's own position is pathable by construction, which
        is what makes this the fix for the unreachable-corner problem: the
        policy now sees known structures per sector in its observation, so
        "go to the sector with the building" resolves to "go to the
        building". `movers` (not the garrison, which isn't going anywhere)
        is what "nearest the army" means here."""
        in_sector = [
            u for u in state.enemy_structures
            if self.spec.grid.sector_of(*orientation.to_canonical(u.x, u.y)) == sector
        ]
        if not in_sector:
            return None
        n = len(movers)
        army_x = sum(m.x for m in movers) / n
        army_y = sum(m.y for m in movers) / n
        for structure in sorted(in_sector, key=lambda u: (u.x - army_x) ** 2 + (u.y - army_y) ** 2):
            if self.pathing is None:
                return structure.x, structure.y
            # Same terrain level as the building: the Euclidean-nearest
            # reachable cell is often at the foot of the cliff below it.
            snapped = self.pathing.nearest_within(
                structure.x, structure.y, self._STRUCTURE_SNAP_RADIUS,
                same_level_as=self.pathing.height_at(structure.x, structure.y),
            )
            if snapped is not None:
                return snapped
        # Every known structure here is out of reach (e.g. high ground with
        # no ramp from this side): fall back to the sector's own target.
        return None

    def _clamp_to_playable(self, x: float, y: float) -> tuple[float, float]:
        grid = self.spec.grid
        return (
            max(grid.min_x + _PLAYABLE_MARGIN, min(grid.max_x - _PLAYABLE_MARGIN, x)),
            max(grid.min_y + _PLAYABLE_MARGIN, min(grid.max_y - _PLAYABLE_MARGIN, y)),
        )

    def _offset_point(self, x: float, y: float) -> tuple[float, float]:
        dx = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        dy = self._rng.uniform(-_BUILD_OFFSET_RANGE, _BUILD_OFFSET_RANGE)
        return x + dx, y + dy
