"""Semantic definition of the flat Discrete(N) action space: N = 4 fixed actions
(economy/no-op) + one move-to-sector action per grid cell. This is the naming
layer a future generative-AI command layer can also address by name (e.g.
"build_barracks") rather than only by numeric index.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .sector_grid import SectorGrid

NUM_FIXED_ACTIONS = 4


class FixedAction(IntEnum):
    NO_OP = 0
    BUILD_SUPPLY_DEPOT = 1
    BUILD_BARRACKS = 2
    TRAIN_MARINE = 3


_FIXED_ACTION_NAMES = {
    FixedAction.NO_OP: "no_op",
    FixedAction.BUILD_SUPPLY_DEPOT: "build_supply_depot",
    FixedAction.BUILD_BARRACKS: "build_barracks",
    FixedAction.TRAIN_MARINE: "train_marine",
}
_NAME_TO_FIXED_ACTION = {name: action for action, name in _FIXED_ACTION_NAMES.items()}


@dataclass(frozen=True)
class ActionSpaceSpec:
    grid: SectorGrid

    @property
    def num_actions(self) -> int:
        return NUM_FIXED_ACTIONS + self.grid.num_sectors

    def is_move_action(self, index: int) -> bool:
        return index >= NUM_FIXED_ACTIONS

    def sector_for_move_action(self, index: int) -> int:
        if not self.is_move_action(index):
            raise ValueError(f"action {index} is not a move action")
        return index - NUM_FIXED_ACTIONS

    def move_action_for_sector(self, sector: int) -> int:
        return NUM_FIXED_ACTIONS + sector

    def name(self, index: int) -> str:
        if self.is_move_action(index):
            return f"move_army_to_sector_{self.sector_for_move_action(index)}"
        return _FIXED_ACTION_NAMES[FixedAction(index)]

    def index_for_name(self, name: str) -> int:
        if name in _NAME_TO_FIXED_ACTION:
            return int(_NAME_TO_FIXED_ACTION[name])
        prefix = "move_army_to_sector_"
        if name.startswith(prefix):
            sector = int(name[len(prefix):])
            return self.move_action_for_sector(sector)
        raise ValueError(f"unknown action name: {name!r}")

    def all_names(self) -> list[str]:
        return [self.name(i) for i in range(self.num_actions)]
