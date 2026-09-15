"""Stateless grid-sector math shared by the action space (movement targets) and
observation featurization (per-sector unit density). Pure functions/data only —
no PySC2 imports — so it is trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorGrid:
    map_size: int
    cols: int
    rows: int

    @property
    def num_sectors(self) -> int:
        return self.cols * self.rows

    @property
    def cell_width(self) -> float:
        return self.map_size / self.cols

    @property
    def cell_height(self) -> float:
        return self.map_size / self.rows

    def sector_index(self, col: int, row: int) -> int:
        return row * self.cols + col

    def sector_coords(self, sector: int) -> tuple[int, int]:
        return sector % self.cols, sector // self.cols

    def sector_center(self, sector: int) -> tuple[float, float]:
        col, row = self.sector_coords(sector)
        x = (col + 0.5) * self.cell_width
        y = (row + 0.5) * self.cell_height
        return x, y

    def sector_attack_target(self, sector: int) -> tuple[float, float]:
        """Like sector_center(), but the outermost ring of sectors is biased
        all the way to the true map edge instead of stopping at the cell's
        geometric center. An attack-move order needs to actually reach a
        building tucked into a corner/edge -- on a 64-map/4x4 grid the outer
        sector's center sits ~11 units short of the true corner, well beyond
        a marine's weapon (~5) or even sight (~9) range, so attacks would
        arrive and stop short with nothing to engage. Interior sectors are
        unaffected -- their center is already reachable and correct.
        """
        col, row = self.sector_coords(sector)
        if col == 0:
            x = 0.0
        elif col == self.cols - 1:
            x = float(self.map_size)
        else:
            x = (col + 0.5) * self.cell_width
        if row == 0:
            y = 0.0
        elif row == self.rows - 1:
            y = float(self.map_size)
        else:
            y = (row + 0.5) * self.cell_height
        return x, y

    def sector_of(self, x: float, y: float) -> int:
        col = min(max(int(x // self.cell_width), 0), self.cols - 1)
        row = min(max(int(y // self.cell_height), 0), self.rows - 1)
        return self.sector_index(col, row)

    def friendly_quadrant(self, x: float, y: float) -> tuple[float, float, float, float]:
        """Return (min_x, max_x, min_y, max_y) of the map quadrant containing (x, y).

        Used to bound where a command center's starting quadrant is, so build
        placement and other heuristics stay within the player's own base area.
        """
        half = self.map_size / 2
        if x <= half:
            min_x, max_x = 0.0, half
        else:
            min_x, max_x = half, float(self.map_size)
        if y <= half:
            min_y, max_y = 0.0, half
        else:
            min_y, max_y = half, float(self.map_size)
        return min_x, max_x, min_y, max_y


@dataclass(frozen=True)
class SpawnOrientation:
    """Maps real map coordinates to a canonical frame where the player's own
    starting position always falls near (0, 0) -- so sector 0 always means
    "near home" and the farthest sector always means "toward the enemy",
    regardless of which corner the player actually spawned in that episode.

    Maps like Simple64 randomize which corner each side spawns in between
    episodes (confirmed empirically: resets in the same process landed the
    command center in different quadrants across episodes). Without this,
    the same sector index means opposite things in different episodes, so a
    policy can never learn a stable "explore toward the enemy" or "return
    home to defend" action -- this canonicalization is what makes that
    learnable at all.
    """

    map_size: int
    mirror_x: bool
    mirror_y: bool

    @staticmethod
    def from_home_position(map_size: int, home_x: float, home_y: float) -> "SpawnOrientation":
        half = map_size / 2
        return SpawnOrientation(map_size=map_size, mirror_x=home_x > half, mirror_y=home_y > half)

    def to_canonical(self, x: float, y: float) -> tuple[float, float]:
        cx = (self.map_size - x) if self.mirror_x else x
        cy = (self.map_size - y) if self.mirror_y else y
        return cx, cy

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        # Mirroring is its own inverse.
        return self.to_canonical(x, y)
