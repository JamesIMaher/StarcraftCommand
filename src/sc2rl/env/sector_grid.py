"""Stateless grid-sector math shared by the action space (movement targets) and
observation featurization (per-sector unit density). Pure functions/data only —
no PySC2 imports — so it is trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


Bounds = tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


@dataclass(frozen=True)
class SectorGrid:
    map_size: int
    cols: int
    rows: int
    # The rectangle the grid is laid over, in the same (raw) coordinate frame
    # as unit positions. Defaults to the full map_size square, but the env
    # replaces it with the game's actual playable area once a game is
    # running (see SC2FightEnv._read_playable_area): on Simple64 the
    # playable area is a ~43-unit square sitting off-center in the 64x64 raw
    # frame, so a grid over the full square wasted the entire last column and
    # first row on sectors nothing could ever reach. Their `explored` flag
    # then stayed 0 forever and the exploration/stale-search incentives kept
    # pulling the army toward those edges -- confirmed live as the army
    # bunching at a map edge.
    bounds: Bounds | None = None

    @property
    def min_x(self) -> float:
        return self.bounds[0] if self.bounds else 0.0

    @property
    def min_y(self) -> float:
        return self.bounds[1] if self.bounds else 0.0

    @property
    def max_x(self) -> float:
        return self.bounds[2] if self.bounds else float(self.map_size)

    @property
    def max_y(self) -> float:
        return self.bounds[3] if self.bounds else float(self.map_size)

    def with_bounds(self, bounds: Bounds) -> "SectorGrid":
        return SectorGrid(map_size=self.map_size, cols=self.cols, rows=self.rows, bounds=bounds)

    @property
    def num_sectors(self) -> int:
        return self.cols * self.rows

    @property
    def cell_width(self) -> float:
        return (self.max_x - self.min_x) / self.cols

    @property
    def cell_height(self) -> float:
        return (self.max_y - self.min_y) / self.rows

    def sector_index(self, col: int, row: int) -> int:
        return row * self.cols + col

    def sector_coords(self, sector: int) -> tuple[int, int]:
        return sector % self.cols, sector // self.cols

    def sector_center(self, sector: int) -> tuple[float, float]:
        col, row = self.sector_coords(sector)
        x = self.min_x + (col + 0.5) * self.cell_width
        y = self.min_y + (row + 0.5) * self.cell_height
        return x, y

    def sector_of(self, x: float, y: float) -> int:
        col = min(max(int((x - self.min_x) // self.cell_width), 0), self.cols - 1)
        row = min(max(int((y - self.min_y) // self.cell_height), 0), self.rows - 1)
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
    # Mirroring reflects about the center of this rectangle (same meaning
    # and default as SectorGrid.bounds). It must be the playable area, not
    # the nominal map square: Simple64's playable area is off-center in the
    # raw frame, so reflecting about the square's center mapped the two
    # spawn corners onto different ground -- the same canonical sector
    # covered different parts of the map depending on where you spawned.
    bounds: Bounds | None = None

    @staticmethod
    def from_home_position(
        map_size: int, home_x: float, home_y: float, bounds: Bounds | None = None
    ) -> "SpawnOrientation":
        min_x, min_y, max_x, max_y = bounds or (0.0, 0.0, float(map_size), float(map_size))
        return SpawnOrientation(
            map_size=map_size,
            mirror_x=home_x > (min_x + max_x) / 2,
            mirror_y=home_y > (min_y + max_y) / 2,
            bounds=bounds,
        )

    def to_canonical(self, x: float, y: float) -> tuple[float, float]:
        min_x, min_y, max_x, max_y = self.bounds or (0.0, 0.0, float(self.map_size), float(self.map_size))
        cx = (min_x + max_x - x) if self.mirror_x else x
        cy = (min_y + max_y - y) if self.mirror_y else y
        return cx, cy

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        # Mirroring is its own inverse.
        return self.to_canonical(x, y)


class _HasPosition(Protocol):
    x: float
    y: float


def sectors_of(units: Iterable[_HasPosition], grid: SectorGrid, orientation: SpawnOrientation) -> set[int]:
    return {grid.sector_of(*orientation.to_canonical(u.x, u.y)) for u in units}


def home_sector(
    command_center_pos: tuple[float, float] | None, grid: SectorGrid, orientation: SpawnOrientation
) -> int:
    """The sector the command center is actually in. "Home" used to be
    hard-coded as canonical sector 0 -- true enough with big cells, but
    once the grid was laid over the playable area (~7-unit cells) the base
    straddles several sectors and the command center isn't necessarily in
    the corner one. Everything that meant "home" (defense trigger, recall
    mask, home-defense penalty) silently pointed at the wrong cell, so a
    counterattack on the base while the army was away never registered.
    Falls back to sector 0 with no command center (e.g. after it's lost).
    """
    if command_center_pos is None:
        return 0
    return grid.sector_of(*orientation.to_canonical(*command_center_pos))
