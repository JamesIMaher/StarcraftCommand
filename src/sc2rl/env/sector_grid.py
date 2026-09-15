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
