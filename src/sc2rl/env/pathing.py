"""Reachable, same-level attack targets from the game's own terrain layers.

Clamping targets to the playable rectangle is not enough: the rectangle
contains unpathable terrain (cliffs, and on Simple64 the two corners that
aren't bases). *Pathable* is not enough either: the pathing layer marks
cliff-top plateaus and isolated pockets as pathable even when no ramp
connects them, so targets are restricted to the connected component of
pathable ground containing the command center (a flood fill, once per
reset). And *reachable* is still not enough: Simple64 is two plateaus with
cliff lines between them, and the Euclidean-nearest reachable cell to a
point on a plateau is often at the FOOT of the cliff directly below it --
legal, reachable, and on the wrong level. Marines sent there stand under
the cliff with no vision of the building above, and for the scripted
teacher that sector's structure then never dies, so it re-targets it
forever. Confirmed live (three times, as "trying to reach cliff tiles").
So a known structure's target must be on the same terrain level as the
structure (the minimap `height_map` layer), and a sector's sweep target
prefers interior cells over cliff-edge cells.

The minimap `pathable` and `height_map` feature layers, at minimap size ==
raw_resolution, are in exactly the raw coordinate frame unit positions use
([y][x], y already flipped), so they can be indexed by raw coordinates.
Pure numpy, no PySC2 import -- the env extracts the layers and passes them in.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# height_map is 8-bit; cells on one terrain level share a value (ramps
# interpolate between levels, cliff levels differ by far more than this).
_SAME_LEVEL_TOLERANCE = 4


class PathingMap:
    def __init__(self, pathable: np.ndarray, height: np.ndarray | None = None):
        self._pathable = np.asarray(pathable, dtype=bool)  # indexed [y][x]
        self._height = None if height is None else np.asarray(height, dtype=np.int32)
        padded = np.pad(self._pathable, 1, constant_values=False)
        neighbours_pathable = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        self._interior = self._pathable & neighbours_pathable

    @property
    def shape(self) -> tuple[int, int]:
        return self._pathable.shape

    @property
    def grid(self) -> np.ndarray:
        """The bool [y][x] array itself (read-only use), for diagnostics."""
        return self._pathable

    @property
    def height(self) -> np.ndarray | None:
        return self._height

    @property
    def cell_count(self) -> int:
        return int(self._pathable.sum())

    def is_pathable(self, x: float, y: float) -> bool:
        h, w = self._pathable.shape
        ix, iy = int(x), int(y)
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        return bool(self._pathable[iy, ix])

    def height_at(self, x: float, y: float) -> int | None:
        if self._height is None:
            return None
        h, w = self._pathable.shape
        ix, iy = int(x), int(y)
        if not (0 <= ix < w and 0 <= iy < h):
            return None
        return int(self._height[iy, ix])

    def reachable_from(self, x: float, y: float, search_radius: float = 8.0) -> "PathingMap":
        """The connected component (4-neighbour flood fill) of pathable ground
        containing the pathable cell nearest (x, y). A command center's own
        footprint is unpathable, so the seed is the nearest pathable cell
        within search_radius of it. Returns self unchanged if no seed exists."""
        seed = self.nearest_within(x, y, search_radius)
        if seed is None:
            return self
        h, w = self._pathable.shape
        reachable = np.zeros_like(self._pathable)
        sx, sy = int(seed[0]), int(seed[1])
        queue = deque([(sx, sy)])
        reachable[sy, sx] = True
        while queue:
            cx, cy = queue.popleft()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < w and 0 <= ny < h and self._pathable[ny, nx] and not reachable[ny, nx]:
                    reachable[ny, nx] = True
                    queue.append((nx, ny))
        return PathingMap(reachable, self._height)

    def nearest_pathable(
        self,
        x: float,
        y: float,
        rect: tuple[float, float, float, float],
        prefer_interior: bool = False,
        same_level_as: int | None = None,
    ) -> tuple[float, float] | None:
        """Center of the best pathable cell inside `rect` (min_x, min_y,
        max_x, max_y) for a target at (x, y), or None if nothing in the rect
        is pathable. "Best" is nearest, except that with `prefer_interior`
        any cell with an unpathable 4-neighbour (a cliff edge) loses to any
        interior cell, and with `same_level_as` (a height_map value) any cell
        on a different terrain level loses to any cell on that level."""
        min_x, min_y, max_x, max_y = rect
        h, w = self._pathable.shape
        x0, x1 = max(int(math.floor(min_x)), 0), min(int(math.ceil(max_x)), w)
        y0, y1 = max(int(math.floor(min_y)), 0), min(int(math.ceil(max_y)), h)
        if x1 <= x0 or y1 <= y0:
            return None
        ys, xs = np.nonzero(self._pathable[y0:y1, x0:x1])
        if len(xs) == 0:
            return None
        ys, xs = ys + y0, xs + x0
        cx = xs + 0.5
        cy = ys + 0.5
        score = (cx - x) ** 2 + (cy - y) ** 2
        if prefer_interior:
            score = score + np.where(self._interior[ys, xs], 0.0, 1e4)
        if same_level_as is not None and self._height is not None:
            off_level = np.abs(self._height[ys, xs] - same_level_as) > _SAME_LEVEL_TOLERANCE
            score = score + np.where(off_level, 1e6, 0.0)
        i = int(np.argmin(score))
        return float(cx[i]), float(cy[i])

    def nearest_within(
        self, x: float, y: float, radius: float, same_level_as: int | None = None
    ) -> tuple[float, float] | None:
        return self.nearest_pathable(x, y, (x - radius, y - radius, x + radius, y + radius), same_level_as=same_level_as)
