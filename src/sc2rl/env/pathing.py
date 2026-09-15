"""Pathable attack targets from the game's own pathing layer.

Clamping targets to the playable rectangle is not enough: the rectangle
contains unpathable terrain (cliffs, and on Simple64 the two corners that
aren't bases). A sector whose center is on a cliff sends an attack-move to
the nearest reachable point at the cliff's edge, where the army then parks
-- confirmed live as marines "trying to reach areas off the screen." The
minimap `pathable` feature layer answers this directly, and at minimap size
== raw_resolution it is in exactly the raw coordinate frame unit positions
use ([y][x], y already flipped), so it can be indexed by raw coordinates.
Pure numpy, no PySC2 import -- the env extracts the layer and passes it in.
"""

from __future__ import annotations

import math

import numpy as np


class PathingMap:
    def __init__(self, pathable: np.ndarray):
        self._pathable = np.asarray(pathable, dtype=bool)  # indexed [y][x]

    @property
    def shape(self) -> tuple[int, int]:
        return self._pathable.shape

    def is_pathable(self, x: float, y: float) -> bool:
        h, w = self._pathable.shape
        ix, iy = int(x), int(y)
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        return bool(self._pathable[iy, ix])

    def nearest_pathable(
        self, x: float, y: float, rect: tuple[float, float, float, float]
    ) -> tuple[float, float] | None:
        """Center of the pathable cell inside `rect` (min_x, min_y, max_x,
        max_y) nearest to (x, y), or None if nothing in the rect is pathable."""
        min_x, min_y, max_x, max_y = rect
        h, w = self._pathable.shape
        x0, x1 = max(int(math.floor(min_x)), 0), min(int(math.ceil(max_x)), w)
        y0, y1 = max(int(math.floor(min_y)), 0), min(int(math.ceil(max_y)), h)
        if x1 <= x0 or y1 <= y0:
            return None
        ys, xs = np.nonzero(self._pathable[y0:y1, x0:x1])
        if len(xs) == 0:
            return None
        cx = xs + x0 + 0.5
        cy = ys + y0 + 0.5
        i = int(np.argmin((cx - x) ** 2 + (cy - y) ** 2))
        return float(cx[i]), float(cy[i])
