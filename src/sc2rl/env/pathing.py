"""Reachable attack targets from the game's own pathing layer.

Clamping targets to the playable rectangle is not enough: the rectangle
contains unpathable terrain (cliffs, and on Simple64 the two corners that
aren't bases). And *pathable* is not enough either: the pathing layer marks
cliff-top plateaus and isolated pockets as pathable even when no ramp
connects them to where the army is, so "the pathable cell nearest the sector
center" can still be somewhere the army can never stand. An attack-move to
such a point makes units path to the closest connected point and stop at the
cliff wall -- confirmed live, twice, as marines "trying to reach areas they
can't reach." So targets are restricted to the connected component of
pathable ground containing the command center (a flood fill, once per reset).

The minimap `pathable` feature layer, at minimap size == raw_resolution, is
in exactly the raw coordinate frame unit positions use ([y][x], y already
flipped), so it can be indexed by raw coordinates directly. Pure numpy, no
PySC2 import -- the env extracts the layer and passes it in.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class PathingMap:
    def __init__(self, pathable: np.ndarray):
        self._pathable = np.asarray(pathable, dtype=bool)  # indexed [y][x]

    @property
    def shape(self) -> tuple[int, int]:
        return self._pathable.shape

    @property
    def grid(self) -> np.ndarray:
        """The bool [y][x] array itself (read-only use), for diagnostics."""
        return self._pathable

    @property
    def cell_count(self) -> int:
        return int(self._pathable.sum())

    def is_pathable(self, x: float, y: float) -> bool:
        h, w = self._pathable.shape
        ix, iy = int(x), int(y)
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        return bool(self._pathable[iy, ix])

    def reachable_from(self, x: float, y: float, search_radius: float = 8.0) -> "PathingMap":
        """The connected component (4-neighbour flood fill) of pathable ground
        containing the pathable cell nearest (x, y). A command center's own
        footprint is unpathable, so the seed is the nearest pathable cell
        within search_radius of it. Returns self unchanged if no seed exists."""
        seed = self.nearest_pathable(x, y, (x - search_radius, y - search_radius, x + search_radius, y + search_radius))
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
        return PathingMap(reachable)

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

    def nearest_within(self, x: float, y: float, radius: float) -> tuple[float, float] | None:
        return self.nearest_pathable(x, y, (x - radius, y - radius, x + radius, y + radius))
