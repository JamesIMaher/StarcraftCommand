import numpy as np

from sc2rl.env.pathing import PathingMap


def test_nearest_pathable_picks_the_closest_pathable_cell_inside_the_rect():
    pathable = np.zeros((10, 10), dtype=bool)
    pathable[2, 3] = True  # (x=3, y=2)
    pathable[8, 8] = True  # (x=8, y=8), farther
    pm = PathingMap(pathable)
    assert pm.nearest_pathable(5.0, 5.0, (0, 0, 10, 10)) == (3.5, 2.5)


def test_nearest_pathable_ignores_pathable_cells_outside_the_rect():
    pathable = np.zeros((10, 10), dtype=bool)
    pathable[2, 3] = True
    pm = PathingMap(pathable)
    assert pm.nearest_pathable(7.0, 7.0, (5, 5, 10, 10)) is None


def test_nearest_pathable_returns_none_for_a_fully_unpathable_rect():
    pm = PathingMap(np.zeros((10, 10), dtype=bool))
    assert pm.nearest_pathable(5.0, 5.0, (0, 0, 10, 10)) is None


def test_is_pathable_is_false_outside_the_map():
    pm = PathingMap(np.ones((10, 10), dtype=bool))
    assert pm.is_pathable(3.2, 4.9)
    assert not pm.is_pathable(-1, 0)
    assert not pm.is_pathable(10, 0)
