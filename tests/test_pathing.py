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


def test_reachable_from_keeps_only_the_component_connected_to_the_seed():
    # Pathable is not reachable: a cliff-top plateau with no ramp is marked
    # pathable but the army can never get there. Two pathable regions
    # separated by an unpathable wall; only the seed's side survives.
    pathable = np.ones((10, 10), dtype=bool)
    pathable[:, 5] = False  # vertical wall
    pm = PathingMap(pathable).reachable_from(2.0, 2.0)
    assert pm.is_pathable(1, 1)
    assert pm.is_pathable(4, 9)
    assert not pm.is_pathable(6, 1)
    assert not pm.is_pathable(9, 9)
    assert pm.nearest_pathable(8.0, 8.0, (6, 6, 10, 10)) is None
    assert pm.cell_count == 50


def test_reachable_from_seeds_at_the_nearest_pathable_cell_to_an_unpathable_point():
    # A command center's own footprint is unpathable; the seed must still
    # land next to it rather than nowhere.
    pathable = np.ones((10, 10), dtype=bool)
    pathable[3:6, 3:6] = False  # the building
    pm = PathingMap(pathable).reachable_from(4.0, 4.0)
    assert pm.cell_count == 100 - 9


def test_nearest_within_searches_a_square_around_the_point():
    pathable = np.zeros((10, 10), dtype=bool)
    pathable[4, 6] = True  # (x=6, y=4)
    pm = PathingMap(pathable)
    assert pm.nearest_within(4.0, 4.0, 3.0) == (6.5, 4.5)
    assert pm.nearest_within(4.0, 4.0, 2.0) is None


def test_is_pathable_is_false_outside_the_map():
    pm = PathingMap(np.ones((10, 10), dtype=bool))
    assert pm.is_pathable(3.2, 4.9)
    assert not pm.is_pathable(-1, 0)
    assert not pm.is_pathable(10, 0)
