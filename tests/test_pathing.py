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


def test_prefer_interior_avoids_cliff_edge_cells():
    # A 3-wide pathable strip: its middle column is interior, its two outer
    # columns border unpathable ground. The nearest cell to a point just
    # outside the strip is an edge cell; with prefer_interior the interior
    # column wins even though it's farther.
    pathable = np.zeros((10, 10), dtype=bool)
    pathable[:, 3:6] = True
    pm = PathingMap(pathable)
    assert pm.nearest_pathable(7.0, 5.5, (0, 0, 10, 10)) == (5.5, 5.5)
    assert pm.nearest_pathable(7.0, 5.5, (0, 0, 10, 10), prefer_interior=True) == (4.5, 5.5)


def test_prefer_interior_still_returns_an_edge_cell_when_nothing_else_exists():
    pathable = np.zeros((10, 10), dtype=bool)
    pathable[5, 5] = True
    pm = PathingMap(pathable)
    assert pm.nearest_pathable(0.0, 0.0, (0, 0, 10, 10), prefer_interior=True) == (5.5, 5.5)


def test_same_level_snapping_prefers_the_plateau_over_the_cliff_foot():
    # A building on a plateau (height 200) whose footprint is unpathable,
    # with low ground (height 100) directly below it across a cliff line.
    # The Euclidean-nearest reachable cell is on the low ground; the target
    # must instead be the plateau cell beside the building.
    pathable = np.ones((20, 20), dtype=bool)
    height = np.full((20, 20), 100, dtype=np.int32)
    height[:10, :] = 200  # rows 0..9 are the plateau
    pathable[10, :] = False  # cliff line between the levels
    pathable[10, 0] = True  # a ramp at the far left keeps both levels connected
    pathable[4:10, 4:17] = False  # a wide building's footprint, right at the plateau's edge
    pm = PathingMap(pathable, height).reachable_from(15.0, 15.0)
    structure = (10.0, 8.5)  # its center, inside the footprint
    ref = pm.height_at(*structure)
    assert ref == 200
    low = pm.nearest_within(*structure, 6.0)
    assert low[1] >= 11  # nearest by distance: just below the cliff (3 away vs 5 to the plateau row above)
    same_level = pm.nearest_within(*structure, 6.0, same_level_as=ref)
    assert same_level[1] < 4  # on the plateau, above the building


def test_is_pathable_is_false_outside_the_map():
    pm = PathingMap(np.ones((10, 10), dtype=bool))
    assert pm.is_pathable(3.2, 4.9)
    assert not pm.is_pathable(-1, 0)
    assert not pm.is_pathable(10, 0)
