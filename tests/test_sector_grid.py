from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation


def make_grid():
    return SectorGrid(map_size=64, cols=4, rows=4)


def test_num_sectors():
    grid = make_grid()
    assert grid.num_sectors == 16


def test_sector_center_top_left():
    grid = make_grid()
    x, y = grid.sector_center(0)
    assert x == 8.0
    assert y == 8.0


def test_sector_center_bottom_right():
    grid = make_grid()
    last_sector = grid.num_sectors - 1
    x, y = grid.sector_center(last_sector)
    assert x == 56.0
    assert y == 56.0


def test_sector_of_matches_center_round_trip():
    grid = make_grid()
    for sector in range(grid.num_sectors):
        cx, cy = grid.sector_center(sector)
        assert grid.sector_of(cx, cy) == sector


def test_sector_of_clamps_out_of_bounds():
    grid = make_grid()
    assert grid.sector_of(-5, -5) == grid.sector_of(0, 0)
    assert grid.sector_of(999, 999) == grid.sector_of(63, 63)


def test_sector_attack_target_reaches_true_corner():
    # Regression test: sector_center() stops at the cell's geometric center,
    # ~11 units short of the true corner on a 64-map/4x4 grid -- well beyond
    # marine weapon/sight range, so an attack order using it can arrive and
    # stop short of a building tucked into the corner. sector_attack_target()
    # must reach the actual boundary instead.
    grid = make_grid()
    top_left = grid.sector_attack_target(0)
    assert top_left == (0.0, 0.0)

    last_sector = grid.num_sectors - 1
    bottom_right = grid.sector_attack_target(last_sector)
    assert bottom_right == (64.0, 64.0)


def test_sector_attack_target_only_biases_the_edge_axis():
    # A sector on the top edge but not a side edge (e.g. col=1, row=0) should
    # only be pushed to the true edge on the row axis; the column axis keeps
    # its normal center, matching sector_center().
    grid = make_grid()
    edge_sector = grid.sector_index(col=1, row=0)
    x, y = grid.sector_attack_target(edge_sector)
    assert x == 24.0  # unchanged interior-column center
    assert y == 0.0   # biased to the true top edge


def test_sector_attack_target_matches_center_for_interior_sectors():
    grid = make_grid()
    interior_sector = grid.sector_index(col=1, row=1)
    assert grid.sector_attack_target(interior_sector) == grid.sector_center(interior_sector)


def test_friendly_quadrant_top_left():
    grid = make_grid()
    min_x, max_x, min_y, max_y = grid.friendly_quadrant(10, 10)
    assert (min_x, max_x, min_y, max_y) == (0.0, 32.0, 0.0, 32.0)


def test_friendly_quadrant_bottom_right():
    grid = make_grid()
    min_x, max_x, min_y, max_y = grid.friendly_quadrant(50, 50)
    assert (min_x, max_x, min_y, max_y) == (32.0, 64.0, 32.0, 64.0)


def test_friendly_quadrant_top_right_not_swapped_with_max_x():
    # Regression test for the old repo's ScreenSectors.getFriendlyArea bug,
    # which returned maxX in place of maxY for the Y bound.
    grid = make_grid()
    min_x, max_x, min_y, max_y = grid.friendly_quadrant(50, 10)
    assert (min_x, max_x, min_y, max_y) == (32.0, 64.0, 0.0, 32.0)
    assert max_y != max_x


def test_spawn_orientation_identity_when_home_in_top_left():
    orientation = SpawnOrientation.from_home_position(map_size=64, home_x=10, home_y=10)
    assert orientation.mirror_x is False
    assert orientation.mirror_y is False
    assert orientation.to_canonical(30, 40) == (30, 40)


def test_spawn_orientation_mirrors_when_home_in_bottom_right():
    orientation = SpawnOrientation.from_home_position(map_size=64, home_x=50, home_y=50)
    assert orientation.mirror_x is True
    assert orientation.mirror_y is True
    # home itself should land near the canonical origin
    cx, cy = orientation.to_canonical(50, 50)
    assert (cx, cy) == (14, 14)


def test_spawn_orientation_mirrors_only_the_axis_that_needs_it():
    orientation = SpawnOrientation.from_home_position(map_size=64, home_x=50, home_y=10)
    assert orientation.mirror_x is True
    assert orientation.mirror_y is False
    assert orientation.to_canonical(10, 10) == (54, 10)


def test_spawn_orientation_to_world_is_its_own_inverse():
    orientation = SpawnOrientation.from_home_position(map_size=64, home_x=50, home_y=50)
    for x, y in [(0, 0), (63, 63), (10, 40), (40, 10)]:
        cx, cy = orientation.to_canonical(x, y)
        wx, wy = orientation.to_world(cx, cy)
        assert (wx, wy) == (x, y)
