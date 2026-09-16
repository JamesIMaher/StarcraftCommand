import pytest

from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation, home_sector, real_corner_sectors


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


def test_grid_with_bounds_covers_only_the_playable_rectangle():
    # Regression test: the grid used to be laid over the full nominal map
    # square, but Simple64's playable area is a ~43-unit square sitting
    # off-center in the 64x64 raw frame -- so the entire last column and
    # first row were sectors nothing could ever reach, and the exploration
    # incentives kept pulling the army toward those edges.
    grid = SectorGrid(map_size=64, cols=4, rows=4, bounds=(8.0, 13.3, 50.7, 56.0))
    assert grid.cell_width == (50.7 - 8.0) / 4
    assert grid.cell_height == (56.0 - 13.3) / 4
    assert grid.sector_center(0) == (8.0 + grid.cell_width / 2, 13.3 + grid.cell_height / 2)
    assert grid.sector_of(8.1, 13.4) == 0
    assert grid.sector_of(50.6, 55.9) == grid.num_sectors - 1
    # Coordinates outside the bounds clamp to the nearest edge sector.
    assert grid.sector_of(0.0, 0.0) == 0
    assert grid.sector_of(63.9, 63.9) == grid.num_sectors - 1
    for sector in range(grid.num_sectors):
        cx, cy = grid.sector_center(sector)
        assert grid.sector_of(cx, cy) == sector
        assert 8.0 < cx < 50.7 and 13.3 < cy < 56.0


def test_with_bounds_keeps_sector_count():
    grid = SectorGrid(map_size=64, cols=6, rows=6).with_bounds((8.0, 13.3, 50.7, 56.0))
    assert grid.num_sectors == 36
    assert grid.bounds == (8.0, 13.3, 50.7, 56.0)


def test_spawn_orientation_mirrors_about_the_bounds_center_not_the_map_center():
    # The playable area is off-center in the raw frame, so reflecting about
    # the map square's center mapped the two spawn corners onto different
    # ground. Reflecting about the bounds' center makes them symmetric: the
    # bounds' far corner lands exactly on its near corner.
    bounds = (8.0, 13.3, 50.7, 56.0)
    orientation = SpawnOrientation.from_home_position(64, home_x=45, home_y=50, bounds=bounds)
    assert orientation.mirror_x and orientation.mirror_y
    cx, cy = orientation.to_canonical(50.7, 56.0)
    assert (cx, cy) == pytest.approx((8.0, 13.3))
    # A home just past the map-square center but on the near side of the
    # bounds' center is NOT mirrored.
    near = SpawnOrientation.from_home_position(64, home_x=28, home_y=33, bounds=bounds)
    assert not near.mirror_x and not near.mirror_y


def test_home_sector_is_the_command_centers_sector_with_fallback_to_zero():
    grid = make_grid()
    identity = SpawnOrientation(map_size=64, mirror_x=False, mirror_y=False)
    assert home_sector((30.0, 30.0), grid, identity) == 5
    assert home_sector(None, grid, identity) == 0
    mirrored = SpawnOrientation.from_home_position(64, home_x=56, home_y=56)
    assert home_sector((56.0, 56.0), grid, mirrored) == 0  # canonicalized back toward the origin


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


def test_real_corner_sectors_identity_matches_the_raw_grid_layout():
    grid = make_grid()
    identity = SpawnOrientation(map_size=64, mirror_x=False, mirror_y=False)
    assert real_corner_sectors(grid, identity) == {
        "top-left": 0, "top-right": 3, "bottom-left": 12, "bottom-right": 15,
    }


def test_real_corner_sectors_flip_diagonally_when_home_is_in_the_opposite_corner():
    # Regression test: reported live as "had to say bottom left to get the
    # marine to the screen's actual top right" -- home spawning in the real
    # bottom-right corner mirrors both axes, so the *canonical* sector
    # number for the real top-right corner is the same one that identity
    # orientation would call bottom-left, and vice versa.
    grid = make_grid()
    mirrored = SpawnOrientation.from_home_position(map_size=64, home_x=50, home_y=50)
    assert mirrored.mirror_x and mirrored.mirror_y
    corners = real_corner_sectors(grid, mirrored)
    assert corners["top-right"] == 12  # what identity orientation calls bottom-left
    assert corners["bottom-left"] == 3  # what identity orientation calls top-right
    assert corners["top-left"] == 15
    assert corners["bottom-right"] == 0


def test_real_corner_sectors_only_flips_the_mirrored_axis():
    grid = make_grid()
    orientation = SpawnOrientation.from_home_position(map_size=64, home_x=50, home_y=10)  # mirror_x only
    corners = real_corner_sectors(grid, orientation)
    assert corners["top-left"] == 3  # x flipped, y unchanged
    assert corners["top-right"] == 0
    assert corners["bottom-left"] == 15
    assert corners["bottom-right"] == 12


def test_spawn_orientation_to_world_is_its_own_inverse():
    orientation = SpawnOrientation.from_home_position(map_size=64, home_x=50, home_y=50)
    for x, y in [(0, 0), (63, 63), (10, 40), (40, 10)]:
        cx, cy = orientation.to_canonical(x, y)
        wx, wy = orientation.to_world(cx, cy)
        assert (wx, wy) == (x, y)
