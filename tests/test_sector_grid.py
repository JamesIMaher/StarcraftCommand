from sc2rl.env.sector_grid import SectorGrid


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
