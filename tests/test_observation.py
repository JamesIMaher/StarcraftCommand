import numpy as np

from sc2rl.env.game_state import GameState
from sc2rl.env.observation import featurize, observation_length
from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation
from tests.fakes import fake_pysc2 as fake


def make_grid():
    return SectorGrid(map_size=64, cols=4, rows=4)


def identity_orientation(grid: SectorGrid) -> SpawnOrientation:
    return SpawnOrientation(map_size=grid.map_size, mirror_x=False, mirror_y=False)


def test_observation_length_matches_featurize_output():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    vec = featurize(state, grid, max_game_loop=1000, orientation=identity_orientation(grid))
    assert vec.shape == (observation_length(grid),)
    assert vec.dtype == np.float32


def test_empty_state_is_all_finite_and_bounded():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    vec = featurize(state, grid, max_game_loop=1000, orientation=identity_orientation(grid))
    assert np.all(np.isfinite(vec))
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)


def test_minerals_normalized_and_clipped():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=5000, food_used=0, food_cap=15)
    vec = featurize(state, grid, max_game_loop=1000, orientation=identity_orientation(grid))
    assert vec[0] == 1.0  # clipped at the norm cap, never > 1.0


def test_marine_sector_placement_reflected_in_features():
    grid = make_grid()
    ts = fake.make_timestep(units=[
        fake.marine(1, x=8, y=8),  # sector 0
        fake.marine(2, x=56, y=56),  # last sector
    ])
    state = GameState.from_observation(ts)
    vec = featurize(state, grid, max_game_loop=1000, orientation=identity_orientation(grid))

    global_len = observation_length(grid) - 4 * grid.num_sectors
    sector0_friendly_count = vec[global_len + 4 * 0]
    last_sector = grid.num_sectors - 1
    last_sector_friendly_count = vec[global_len + 4 * last_sector]

    assert sector0_friendly_count > 0
    assert last_sector_friendly_count > 0
    # sectors with no marines stay at zero
    other_sector_count = vec[global_len + 4 * 5]
    assert other_sector_count == 0.0


def test_mirrored_orientation_flips_which_sector_a_unit_lands_in():
    # Regression test for the spawn-corner bug: the same real-world position
    # must land in sector 0 (home-relative) when home spawned in the opposite
    # corner, even though it would land in the last sector under an identity
    # (un-mirrored) orientation.
    grid = make_grid()
    ts = fake.make_timestep(units=[fake.marine(1, x=56, y=56)])
    state = GameState.from_observation(ts)
    mirrored = SpawnOrientation(map_size=grid.map_size, mirror_x=True, mirror_y=True)

    vec = featurize(state, grid, max_game_loop=1000, orientation=mirrored)

    global_len = observation_length(grid) - 4 * grid.num_sectors
    sector0_count = vec[global_len + 4 * 0]
    last_sector_count = vec[global_len + 4 * (grid.num_sectors - 1)]

    assert sector0_count > 0  # mirrored into the home-relative sector 0
    assert last_sector_count == 0.0


def test_enemy_race_one_hot():
    grid = make_grid()
    ts = fake.make_timestep(units=[
        fake.enemy_unit(1, fake.UNIT_HATCHERY, x=1, y=1),
    ])
    state = GameState.from_observation(ts)
    vec = featurize(state, grid, max_game_loop=1000, orientation=identity_orientation(grid))
    # race one-hot block is the 4 features right before the game-loop fraction
    race_block = vec[14:18]
    assert list(race_block) == [0.0, 0.0, 1.0, 0.0]  # terran, protoss, zerg, unknown


def test_unknown_enemy_race_when_no_town_hall_seen():
    grid = make_grid()
    ts = fake.make_timestep(units=[fake.enemy_unit(1, fake.UNIT_MARINE, x=1, y=1)])
    state = GameState.from_observation(ts)
    vec = featurize(state, grid, max_game_loop=1000, orientation=identity_orientation(grid))
    race_block = vec[14:18]
    assert race_block[3] == 1.0  # "unknown" flag set
