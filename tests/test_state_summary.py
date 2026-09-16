from sc2rl.command.state_summary import describe_state_for_llm
from sc2rl.env.game_state import GameState
from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation
from tests.fakes import fake_pysc2 as fake


def make_grid():
    return SectorGrid(map_size=64, cols=4, rows=4)


def identity_orientation(grid: SectorGrid) -> SpawnOrientation:
    return SpawnOrientation(map_size=grid.map_size, mirror_x=False, mirror_y=False)


def test_describe_state_includes_resources_and_home_sector():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=150, food_used=10, food_cap=15)
    state.command_centers.append(fake.command_center(1, x=30, y=30))  # sector 5 on 4x4/64
    text = describe_state_for_llm(state, grid, identity_orientation(grid))
    assert "Minerals: 150" in text
    assert "10/15" in text
    assert "sector 5" in text


def test_describe_state_lists_known_enemy_and_structure_sectors():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.enemies.append(fake.enemy_unit(1, fake.UNIT_HATCHERY, x=58, y=58))  # sector 15
    text = describe_state_for_llm(state, grid, identity_orientation(grid))
    assert "[15]" in text
    assert "structures" in text.lower()


def test_describe_state_notes_no_known_enemies():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    text = describe_state_for_llm(state, grid, identity_orientation(grid))
    assert "No enemy" in text


def test_describe_state_mentions_garrison_and_mobilization():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for i in range(20):
        state.marines.append(fake.marine(i))
    text = describe_state_for_llm(state, grid, identity_orientation(grid), mobilized=True, garrison_size=4)
    assert "20 total" in text
    assert "4 permanently held at home" in text
    assert "is currently mobilized" in text


def test_describe_state_notes_when_not_mobilized():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    text = describe_state_for_llm(state, grid, identity_orientation(grid), mobilized=False)
    assert "is NOT currently mobilized" in text


def test_describe_state_mentions_player_controlled_count():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    text = describe_state_for_llm(state, grid, identity_orientation(grid), player_controlled_count=2)
    assert "2 currently under direct PLAYER control" in text


def test_describe_state_omits_player_controlled_note_when_zero():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    text = describe_state_for_llm(state, grid, identity_orientation(grid), player_controlled_count=0)
    assert "PLAYER control" not in text


def test_describe_state_lists_banned_actions():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    text = describe_state_for_llm(
        state, grid, identity_orientation(grid), banned_actions=("build_supply_depot", "train_marine"),
    )
    assert "BAN the autonomous policy" in text
    assert "build_supply_depot" in text
    assert "train_marine" in text


def test_describe_state_omits_directive_line_when_no_bans():
    grid = make_grid()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    text = describe_state_for_llm(state, grid, identity_orientation(grid))
    assert "BAN" not in text
