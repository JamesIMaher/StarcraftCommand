from sc2rl.env.game_state import GameState
from tests.fakes import fake_pysc2 as fake


def test_from_observation_sorts_units_by_alliance_and_type():
    ts = fake.make_timestep(
        units=[
            fake.scv(1, idle=True),
            fake.scv(2, idle=False),
            fake.marine(3),
            fake.command_center(4),
            fake.supply_depot(5, complete=True),
            fake.supply_depot(6, complete=False),
            fake.barracks(7, complete=True),
            fake.enemy_unit(8, fake.UNIT_MARINE),
        ],
        minerals=250,
        food_used=10,
        food_cap=15,
        game_loop=42,
    )
    state = GameState.from_observation(ts)

    assert state.minerals == 250
    assert state.food_used == 10
    assert state.food_cap == 15
    assert state.game_loop == 42

    assert len(state.scvs) == 2
    assert len(state.idle_scvs) == 1
    assert len(state.marines) == 1
    assert len(state.command_centers) == 1
    assert len(state.supply_depots) == 2
    assert len(state.complete_supply_depots) == 1
    assert len(state.barracks) == 1
    assert len(state.complete_barracks) == 1
    assert len(state.enemies) == 1


def test_enemy_command_center_marks_terran_race():
    ts = fake.make_timestep(units=[fake.enemy_unit(1, fake.UNIT_COMMAND_CENTER)])
    state = GameState.from_observation(ts)
    assert state.enemy_race_terran
    assert not state.enemy_race_protoss
    assert not state.enemy_race_zerg


def test_supply_headroom():
    ts = fake.make_timestep(minerals=0, food_used=12, food_cap=15)
    state = GameState.from_observation(ts)
    assert state.supply_headroom == 3


def test_command_center_pos_none_when_missing():
    ts = fake.make_timestep(units=[])
    state = GameState.from_observation(ts)
    assert state.command_center_pos is None


def test_command_center_pos_returns_coords():
    ts = fake.make_timestep(units=[fake.command_center(1, x=12.5, y=7.5)])
    state = GameState.from_observation(ts)
    assert state.command_center_pos == (12.5, 7.5)


def test_score_cumulative_fields_parsed():
    ts = fake.make_timestep(
        total_value_units=100, total_value_structures=75, killed_value_units=25, killed_value_structures=10,
    )
    state = GameState.from_observation(ts)
    assert state.total_value_units == 100
    assert state.total_value_structures == 75
    assert state.killed_value_units == 25
    assert state.killed_value_structures == 10
