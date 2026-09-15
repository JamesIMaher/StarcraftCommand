import random

from pysc2.lib import actions as sc2_actions

from sc2rl.env.action_space import ActionSpaceSpec, FixedAction
from sc2rl.env.action_translation import ActionTranslator
from sc2rl.env.game_state import GameState
from sc2rl.env.sector_grid import SectorGrid
from tests.fakes import fake_pysc2 as fake


def make_spec():
    return ActionSpaceSpec(grid=SectorGrid(map_size=64, cols=4, rows=4))


def test_no_op_translates_to_raw_no_op():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    calls = translator.translate(FixedAction.NO_OP, state)
    assert len(calls) == 1
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_build_supply_depot_targets_an_idle_scv():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(42, x=10, y=10, idle=True))
    calls = translator.translate(FixedAction.BUILD_SUPPLY_DEPOT, state)
    assert len(calls) == 1
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.Build_SupplyDepot_pt.id
    # unit_tags argument should reference our SCV's tag
    assert 42 in calls[0].arguments[1]


def test_build_supply_depot_no_op_when_no_idle_scv():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    calls = translator.translate(FixedAction.BUILD_SUPPLY_DEPOT, state)
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_train_marine_targets_completed_barracks_round_robin():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.barracks.append(fake.barracks(1, complete=True))
    state.barracks.append(fake.barracks(2, complete=True))

    first = translator.translate(FixedAction.TRAIN_MARINE, state)
    second = translator.translate(FixedAction.TRAIN_MARINE, state)

    assert first[0].function == sc2_actions.RAW_FUNCTIONS.Train_Marine_quick.id
    first_tag = first[0].arguments[1][0]
    second_tag = second[0].arguments[1][0]
    assert first_tag != second_tag  # round-robin, not always the same barracks


def test_train_marine_ignores_incomplete_barracks():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.barracks.append(fake.barracks(1, complete=False))
    calls = translator.translate(FixedAction.TRAIN_MARINE, state)
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_move_army_issues_attack_pt_for_every_marine():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=0, y=0))
    state.marines.append(fake.marine(2, x=1, y=1))

    action_index = spec.move_action_for_sector(5)
    calls = translator.translate(action_index, state)

    assert len(calls) == 2
    for call in calls:
        assert call.function == sc2_actions.RAW_FUNCTIONS.Attack_pt.id

    target_tags = {call.arguments[1][0] for call in calls}
    assert target_tags == {1, 2}


def test_move_army_no_op_with_no_marines():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    calls = translator.translate(spec.move_action_for_sector(0), state)
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_translate_named_matches_translate_by_index():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    by_index = translator.translate(FixedAction.NO_OP, state)
    by_name = translator.translate_named("no_op", state)
    assert by_index[0].function == by_name[0].function
