import random

from pysc2.lib import actions as sc2_actions

from sc2rl.env.action_space import ActionSpaceSpec, FixedAction
from sc2rl.env.action_translation import ActionTranslator
from sc2rl.env.game_state import GameState
from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation
from tests.fakes import fake_pysc2 as fake


def make_spec():
    return ActionSpaceSpec(grid=SectorGrid(map_size=64, cols=4, rows=4))


def identity_orientation(spec: ActionSpaceSpec) -> SpawnOrientation:
    return SpawnOrientation(map_size=spec.grid.map_size, mirror_x=False, mirror_y=False)


def test_no_op_translates_to_raw_no_op():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    calls = translator.translate(FixedAction.NO_OP, state, identity_orientation(spec))
    assert len(calls) == 1
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_build_supply_depot_targets_an_scv():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(42, x=10, y=10, idle=True))
    calls = translator.translate(FixedAction.BUILD_SUPPLY_DEPOT, state, identity_orientation(spec))
    assert len(calls) == 1
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.Build_SupplyDepot_pt.id
    # unit_tags argument should reference our SCV's tag
    assert 42 in calls[0].arguments[1]


def test_build_supply_depot_uses_actively_mining_scv_not_just_idle_ones():
    # Regression test: a live game's starting SCVs are continuously
    # re-issuing harvest orders (order_length != 0), so they are never
    # "idle" in that sense -- but a build order interrupts mining just fine,
    # and requiring an idle worker meant build actions silently resolved to
    # no-ops for the entire game in practice.
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(42, x=10, y=10, idle=False))
    calls = translator.translate(FixedAction.BUILD_SUPPLY_DEPOT, state, identity_orientation(spec))
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.Build_SupplyDepot_pt.id
    assert 42 in calls[0].arguments[1]


def test_build_supply_depot_no_op_when_no_scv_at_all():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    calls = translator.translate(FixedAction.BUILD_SUPPLY_DEPOT, state, identity_orientation(spec))
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_train_marine_targets_completed_barracks_round_robin():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.barracks.append(fake.barracks(1, complete=True))
    state.barracks.append(fake.barracks(2, complete=True))
    orientation = identity_orientation(spec)

    first = translator.translate(FixedAction.TRAIN_MARINE, state, orientation)
    second = translator.translate(FixedAction.TRAIN_MARINE, state, orientation)

    assert first[0].function == sc2_actions.RAW_FUNCTIONS.Train_Marine_quick.id
    first_tag = first[0].arguments[1][0]
    second_tag = second[0].arguments[1][0]
    assert first_tag != second_tag  # round-robin, not always the same barracks


def test_train_marine_ignores_incomplete_barracks():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.barracks.append(fake.barracks(1, complete=False))
    calls = translator.translate(FixedAction.TRAIN_MARINE, state, identity_orientation(spec))
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_move_army_to_corner_sector_reaches_the_true_map_edge():
    # Direct regression test for the reported bug: marines attacking the
    # farthest (corner) sector must actually reach the true map boundary,
    # not stop ~11 units short at the old cell-center target, or a building
    # tucked into the corner is never engaged.
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=0, y=0))

    last_sector = spec.grid.num_sectors - 1
    calls = translator.translate(spec.move_action_for_sector(last_sector), state, identity_orientation(spec))
    target_x, target_y = calls[0].arguments[2]
    assert target_x >= 60  # within a marine's weapon range of the true (64, 64) corner
    assert target_y >= 60


def test_move_army_issues_attack_pt_for_every_marine():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=0, y=0))
    state.marines.append(fake.marine(2, x=1, y=1))

    action_index = spec.move_action_for_sector(5)
    calls = translator.translate(action_index, state, identity_orientation(spec))

    assert len(calls) == 2
    for call in calls:
        assert call.function == sc2_actions.RAW_FUNCTIONS.Attack_pt.id

    target_tags = {call.arguments[1][0] for call in calls}
    assert target_tags == {1, 2}


def test_move_army_targets_are_un_mirrored_back_to_world_coordinates():
    # Regression test for the spawn-corner bug: sector_center() is in
    # home-relative (canonical) space, so under a mirrored orientation the
    # actual Attack_pt target must be mirrored back to real map coordinates,
    # not issued directly in canonical space.
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=0, y=0))
    mirrored = SpawnOrientation(map_size=spec.grid.map_size, mirror_x=True, mirror_y=True)

    # Sector 0's canonical attack target is the true edge (0, 0) on a
    # 64-map/4x4 grid; mirrored, the real-world target should be near
    # (64, 64), clamped to the map boundary.
    calls = translator.translate(spec.move_action_for_sector(0), state, mirrored)
    target_x, target_y = calls[0].arguments[2]
    assert 60 <= target_x <= 64
    assert 60 <= target_y <= 64


def test_move_army_no_op_with_no_marines():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    calls = translator.translate(spec.move_action_for_sector(0), state, identity_orientation(spec))
    assert calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_translate_named_matches_translate_by_index():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    orientation = identity_orientation(spec)
    by_index = translator.translate(FixedAction.NO_OP, state, orientation)
    by_name = translator.translate_named("no_op", state, orientation)
    assert by_index[0].function == by_name[0].function
