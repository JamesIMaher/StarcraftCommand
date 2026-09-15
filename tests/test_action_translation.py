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


def test_move_army_to_empty_corner_sector_targets_its_center_not_the_map_corner():
    # Regression test: attack targets used to be biased all the way to the
    # literal map corner for edge sectors. The playable area is inset from
    # the nominal map size, so that point is unpathable and the attack-move
    # never completes -- confirmed live as the whole army parking at the
    # corner cliff forever. An empty sector is swept via its center.
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=30, y=30))

    calls = translator.translate(spec.move_action_for_sector(0), state, identity_orientation(spec))
    target_x, target_y = calls[0].arguments[2]
    center_x, center_y = spec.grid.sector_center(0)  # (8, 8) on a 64-map/4x4 grid
    assert abs(target_x - center_x) <= 1.0
    assert abs(target_y - center_y) <= 1.0


def test_move_army_seeks_a_known_enemy_structure_in_the_target_sector():
    # With known structures in the observation, "go to the sector with the
    # building" should resolve to "go to the building": the attack-move is
    # aimed at the structure's own (pathable) position, not the sector
    # center -- and at the nearest one to the army when there are several.
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=30, y=30))
    state.enemies.append(fake.enemy_unit(10, fake.UNIT_HATCHERY, x=2, y=2))  # sector 0, far corner
    state.enemies.append(fake.enemy_unit(11, fake.UNIT_HATCHERY, x=14, y=14))  # sector 0, nearer the army
    state.enemies.append(fake.enemy_unit(12, fake.UNIT_MARINE, x=5, y=5))  # a unit, not a structure

    calls = translator.translate(spec.move_action_for_sector(0), state, identity_orientation(spec))
    target_x, target_y = calls[0].arguments[2]
    assert abs(target_x - 14) <= 1.0
    assert abs(target_y - 14) <= 1.0


def test_move_army_to_home_sector_targets_the_command_center_not_the_corner():
    # The home sector's geometric center is in the map corner behind the
    # base, typically off the playable area -- recalling the army there
    # bunched it at the cliff edge. "Home" means the command center.
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=30, y=30))
    state.command_centers.append(fake.command_center(5, x=13, y=12))

    calls = translator.translate(spec.move_action_for_sector(0), state, identity_orientation(spec))
    target_x, target_y = calls[0].arguments[2]
    assert abs(target_x - 13) <= 1.0
    assert abs(target_y - 12) <= 1.0


def test_move_army_ignores_structures_outside_the_target_sector():
    spec = make_spec()
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=30, y=30))
    state.enemies.append(fake.enemy_unit(10, fake.UNIT_HATCHERY, x=60, y=60))  # last sector

    calls = translator.translate(spec.move_action_for_sector(0), state, identity_orientation(spec))
    target_x, target_y = calls[0].arguments[2]
    center_x, center_y = spec.grid.sector_center(0)
    assert abs(target_x - center_x) <= 1.0
    assert abs(target_y - center_y) <= 1.0


def test_move_army_targets_stay_strictly_inside_the_grid_bounds():
    # With the grid laid over the playable area, sector centers are inside
    # it by construction; jitter and a known structure's position could
    # still land on/over the boundary, and a point on or past it is
    # unpathable. Every target must land strictly inside.
    grid = SectorGrid(map_size=64, cols=4, rows=4, bounds=(20.0, 20.0, 44.0, 44.0))
    spec = ActionSpaceSpec(grid=grid)
    translator = ActionTranslator(spec, rng=random.Random(0))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.marines.append(fake.marine(1, x=30, y=30))
    state.enemies.append(fake.enemy_unit(10, fake.UNIT_HATCHERY, x=20.2, y=43.9))  # right at the edge

    for sector in range(spec.grid.num_sectors):
        calls = translator.translate(spec.move_action_for_sector(sector), state, identity_orientation(spec))
        target_x, target_y = calls[0].arguments[2]
        assert 20.0 < target_x < 44.0
        assert 20.0 < target_y < 44.0


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

    # Sector 0's canonical center is (8, 8) on a 64-map/4x4 grid; mirrored,
    # the real-world target should be near (56, 56).
    calls = translator.translate(spec.move_action_for_sector(0), state, mirrored)
    target_x, target_y = calls[0].arguments[2]
    assert abs(target_x - 56) <= 1.0
    assert abs(target_y - 56) <= 1.0


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
