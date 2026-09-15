from sc2rl.env.action_masking import MaskingConfig
from sc2rl.env.action_space import ActionSpaceSpec, FixedAction
from sc2rl.env.game_state import GameState
from sc2rl.env.scripted_policy import ScriptedPolicyConfig, scripted_action
from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation
from tests.fakes import fake_pysc2 as fake


def make_spec():
    return ActionSpaceSpec(grid=SectorGrid(map_size=64, cols=4, rows=4))


def identity_orientation(spec: ActionSpaceSpec) -> SpawnOrientation:
    return SpawnOrientation(map_size=spec.grid.map_size, mirror_x=False, mirror_y=False)


def test_builds_supply_depot_first_when_below_target():
    spec = make_spec()
    masking = MaskingConfig()
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(1))
    action = scripted_action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.BUILD_SUPPLY_DEPOT


def test_builds_barracks_once_depot_target_reached():
    spec = make_spec()
    masking = MaskingConfig()
    policy_config = ScriptedPolicyConfig(target_supply_depots=1, target_barracks=2)
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(1))
    state.supply_depots.append(fake.supply_depot(2, complete=True))
    action = scripted_action(state, spec, masking, identity_orientation(spec), policy_config)
    assert action == FixedAction.BUILD_BARRACKS


def test_trains_marine_once_base_targets_met():
    spec = make_spec()
    masking = MaskingConfig()
    policy_config = ScriptedPolicyConfig(target_supply_depots=1, target_barracks=1)
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(1))
    state.supply_depots.append(fake.supply_depot(2, complete=True))
    state.barracks.append(fake.barracks(3, complete=True))
    action = scripted_action(state, spec, masking, identity_orientation(spec), policy_config)
    assert action == FixedAction.TRAIN_MARINE


def test_advances_toward_far_sector_once_mobilized_and_home_safe():
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    policy_config = ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(4):
        state.marines.append(fake.marine(tag, x=1, y=1))
    action = scripted_action(state, spec, masking, identity_orientation(spec), policy_config)
    assert action == spec.move_action_for_sector(spec.grid.num_sectors - 1)


def test_returns_home_when_home_under_threat_and_undefended():
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    policy_config = ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(4):
        state.marines.append(fake.marine(tag, x=1, y=1))  # near canonical home sector 0
    state.enemies.append(fake.enemy_unit(99, fake.UNIT_MARINE, x=2, y=2))  # also in sector 0
    action = scripted_action(state, spec, masking, identity_orientation(spec), policy_config)
    assert action == spec.move_action_for_sector(0)


def test_no_op_when_nothing_legal():
    spec = make_spec()
    masking = MaskingConfig()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    action = scripted_action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.NO_OP
