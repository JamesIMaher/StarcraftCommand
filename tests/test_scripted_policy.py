from sc2rl.env.action_masking import MaskingConfig
from sc2rl.env.action_space import ActionSpaceSpec, FixedAction
from sc2rl.env.game_state import GameState
from sc2rl.env.scripted_policy import ScriptedPolicy, ScriptedPolicyConfig
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
    policy = ScriptedPolicy()
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.BUILD_SUPPLY_DEPOT


def test_builds_barracks_once_depot_target_reached():
    spec = make_spec()
    masking = MaskingConfig()
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(1))
    state.supply_depots.append(fake.supply_depot(2, complete=True))
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=1, target_barracks=2))
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.BUILD_BARRACKS


def test_trains_marine_once_base_targets_met():
    spec = make_spec()
    masking = MaskingConfig()
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(1))
    state.supply_depots.append(fake.supply_depot(2, complete=True))
    state.barracks.append(fake.barracks(3, complete=True))
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=1, target_barracks=1))
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.TRAIN_MARINE


def test_holds_position_below_attack_threshold_even_though_mobilized():
    # Regression test: mobilized (>= min_marines_to_move, so movement is
    # legal) is not the same as ready to commit to an offensive
    # (attack_threshold) -- below the threshold the teacher should hold
    # rather than advance piecemeal.
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(4):
        state.marines.append(fake.marine(tag, x=1, y=1))
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.NO_OP


def test_commits_to_search_and_destroy_once_attack_threshold_reached():
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        state.marines.append(fake.marine(tag, x=1, y=1))
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == spec.move_action_for_sector(spec.grid.num_sectors - 1)  # farthest sector first


def test_redirects_to_a_spotted_enemy_over_continuing_the_search():
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        state.marines.append(fake.marine(tag, x=1, y=1))
    state.enemies.append(fake.enemy_unit(99, fake.UNIT_MARINE, x=24, y=24))  # sector 5, not the far corner
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == spec.move_action_for_sector(5)


def test_moves_to_next_sector_after_arriving_at_current_target_and_finding_nothing():
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    orientation = identity_orientation(spec)
    far_sector = spec.grid.num_sectors - 1

    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        state.marines.append(fake.marine(tag, x=1, y=1))
    first_action = policy.action(state, spec, masking, orientation)
    assert first_action == spec.move_action_for_sector(far_sector)

    # Army has now arrived at the far sector; still nothing there.
    fx, fy = spec.grid.sector_center(far_sector)
    state_arrived = GameState(game_loop=1, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        state_arrived.marines.append(fake.marine(tag, x=fx, y=fy))
    second_action = policy.action(state_arrived, spec, masking, orientation)
    assert second_action != spec.move_action_for_sector(far_sector)  # moved on to search elsewhere


def test_gives_up_on_an_unreachable_target_after_timeout():
    # Regression test: a target sector near unpathable terrain (cliffs,
    # water) can be approached but never technically "arrived" at -- without
    # a timeout, the search would stall on it forever instead of covering
    # the rest of the map.
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    policy = ScriptedPolicy(ScriptedPolicyConfig(
        target_supply_depots=0, target_barracks=0, attack_threshold=20, search_timeout_steps=3,
    ))
    orientation = identity_orientation(spec)
    far_sector = spec.grid.num_sectors - 1

    # Marines never actually reach the target sector (stuck short of it,
    # simulating unreachable terrain) across every step below the timeout.
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        state.marines.append(fake.marine(tag, x=1, y=1))

    for _ in range(6):
        action = policy.action(state, spec, masking, orientation)
    assert action != spec.move_action_for_sector(far_sector)  # gave up and moved on after the timeout


def test_breaks_off_search_to_defend_home_under_threat():
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        state.marines.append(fake.marine(tag, x=1, y=1))  # near canonical home sector 0
    state.enemies.append(fake.enemy_unit(99, fake.UNIT_MARINE, x=2, y=2))  # also in sector 0
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == spec.move_action_for_sector(0)


def test_no_op_when_nothing_legal():
    spec = make_spec()
    masking = MaskingConfig()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    policy = ScriptedPolicy()
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.NO_OP
