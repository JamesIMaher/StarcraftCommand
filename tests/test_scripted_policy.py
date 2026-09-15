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


def test_does_not_interrupt_marines_that_are_still_busy():
    # Regression test: reissuing a move/attack order while marines are still
    # mid-approach or mid-fight was part of why they'd scatter instead of
    # staying grouped. While the majority are non-idle (busy), the teacher
    # should issue no_op -- which does not interrupt existing orders --
    # instead of retargeting.
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

    # Still traveling / fighting -- majority not idle.
    busy_state = GameState(game_loop=1, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        busy_state.marines.append(fake.marine(tag, x=10, y=10, idle=False))
    second_action = policy.action(busy_state, spec, masking, orientation)
    assert second_action == FixedAction.NO_OP


def test_moves_to_next_sector_once_marines_go_idle_with_nothing_to_fight():
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

    # Cleared the area and stopped -- idle, nothing left to fight.
    fx, fy = spec.grid.sector_center(far_sector)
    idle_state = GameState(game_loop=1, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        idle_state.marines.append(fake.marine(tag, x=fx, y=fy, idle=True))
    second_action = policy.action(idle_state, spec, masking, orientation)
    assert second_action != spec.move_action_for_sector(far_sector)  # moved on to search elsewhere


def test_gives_up_via_timeout_if_never_confirmed_idle():
    # Safety net: if marines somehow never go idle (e.g. perpetually
    # re-engaging kited stragglers one at a time), the search still moves on
    # to a DIFFERENT sector eventually instead of stalling on the current
    # target forever.
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    policy = ScriptedPolicy(ScriptedPolicyConfig(
        target_supply_depots=0, target_barracks=0, attack_threshold=20, search_timeout_steps=3,
    ))
    orientation = identity_orientation(spec)
    far_sector = spec.grid.num_sectors - 1

    idle_state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        idle_state.marines.append(fake.marine(tag, x=1, y=1, idle=True))
    first_action = policy.action(idle_state, spec, masking, orientation)
    assert first_action == spec.move_action_for_sector(far_sector)

    busy_state = GameState(game_loop=1, minerals=0, food_used=0, food_cap=15)
    for tag in range(20):
        busy_state.marines.append(fake.marine(tag, x=1, y=1, idle=False))

    for _ in range(6):
        action = policy.action(busy_state, spec, masking, orientation)
    assert action != spec.move_action_for_sector(far_sector)  # gave up on this target after the timeout


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


def test_defends_the_base_when_the_attack_is_on_a_building_outside_the_command_centers_sector():
    # "Home" follows the actual buildings, not a hard-coded sector 0: with
    # ~7-unit cells the base straddles several sectors, and an enemy at the
    # barracks next door must register as an attack on home. Regression
    # test for the teacher never coming back to defend (and losing most
    # games) after the grid was laid over the playable area.
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.command_centers.append(fake.command_center(1, x=20, y=20))  # sector 5 on 4x4/64
    state.barracks.append(fake.barracks(2, x=36, y=20, complete=True))  # sector 6
    for tag in range(10, 30):
        state.marines.append(fake.marine(tag, x=56, y=56))  # army away in the far corner
    state.enemies.append(fake.enemy_unit(99, fake.UNIT_MARINE, x=37, y=21))  # at the barracks
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == spec.move_action_for_sector(5)  # recall to the command center's sector


def test_regroups_at_home_when_below_attack_threshold_out_in_the_field():
    # Below attack_threshold the teacher holds -- but holding in the middle
    # of the map after taking losses meant sitting there getting picked off
    # while reinforcements piled up at home (observed live). It should fall
    # back to the base instead, and only re-issue that order once idle.
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4, min_marines_to_advance=20)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.command_centers.append(fake.command_center(1, x=8, y=8))  # sector 0
    for tag in range(10):
        state.marines.append(fake.marine(tag, x=32, y=32, idle=True))  # 10 marines mid-map
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    orientation = identity_orientation(spec)

    assert policy.action(state, spec, masking, orientation) == spec.move_action_for_sector(0)

    heading_home = GameState(game_loop=1, minerals=0, food_used=0, food_cap=15)
    heading_home.command_centers.append(fake.command_center(1, x=8, y=8))
    for tag in range(10):
        heading_home.marines.append(fake.marine(tag, x=28, y=28, idle=False))
    assert policy.action(heading_home, spec, masking, orientation) == FixedAction.NO_OP  # don't re-issue


def test_holds_at_home_when_below_attack_threshold_and_already_there():
    spec = make_spec()
    masking = MaskingConfig(min_marines_to_move=4, min_marines_to_advance=20)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    state.command_centers.append(fake.command_center(1, x=8, y=8))
    for tag in range(10):
        state.marines.append(fake.marine(tag, x=9, y=9))
    policy = ScriptedPolicy(ScriptedPolicyConfig(target_supply_depots=0, target_barracks=0, attack_threshold=20))
    assert policy.action(state, spec, masking, identity_orientation(spec)) == FixedAction.NO_OP


def test_no_op_when_nothing_legal():
    spec = make_spec()
    masking = MaskingConfig()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    policy = ScriptedPolicy()
    action = policy.action(state, spec, masking, identity_orientation(spec))
    assert action == FixedAction.NO_OP
