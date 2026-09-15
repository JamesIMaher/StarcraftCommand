from sc2rl.env.action_masking import MaskingConfig, compute_action_masks
from sc2rl.env.action_space import ActionSpaceSpec, FixedAction
from sc2rl.env.game_state import GameState
from sc2rl.env.sector_grid import SectorGrid
from tests.fakes import fake_pysc2 as fake


def make_spec():
    return ActionSpaceSpec(grid=SectorGrid(map_size=64, cols=4, rows=4))


def test_no_op_always_legal():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = compute_action_masks(state, spec, MaskingConfig())
    assert mask[FixedAction.NO_OP]


def test_build_supply_depot_needs_minerals_and_an_scv():
    spec = make_spec()
    config = MaskingConfig()

    # Not enough minerals.
    state = GameState(game_loop=0, minerals=50, food_used=5, food_cap=15)
    state.scvs.append(fake.scv(1))
    assert not compute_action_masks(state, spec, config)[FixedAction.BUILD_SUPPLY_DEPOT]

    # Enough minerals and an SCV, regardless of how much headroom is left --
    # timing is left to the policy to learn, not hard-masked away (a
    # threshold-based gate here previously deadlocked the whole economy: at
    # game start headroom starts above any reasonable "still fine" threshold
    # and nothing else can lower it without a depot existing first).
    state = GameState(game_loop=0, minerals=200, food_used=5, food_cap=15)
    state.scvs.append(fake.scv(1))
    assert compute_action_masks(state, spec, config)[FixedAction.BUILD_SUPPLY_DEPOT]

    # No SCV available -- can't build even if affordable.
    state = GameState(game_loop=0, minerals=200, food_used=13, food_cap=15)
    assert not compute_action_masks(state, spec, config)[FixedAction.BUILD_SUPPLY_DEPOT]


def test_build_barracks_needs_supply_depot():
    spec = make_spec()
    config = MaskingConfig()
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.scvs.append(fake.scv(1))
    assert not compute_action_masks(state, spec, config)[FixedAction.BUILD_BARRACKS]

    state.supply_depots.append(fake.supply_depot(2))
    assert compute_action_masks(state, spec, config)[FixedAction.BUILD_BARRACKS]


def test_train_marine_needs_completed_barracks_not_just_any_barracks():
    spec = make_spec()
    config = MaskingConfig()
    state = GameState(game_loop=0, minerals=300, food_used=0, food_cap=15)
    state.barracks.append(fake.barracks(1, complete=False))
    assert not compute_action_masks(state, spec, config)[FixedAction.TRAIN_MARINE]

    state.barracks.append(fake.barracks(2, complete=True))
    assert compute_action_masks(state, spec, config)[FixedAction.TRAIN_MARINE]


def test_train_marine_blocked_at_supply_cap():
    spec = make_spec()
    config = MaskingConfig()
    state = GameState(game_loop=0, minerals=300, food_used=15, food_cap=15)
    state.barracks.append(fake.barracks(1, complete=True))
    assert not compute_action_masks(state, spec, config)[FixedAction.TRAIN_MARINE]


def test_move_actions_need_minimum_marine_count():
    # Concentration of Force: movement/attack is illegal below the
    # configured minimum marine count, not just "any marines at all" --
    # newly trained marines default to staying clustered at home until mass
    # is reached, instead of being sent out piecemeal. min_marines_to_advance
    # pinned equal to min_marines_to_move here to isolate this general floor
    # from the separate, higher home-vs-elsewhere split (see
    # test_advancing_to_non_home_sectors_needs_higher_marine_count).
    spec = make_spec()
    config = MaskingConfig(min_marines_to_move=4, min_marines_to_advance=4)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = compute_action_masks(state, spec, config)
    assert not any(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))

    for tag in range(1, 4):
        state.marines.append(fake.marine(tag))
    mask = compute_action_masks(state, spec, config)
    assert not any(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))

    state.marines.append(fake.marine(4))
    mask = compute_action_masks(state, spec, config)
    assert all(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))


def test_advancing_to_non_home_sectors_needs_higher_marine_count():
    # Regression test: nothing previously stopped the RL policy from
    # committing the whole army into unexplored territory with only
    # min_marines_to_move marines -- confirmed live as marines exploring too
    # early with too few marines and dying immediately. Moving to the HOME
    # sector (e.g. recalling a scattered force, or defending) stays legal at
    # the lower min_marines_to_move bar; every other sector needs the
    # higher min_marines_to_advance bar.
    spec = make_spec()
    config = MaskingConfig(min_marines_to_move=4, min_marines_to_advance=20)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(4):
        state.marines.append(fake.marine(tag))  # 4 marines: >= move floor, < advance floor

    mask = compute_action_masks(state, spec, config)
    home_action = spec.move_action_for_sector(0)
    other_actions = [spec.move_action_for_sector(s) for s in range(1, spec.grid.num_sectors)]
    assert mask[home_action]
    assert not any(mask[a] for a in other_actions)

    for tag in range(4, 20):
        state.marines.append(fake.marine(tag))  # now 20: meets the advance floor too
    mask = compute_action_masks(state, spec, config)
    assert mask[home_action]
    assert all(mask[a] for a in other_actions)


def test_home_sector_for_the_low_move_threshold_follows_the_command_center():
    # The one sector that only needs min_marines_to_move is wherever the
    # command center actually is, not a hard-coded sector 0.
    spec = make_spec()
    config = MaskingConfig(min_marines_to_move=4, min_marines_to_advance=20)
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for tag in range(4):
        state.marines.append(fake.marine(tag))
    mask = compute_action_masks(state, spec, config, home_sector=5)
    assert mask[spec.move_action_for_sector(5)]
    assert not mask[spec.move_action_for_sector(0)]


def test_barracks_and_supply_depot_caps_respected():
    spec = make_spec()
    config = MaskingConfig(max_barracks=1, max_supply_depots=1)
    state = GameState(game_loop=0, minerals=1000, food_used=13, food_cap=15)
    state.scvs.append(fake.scv(1))
    state.supply_depots.append(fake.supply_depot(2))
    state.barracks.append(fake.barracks(3, complete=True))

    mask = compute_action_masks(state, spec, config)
    assert not mask[FixedAction.BUILD_SUPPLY_DEPOT]
    assert not mask[FixedAction.BUILD_BARRACKS]
