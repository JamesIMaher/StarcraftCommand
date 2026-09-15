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


def test_move_actions_need_marines():
    spec = make_spec()
    config = MaskingConfig()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = compute_action_masks(state, spec, config)
    assert not any(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))

    state.marines.append(fake.marine(1))
    mask = compute_action_masks(state, spec, config)
    assert all(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))


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
