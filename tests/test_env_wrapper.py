"""Exercises SC2FightEnv's step()/reset() contract against a stubbed SC2Env
(no live StarCraft II client involved) via the env_factory injection seam."""

import pytest
from pysc2.lib import actions as sc2_actions

from sc2rl.config import EnvConfig
from sc2rl.env.action_space import FixedAction
from sc2rl.env.sc2_env_wrapper import SC2FightEnv
from tests.fakes import fake_pysc2 as fake


class StubSC2Env:
    def __init__(self, timesteps):
        self._timesteps = list(timesteps)
        self._cursor = 0
        self.received_actions = []

    def reset(self):
        self._cursor = 0
        return [self._timesteps[self._cursor]]

    def step(self, actions):
        self.received_actions.append(actions)
        self._cursor = min(self._cursor + 1, len(self._timesteps) - 1)
        return [self._timesteps[self._cursor]]

    def close(self):
        pass


def make_env(
    timesteps, config: EnvConfig | None = None, keep_time_penalty: bool = False
) -> tuple[SC2FightEnv, StubSC2Env]:
    config = config or EnvConfig()
    if not keep_time_penalty:
        # The flat per-step time penalty applies from step one whenever
        # shaping is on; it has its own test, and zeroing it here keeps every
        # other component's test exact instead of off by 0.001 per step.
        config.reward.time_penalty_per_step = 0.0
    stub = StubSC2Env(timesteps)
    env = SC2FightEnv(config, env_factory=lambda cfg: stub)
    return env, stub


def test_reset_reads_playable_area_and_converts_it_to_the_raw_frame():
    # game_info is in world coordinates; raw_units and raw actions are in
    # pysc2's raw_resolution frame: scaled by raw_resolution / max(map
    # dims) and y-flipped. Regression test: fed in untransformed, the clamp
    # landed in the wrong place entirely.
    from types import SimpleNamespace

    class StubWithGameInfo(StubSC2Env):
        game_info = [SimpleNamespace(start_raw=SimpleNamespace(
            map_size=SimpleNamespace(x=88, y=96),
            playable_area=SimpleNamespace(p0=SimpleNamespace(x=10, y=12), p1=SimpleNamespace(x=54, y=52)),
        ))]

    ts0 = fake.make_timestep(minerals=50, food_cap=15)
    stub = StubWithGameInfo([ts0])
    config = EnvConfig()
    config.map_size = 64  # raw_resolution
    env = SC2FightEnv(config, env_factory=lambda cfg: stub)
    env.reset()

    scale = 64 / 96
    min_x, min_y, max_x, max_y = env.grid.bounds
    assert min_x == pytest.approx(10 * scale)
    assert max_x == pytest.approx(54 * scale)
    assert min_y == pytest.approx((96 - 52) * scale)  # y flipped: world p1.y is the raw minimum
    assert max_y == pytest.approx((96 - 12) * scale)
    # The grid, the action spec, the translator and the orientation all see
    # the same playable-area geometry; only the geometry changed, not sizes.
    assert env.action_spec.grid is env.grid
    assert env._translator.spec is env.action_spec
    assert env.orientation.bounds == env.grid.bounds
    assert env.observation_space.shape == env.reset()[0].shape
    assert env.grid.sector_of(min_x + 0.1, min_y + 0.1) == 0
    assert env.grid.sector_of(max_x - 0.1, max_y - 0.1) == env.grid.num_sectors - 1


def test_reset_falls_back_to_full_map_when_game_info_unavailable():
    ts0 = fake.make_timestep(minerals=50, food_cap=15)
    env, _ = make_env([ts0])
    env.reset()
    assert env.grid.bounds is None
    assert env.grid.sector_of(63.9, 63.9) == env.grid.num_sectors - 1


def test_reset_returns_correctly_shaped_observation():
    ts0 = fake.make_timestep(minerals=50, food_cap=15)
    env, _ = make_env([ts0])
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == env.observation_space.dtype
    assert info == {}


def test_only_no_op_legal_with_no_units():
    ts0 = fake.make_timestep(minerals=50, food_cap=15)
    env, _ = make_env([ts0])
    env.reset()
    mask = env.action_masks()
    assert mask[FixedAction.NO_OP]
    assert not mask[FixedAction.BUILD_SUPPLY_DEPOT]
    assert not mask[FixedAction.BUILD_BARRACKS]
    assert not mask[FixedAction.TRAIN_MARINE]
    assert not any(mask[env.action_spec.move_action_for_sector(s)] for s in range(env.action_spec.grid.num_sectors))


def test_illegal_action_is_silently_converted_to_no_op():
    ts0 = fake.make_timestep(minerals=50, food_cap=15)
    ts1 = fake.make_timestep(minerals=50, food_cap=15)
    env, stub = make_env([ts0, ts1])
    env.reset()

    illegal_move = env.action_spec.move_action_for_sector(0)  # no marines exist -> illegal
    env.step(illegal_move)

    sent_calls = stub.received_actions[-1][0]
    assert len(sent_calls) == 1
    assert sent_calls[0].function == sc2_actions.RAW_FUNCTIONS.no_op.id


def test_legal_build_action_is_translated_and_cooldown_applied():
    ts0 = fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=200, food_cap=15, food_used=13)
    ts1 = fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=100, food_cap=15, food_used=13)
    config = EnvConfig()
    config.build_cooldown_steps = 3
    env, stub = make_env([ts0, ts1], config)
    env.reset()

    assert env.action_masks()[FixedAction.BUILD_SUPPLY_DEPOT]
    env.step(FixedAction.BUILD_SUPPLY_DEPOT)

    sent_calls = stub.received_actions[-1][0]
    assert sent_calls[0].function == sc2_actions.RAW_FUNCTIONS.Build_SupplyDepot_pt.id
    # cooldown should now block re-issuing the same build action immediately
    assert not env.action_masks()[FixedAction.BUILD_SUPPLY_DEPOT]


def test_terminal_reward_passed_through():
    ts0 = fake.make_timestep(minerals=0, food_cap=15)
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=1.0, step_type="LAST")
    config = EnvConfig()
    config.reward.terminal_reward_scale = 1.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 1.0
    assert terminated is True
    assert truncated is False


def test_shaping_applies_on_terminal_step_and_captures_collapse():
    # Regression test: shaping used to be skipped entirely on ts.last(), so
    # a loss's final wipeout (base/army destroyed) was never charged against
    # reward already banked from building an economy earlier in the episode
    # -- a losing episode's summed reward could still come out positive.
    ts0 = fake.make_timestep(minerals=0, food_cap=15, total_value_units=400)  # healthy economy
    ts1 = fake.make_timestep(  # base wiped out on the terminal step
        minerals=0, food_cap=15, reward=-1.0, step_type="LAST", total_value_units=0,
    )
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.terminal_reward_scale = 1.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    # terminal -1.0, plus the collapse delta (0 - 400) * coefficient 1.0
    assert reward == -1.0 + (0 - 400)


def test_terminal_reward_scale_multiplies_win_loss_reward():
    ts0 = fake.make_timestep(minerals=0, food_cap=15)
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=1.0, step_type="LAST")
    config = EnvConfig()
    config.reward.terminal_reward_scale = 3.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 3.0


def test_reward_shaping_disabled_by_default_matches_terminal_reward_only():
    ts0 = fake.make_timestep(units=[fake.marine(1, x=1, y=1)], minerals=0, food_cap=15)
    ts1 = fake.make_timestep(
        units=[fake.marine(1, x=1, y=1), fake.marine(2, x=1, y=1)], minerals=0, food_cap=15, reward=0.0,
    )
    env, _ = make_env([ts0, ts1])
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 0.0  # shaping off by default, mid-episode reward stays 0


def test_reward_shaping_rewards_rising_army_value_regardless_of_marine_count():
    ts0 = fake.make_timestep(minerals=0, food_cap=15, total_value_units=50)
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=0.0, total_value_units=100)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 50.0  # army-value delta is never scaled by concentration


def test_economic_value_capped_to_prevent_indefinite_hoarding():
    # Regression test: confirmed live that a losing episode earned +2.85 in
    # economic reward alone, because marines that never engage also never
    # die, so total_value_units can grow without bound. Growth below the cap
    # is still fully rewarded; growth past it earns nothing further.
    ts0 = fake.make_timestep(minerals=0, food_cap=15, total_value_units=90)
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=0.0, total_value_units=150)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.economic_value_cap = 100.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 10.0  # only the 90 -> 100 portion counts, not 90 -> 150


def test_economic_value_growth_earns_nothing_once_already_past_the_cap():
    ts0 = fake.make_timestep(minerals=0, food_cap=15, total_value_units=200)  # already over the cap
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=0.0, total_value_units=300)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.economic_value_cap = 100.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 0.0


def test_kill_value_capped_to_prevent_unbounded_grinding_reward():
    # Regression test: killed_value only ever increases (kills aren't
    # "undone"), so unlike a hard-capped economic value, a long grindy fight
    # against a continuously-spawning bot has no natural ceiling without
    # this. Growth below the cap is still fully rewarded; growth past it
    # earns nothing further.
    marines = [fake.marine(i, x=1, y=1) for i in range(4)]  # concentration_factor = 1.0 at threshold
    ts0 = fake.make_timestep(units=marines, minerals=0, food_cap=15, killed_value_units=90)
    ts1 = fake.make_timestep(units=marines, minerals=0, food_cap=15, reward=0.0, killed_value_units=150)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.kill_value_scale = 1.0
    config.reward.concentration_threshold = 4
    config.reward.kill_value_cap = 100.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 10.0  # only the 90 -> 100 portion counts, not 90 -> 150


def test_reward_shaping_rewards_building_structures_not_just_training_units():
    # Regression test: building a supply depot/barracks previously earned
    # zero immediate shaped reward (only total_value_units was tracked) --
    # a weak, indirect signal for "build infrastructure early" that showed
    # up in practice as the policy learning to delay barracks construction.
    ts0 = fake.make_timestep(minerals=0, food_cap=15, total_value_structures=0)
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=0.0, total_value_structures=100)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 100.0


def test_reward_shaping_penalizes_falling_army_value():
    ts0 = fake.make_timestep(minerals=0, food_cap=15, total_value_units=100)
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=0.0, total_value_units=50)  # a marine died
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == -50.0


def test_reward_shaping_kill_value_gets_partial_credit_below_concentration_threshold():
    marines = [fake.marine(i, x=1, y=1) for i in range(2)]  # 2 marines, below threshold of 4
    ts0 = fake.make_timestep(units=marines, minerals=0, food_cap=15, killed_value_units=0)
    ts1 = fake.make_timestep(units=marines, minerals=0, food_cap=15, reward=0.0, killed_value_units=40)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.kill_value_scale = 1.0  # isolate concentration scaling from the separate kill discount
    config.reward.concentration_threshold = 4
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    # concentration_factor = 2/4 = 0.5, so only half the kill value is credited
    assert reward == 20.0


def test_reward_shaping_kill_value_gets_full_credit_at_concentration_threshold():
    marines = [fake.marine(i, x=1, y=1) for i in range(4)]  # meets the threshold of 4
    ts0 = fake.make_timestep(units=marines, minerals=0, food_cap=15, killed_value_units=0)
    ts1 = fake.make_timestep(units=marines, minerals=0, food_cap=15, reward=0.0, killed_value_units=40)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.kill_value_scale = 1.0  # isolate concentration scaling from the separate kill discount
    config.reward.concentration_threshold = 4
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 40.0


def test_reward_shaping_kill_value_discounted_by_kill_value_scale():
    # Regression test for the reported bug: a losing episode's ep_rew_mean
    # went UP after a loss, because kill-value credit (uncapped, not offset
    # by an eventual defeat the way economic value is) outweighed the
    # terminal penalty. kill_value_scale discounts it further on top of the
    # concentration-of-force scaling.
    marines = [fake.marine(i, x=1, y=1) for i in range(4)]  # at/above threshold -> concentration_factor = 1.0
    ts0 = fake.make_timestep(units=marines, minerals=0, food_cap=15, killed_value_units=0)
    ts1 = fake.make_timestep(units=marines, minerals=0, food_cap=15, reward=0.0, killed_value_units=40)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.kill_value_scale = 0.1
    config.reward.concentration_threshold = 4
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 4.0  # 40 * kill_value_scale(0.1) * concentration_factor(1.0)


def test_home_defense_penalty_applied_when_home_undefended():
    # Home sector (canonical sector 0) has an enemy and no friendly marines.
    # scouting_bonus zeroed to isolate the defense-penalty term -- this same
    # enemy sighting would otherwise also trigger a first-sighting bonus.
    ts0 = fake.make_timestep(minerals=0, food_cap=15)
    ts1 = fake.make_timestep(units=[fake.enemy_unit(1, fake.UNIT_MARINE, x=1, y=1)], minerals=0, food_cap=15)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.home_defense_penalty = 0.05
    config.reward.scouting_bonus = 0.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == -0.05


def test_home_defense_penalty_not_applied_when_defenders_present():
    ts0 = fake.make_timestep(minerals=0, food_cap=15)
    ts1 = fake.make_timestep(
        units=[fake.enemy_unit(1, fake.UNIT_MARINE, x=1, y=1), fake.marine(2, x=1, y=1)],
        minerals=0, food_cap=15,
    )
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.home_defense_penalty = 0.05
    config.reward.scouting_bonus = 0.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 0.0


def test_home_defense_penalty_capped_per_episode():
    # Regression test: episodes have no step limit (only PySC2's own
    # game-end conditions), so this per-step penalty accumulating unbounded
    # across a long episode was the actual root cause of ep_rew_mean
    # reaching -46.4 after only 18,000 timesteps in a single early,
    # untrained episode -- far beyond what the terminal_reward_scale
    # analysis accounted for, since that analysis only ever reasoned about a
    # single-step transition.
    undefended_home = fake.make_timestep(
        units=[fake.enemy_unit(1, fake.UNIT_MARINE, x=1, y=1)], minerals=0, food_cap=15,
    )
    timesteps = [fake.make_timestep(minerals=0, food_cap=15)] + [undefended_home] * 30
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.home_defense_penalty = 0.05
    config.reward.home_defense_penalty_cap = 0.15  # exactly 3 firings of 0.05, no fractional remainder
    config.reward.scouting_bonus = 0.0
    env, _ = make_env(timesteps, config)
    env.reset()

    total = 0.0
    for _ in range(30):
        _, reward, _, _, _ = env.step(FixedAction.NO_OP)
        total += reward
    # Without a cap this would be -1.5 (30 steps * -0.05); capped at -0.15.
    assert total == pytest.approx(-0.15)


def test_home_defense_penalty_covers_every_sector_with_a_friendly_building():
    # Regression test: home used to be hard-coded as sector 0. With ~7-unit
    # cells the base straddles several sectors, so an enemy at the barracks
    # next door (and no marine there) must count as an undefended attack on
    # home even though the command center's sector is quiet.
    base = [fake.command_center(1, x=8, y=8), fake.barracks(2, x=36, y=20, complete=True)]  # sectors 0 and 6
    ts0 = fake.make_timestep(units=base, minerals=0, food_cap=15)
    ts1 = fake.make_timestep(
        units=base + [fake.enemy_unit(9, fake.UNIT_MARINE, x=37, y=21), fake.marine(3, x=8, y=8)],
        minerals=0, food_cap=15,
    )
    config = EnvConfig()
    config.grid.cols = config.grid.rows = 4
    config.reward.shaping_enabled = True
    config.reward.home_defense_penalty = 0.05
    config.reward.scouting_bonus = 0.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    _, reward, _, _, _ = env.step(FixedAction.NO_OP)
    assert reward == -0.05  # marine at the CC doesn't defend the barracks' sector


def test_base_sectors_start_explored_and_recall_mask_follows_the_command_center():
    from sc2rl.env.observation import FEATURES_PER_SECTOR, observation_length

    base = [fake.command_center(1, x=30, y=30)]  # sector 5 on 4x4/64
    ts0 = fake.make_timestep(units=base + [fake.marine(i, x=31, y=31) for i in range(4)], minerals=0, food_cap=15)
    config = EnvConfig()
    config.grid.cols = config.grid.rows = 4
    config.masking.min_marines_to_move = 4
    config.masking.min_marines_to_advance = 20
    env, _ = make_env([ts0], config)
    obs, _ = env.reset()

    global_len = observation_length(env.grid) - FEATURES_PER_SECTOR * env.grid.num_sectors
    assert obs[global_len + FEATURES_PER_SECTOR * 5 + 5] == 1.0  # CC's sector starts explored
    assert obs[global_len + FEATURES_PER_SECTOR * 0 + 5] == 0.0  # sector 0 is nothing special now
    mask = env.action_masks()
    assert mask[env.action_spec.move_action_for_sector(5)]
    assert not mask[env.action_spec.move_action_for_sector(0)]


def test_time_penalty_applies_every_step_and_is_capped():
    ts = fake.make_timestep(minerals=0, food_cap=15)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.time_penalty_per_step = 0.5
    config.reward.time_penalty_cap = 1.2
    env, _ = make_env([ts] * 5, config, keep_time_penalty=True)
    env.reset()
    rewards = [env.step(FixedAction.NO_OP)[1] for _ in range(4)]
    assert rewards == pytest.approx([-0.5, -0.5, -0.2, 0.0])


def _approach_config() -> EnvConfig:
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.scouting_bonus = 0.0
    config.reward.exploration_bonus = 0.0
    config.reward.approach_reward_scale = 2.0
    config.reward.approach_reward_cap = 3.0
    return config


def test_approach_reward_pays_for_closing_distance_to_a_known_enemy_structure():
    hatchery = fake.enemy_unit(9, fake.UNIT_HATCHERY, x=60, y=60)
    ts0 = fake.make_timestep(units=[fake.marine(1, x=10, y=10)], minerals=0, food_cap=15)
    ts1 = fake.make_timestep(units=[fake.marine(1, x=10, y=10), hatchery], minerals=0, food_cap=15)
    ts2 = fake.make_timestep(units=[fake.marine(1, x=20, y=20), hatchery], minerals=0, food_cap=15)
    ts3 = fake.make_timestep(units=[fake.marine(1, x=15, y=15), hatchery], minerals=0, food_cap=15)
    env, _ = make_env([ts0, ts1, ts2, ts3], _approach_config())
    env.reset()

    _, r1, _, _, _ = env.step(FixedAction.NO_OP)  # structure just discovered: baseline only
    assert r1 == 0.0
    _, r2, _, _, info2 = env.step(FixedAction.NO_OP)  # closed in
    diagonal = 64 * 2 ** 0.5
    expected = 2.0 * ((50 * 2 ** 0.5) - (40 * 2 ** 0.5)) / diagonal
    assert info2["reward_approach"] == pytest.approx(expected)
    assert r2 == pytest.approx(expected)
    _, r3, _, _, _ = env.step(FixedAction.NO_OP)  # backed off: symmetric charge
    assert r3 == pytest.approx(-expected / 2)


def test_approach_reward_is_silent_on_the_step_the_known_structure_set_changes():
    # Killing the last building of a base makes the "nearest structure"
    # jump to a farther one -- that must not read as backing away.
    near = fake.enemy_unit(9, fake.UNIT_HATCHERY, x=30, y=30)
    far = fake.enemy_unit(10, fake.UNIT_HATCHERY, x=60, y=60)
    ts0 = fake.make_timestep(units=[fake.marine(1, x=28, y=28)], minerals=0, food_cap=15)
    ts1 = fake.make_timestep(units=[fake.marine(1, x=28, y=28), near, far], minerals=0, food_cap=15)
    ts2 = fake.make_timestep(units=[fake.marine(1, x=29, y=29), near, far], minerals=0, food_cap=15)
    ts3 = fake.make_timestep(units=[fake.marine(1, x=30, y=30), far], minerals=0, food_cap=15)  # near killed
    env, _ = make_env([ts0, ts1, ts2, ts3], _approach_config())
    env.reset()
    env.step(FixedAction.NO_OP)
    _, r2, _, _, _ = env.step(FixedAction.NO_OP)
    assert r2 > 0.0
    _, r3, _, _, info3 = env.step(FixedAction.NO_OP)
    assert info3["reward_approach"] == 0.0


def test_unreachable_sectors_are_masked_pre_explored_and_targets_are_pathable():
    import numpy as np

    from sc2rl.env.observation import FEATURES_PER_SECTOR, observation_length

    # 4x4 grid over a 64 map: 16-unit cells. Sector 15 (x,y in 48..64) is a
    # pathable plateau with no connection to the rest of the map (a wall of
    # unpathable cells around it) -- pathable, but not reachable, so it must
    # count as unreachable. Sector 0 is pathable only in a strip x < 4.
    pathable = np.ones((64, 64), dtype=bool)
    pathable[47, 47:64] = False
    pathable[47:64, 47] = False
    pathable[0:16, 4:16] = False
    marines = [fake.marine(i, x=30, y=30) for i in range(25)]
    base = [fake.command_center(99, x=30, y=30)]  # near side of both axes: identity orientation
    ts0 = fake.make_timestep(units=marines + base, minerals=0, food_cap=15, pathable=pathable)
    config = EnvConfig()
    config.grid.cols = config.grid.rows = 4
    config.masking.min_marines_to_advance = 20
    env, _ = make_env([ts0, ts0], config)
    obs, _ = env.reset()

    assert env.unreachable_sectors == frozenset({15})
    mask = env.action_masks()
    assert not mask[env.action_spec.move_action_for_sector(15)]
    assert mask[env.action_spec.move_action_for_sector(14)]
    global_len = observation_length(env.grid) - FEATURES_PER_SECTOR * env.grid.num_sectors
    assert obs[global_len + FEATURES_PER_SECTOR * 15 + 5] == 1.0  # nothing there to explore
    assert obs[global_len + FEATURES_PER_SECTOR * 14 + 5] == 0.0

    targets = env._translator.sector_targets
    assert targets[15] is None
    assert targets[0] == (3.5, 7.5)  # nearest pathable cell to the (8, 8) center, inside the strip
    # Fully pathable sector: a cell within one unit of its own (24, 24) center
    # (which sits on a cell corner, so several cells tie at equal distance).
    assert abs(targets[5][0] - 24) <= 1.0 and abs(targets[5][1] - 24) <= 1.0


def test_pathing_debug_map_is_written_once_when_configured(tmp_path):
    import numpy as np

    pathable = np.ones((64, 64), dtype=bool)
    pathable[47, 47:64] = False
    pathable[47:64, 47] = False
    base = [fake.command_center(99, x=30, y=30)]
    ts0 = fake.make_timestep(units=base + [fake.marine(1, x=31, y=31)], minerals=0, food_cap=15, pathable=pathable)
    config = EnvConfig()
    config.grid.cols = config.grid.rows = 4
    config.pathing_debug_path = str(tmp_path / "pathing_debug.txt")
    env, _ = make_env([ts0, ts0], config)
    env.reset()

    text = (tmp_path / "pathing_debug.txt").read_text(encoding="utf-8")
    assert "unreachable sectors = [15]" in text
    assert "command center raw = (30.0, 30.0)" in text
    import re

    rows = [m.group(1) for m in (re.match(r"^\s*\d+ ([#~.CmTE]{64})$", line) for line in text.splitlines()) if m]
    assert len(rows) == 64
    assert rows[30][30] == "C"
    assert rows[31][31] == "m"
    assert rows[50][50] == "~"  # pathable plateau, cut off from the base
    assert rows[47][50] == "#"  # the wall
    assert rows[5][5] == "."


def test_scouting_bonus_awarded_once_per_newly_seen_enemy_sector():
    ts0 = fake.make_timestep(minerals=0, food_cap=15)  # no enemies visible yet
    ts1 = fake.make_timestep(units=[fake.enemy_unit(1, fake.UNIT_MARINE, x=56, y=56)], minerals=0, food_cap=15)
    ts2 = fake.make_timestep(units=[fake.enemy_unit(1, fake.UNIT_MARINE, x=56, y=56)], minerals=0, food_cap=15)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.scouting_bonus = 0.02
    env, _ = make_env([ts0, ts1, ts2], config)
    env.reset()

    _, first_reward, _, _, _ = env.step(FixedAction.NO_OP)
    assert first_reward == 0.02  # first sighting of that sector

    _, second_reward, _, _, _ = env.step(FixedAction.NO_OP)
    assert second_reward == 0.0  # same sector already seen this episode


def test_observation_remembers_explored_sectors_across_steps_even_with_shaping_off():
    # The `explored` per-sector feature is episode memory the env maintains:
    # once a marine has been in a sector it stays flagged after the marine
    # leaves, and the memory must update whether or not reward shaping is on
    # (it used to live inside the exploration-bonus reward code).
    from sc2rl.env.observation import FEATURES_PER_SECTOR, observation_length

    ts0 = fake.make_timestep(units=[fake.marine(1, x=1, y=1)], minerals=0, food_cap=15)
    ts1 = fake.make_timestep(units=[fake.marine(1, x=56, y=56)], minerals=0, food_cap=15)
    ts2 = fake.make_timestep(units=[fake.marine(1, x=1, y=1)], minerals=0, food_cap=15)  # back home
    config = EnvConfig()
    config.reward.shaping_enabled = False
    env, _ = make_env([ts0, ts1, ts2], config)
    obs0, _ = env.reset()

    def explored_flag(obs, sector):
        global_len = observation_length(env.grid) - FEATURES_PER_SECTOR * env.grid.num_sectors
        return obs[global_len + FEATURES_PER_SECTOR * sector + 5]

    far = env.grid.num_sectors - 1
    assert explored_flag(obs0, 0) == 1.0  # home is explored from the start
    assert explored_flag(obs0, far) == 0.0

    obs1, _, _, _, _ = env.step(FixedAction.NO_OP)
    assert explored_flag(obs1, far) == 1.0

    obs2, _, _, _, _ = env.step(FixedAction.NO_OP)
    assert explored_flag(obs2, far) == 1.0  # still remembered after leaving
    assert explored_flag(obs2, 0) == 1.0

    obs_after_reset, _ = env.reset()
    assert explored_flag(obs_after_reset, far) == 0.0  # memory is per-episode


def test_exploration_bonus_awarded_once_per_newly_visited_sector():
    ts0 = fake.make_timestep(units=[fake.marine(1, x=1, y=1)], minerals=0, food_cap=15)  # home sector
    ts1 = fake.make_timestep(units=[fake.marine(1, x=56, y=56)], minerals=0, food_cap=15)  # far sector
    ts2 = fake.make_timestep(units=[fake.marine(1, x=56, y=56)], minerals=0, food_cap=15)  # same sector again
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.exploration_bonus = 0.02
    env, _ = make_env([ts0, ts1, ts2], config)
    env.reset()

    _, first_reward, _, _, _ = env.step(FixedAction.NO_OP)
    assert first_reward == 0.02  # first time a marine enters this sector

    _, second_reward, _, _, _ = env.step(FixedAction.NO_OP)
    assert second_reward == 0.0  # same sector already visited this episode


def test_exploration_bonus_not_awarded_for_home_sector_at_spawn():
    # Regression guard: marines start in the home sector, so it must count
    # as already "visited" -- otherwise every episode would pay a free
    # exploration bonus for simply existing at spawn.
    ts0 = fake.make_timestep(units=[fake.marine(1, x=1, y=1)], minerals=0, food_cap=15)
    ts1 = fake.make_timestep(units=[fake.marine(1, x=1, y=1)], minerals=0, food_cap=15)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.exploration_bonus = 0.02
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    _, reward, _, _, _ = env.step(FixedAction.NO_OP)
    assert reward == 0.0


def test_stale_search_penalty_applies_once_patience_exceeded_without_new_sector():
    # Regression test: once the army has visited what it's going to visit
    # for a while, exploration_bonus/scouting_bonus (both one-time per
    # sector) stop pulling it onward -- observed live as a large army
    # parking in one sector indefinitely ("a huge pile of marines in one
    # location"). This penalty is the counterweight: it should stay silent
    # while patience hasn't been exceeded, then kick in.
    marines = [fake.marine(1, x=1, y=1)]  # stays in the home sector every step
    timesteps = [fake.make_timestep(units=marines, minerals=0, food_cap=15) for _ in range(4)]
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.stale_search_penalty = 0.01
    config.reward.stale_search_patience = 2
    config.masking.min_marines_to_move = 1
    config.masking.min_marines_to_advance = 1
    env, _ = make_env(timesteps, config)
    env.reset()

    # Home sector was already "visited" at spawn, so this marine never
    # triggers exploration_bonus -- every step here is "no progress."
    _, r1, _, _, _ = env.step(FixedAction.NO_OP)  # steps_since_new_sector -> 1
    assert r1 == 0.0
    _, r2, _, _, _ = env.step(FixedAction.NO_OP)  # steps_since_new_sector -> 2, still within patience
    assert r2 == 0.0
    _, r3, _, _, _ = env.step(FixedAction.NO_OP)  # steps_since_new_sector -> 3, patience exceeded
    assert r3 == -0.01


def test_stale_search_penalty_silent_before_army_can_legally_leave_home():
    # Standing at home while the army is still too small to advance (below
    # min_marines_to_advance) must never be penalized -- that is correct,
    # required behavior, not stalling. Regression test: this used to gate on
    # the lower min_marines_to_move, so marines 4..19 were a penalty stream
    # the policy could only stop by never building the 4th marine -- and it
    # learned exactly that.
    marines = [fake.marine(i, x=1, y=1) for i in range(10)]  # >= move floor, < advance floor
    ts = fake.make_timestep(units=marines, minerals=0, food_cap=15)
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.stale_search_penalty = 0.01
    config.reward.stale_search_patience = 0
    config.masking.min_marines_to_move = 4
    config.masking.min_marines_to_advance = 20
    env, _ = make_env([ts, ts, ts], config)
    env.reset()
    _, r1, _, _, _ = env.step(FixedAction.NO_OP)
    _, r2, _, _, _ = env.step(FixedAction.NO_OP)
    assert r1 == 0.0
    assert r2 == 0.0


def test_stale_search_penalty_resets_on_reaching_a_new_sector():
    marines_home = [fake.marine(1, x=1, y=1)]
    marines_far = [fake.marine(1, x=56, y=56)]
    ts0 = fake.make_timestep(units=marines_home, minerals=0, food_cap=15)
    ts1 = fake.make_timestep(units=marines_far, minerals=0, food_cap=15)  # newly visited sector
    ts2 = fake.make_timestep(units=marines_far, minerals=0, food_cap=15)  # same sector, no progress
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.stale_search_penalty = 0.01
    config.reward.stale_search_patience = 1
    config.reward.exploration_bonus = 0.0  # isolate the stale-penalty counter from its own bonus
    config.masking.min_marines_to_move = 1
    config.masking.min_marines_to_advance = 1
    env, _ = make_env([ts0, ts1, ts2], config)
    env.reset()

    _, r1, _, _, _ = env.step(FixedAction.NO_OP)  # entered a new sector -> counter reset to 0
    assert r1 == 0.0
    _, r2, _, _, _ = env.step(FixedAction.NO_OP)  # counter -> 1, still within patience
    assert r2 == 0.0


def test_stale_search_penalty_capped_per_episode():
    # Same unbounded-episode-length problem as home_defense_penalty -- see
    # test_home_defense_penalty_capped_per_episode.
    marines = [fake.marine(1, x=1, y=1)]  # never leaves the home sector
    timesteps = [fake.make_timestep(units=marines, minerals=0, food_cap=15) for _ in range(30)]
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.stale_search_penalty = 0.01
    config.reward.stale_search_patience = 0
    config.reward.stale_search_penalty_cap = 0.03  # exactly 3 firings of 0.01
    config.masking.min_marines_to_move = 1
    config.masking.min_marines_to_advance = 1
    env, _ = make_env(timesteps, config)
    env.reset()

    total = 0.0
    for _ in range(10):
        _, reward, _, _, _ = env.step(FixedAction.NO_OP)
        total += reward
    # Without a cap this would be -0.10 (10 steps * -0.01); capped at -0.03.
    assert total == pytest.approx(-0.03)


def test_step_info_exposes_per_component_reward_breakdown():
    # So an imbalance between components (e.g. kill-value outweighing a
    # loss) is directly inspectable instead of needing to be reasoned about
    # from formulas after the fact.
    ts0 = fake.make_timestep(minerals=0, food_cap=15, total_value_units=50)
    ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=1.0, total_value_units=100, step_type="LAST")
    config = EnvConfig()
    config.reward.shaping_enabled = True
    config.reward.shaping_coefficient = 1.0
    config.reward.terminal_reward_scale = 1.0
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert info["reward_terminal"] == 1.0
    assert info["reward_economic"] == 50.0
    assert set(info.keys()) == {
        "reward_terminal", "reward_economic", "reward_kill", "reward_home_defense",
        "reward_scouting", "reward_exploration", "reward_stale_search", "reward_time", "reward_approach",
    }
    assert reward == sum(info.values())


def test_default_config_guarantees_any_win_outscores_a_heavily_shaped_loss():
    # The core invariant terminal_reward_scale exists to protect: winning
    # must always beat losing, no matter how much shaping reward a losing
    # episode racks up. Uses the actual default config (no overrides) so
    # this catches drift if the caps/scale are ever retuned out of balance
    # again -- exactly the class of bug that recurred multiple times before
    # kill_value_cap and a dominant terminal_reward_scale were added.
    config = EnvConfig()
    config.reward.shaping_enabled = True

    # Losing episode that racks up close to the maximum plausible shaping:
    # economic and kill value both driven to their caps, plus some scouting.
    enemies = [fake.enemy_unit(100 + i, fake.UNIT_MARINE, x=float(i * 10), y=1.0) for i in range(3)]
    marines = [fake.marine(i, x=1, y=1) for i in range(20)]  # meets concentration_threshold -> factor = 1.0
    loss_ts0 = fake.make_timestep(units=marines, minerals=0, food_cap=15)
    loss_ts1 = fake.make_timestep(
        units=marines + enemies, minerals=0, food_cap=15, reward=-1.0, step_type="LAST",
        total_value_units=config.reward.economic_value_cap,
        killed_value_units=config.reward.kill_value_cap,
    )
    loss_env, _ = make_env([loss_ts0, loss_ts1], config)
    loss_env.reset()
    _, loss_reward, _, _, _ = loss_env.step(FixedAction.NO_OP)

    # Winning episode with zero shaping at all.
    win_ts0 = fake.make_timestep(minerals=0, food_cap=15)
    win_ts1 = fake.make_timestep(minerals=0, food_cap=15, reward=1.0, step_type="LAST")
    win_env, _ = make_env([win_ts0, win_ts1], config)
    win_env.reset()
    _, win_reward, _, _, _ = win_env.step(FixedAction.NO_OP)

    assert win_reward > loss_reward


def test_win_reward_stays_positive_despite_a_long_troubled_episode():
    # Regression test for the actual reported bug: episodes have no step
    # limit, so home_defense_penalty and stale_search_penalty (both flat
    # per-step terms) could accumulate for however long an early, untrained
    # episode dragged on before they were capped -- confirmed live as
    # ep_rew_mean reaching -46.4 after only 18,000 timesteps in a single
    # episode. The earlier win/loss invariant test above only ever exercised
    # a single step() call, so it could never have caught this. Uses the
    # actual default config (no overrides) and a long run of steps that are
    # each individually adverse (home undefended, army stalled in the same
    # sector, economy having just collapsed) to confirm a win's total
    # reward stays positive regardless of episode length.
    config = EnvConfig()
    config.reward.shaping_enabled = True

    marines = [fake.marine(i, x=56, y=56) for i in range(20)]  # mobilized, parked away from home
    enemy_at_home = [fake.enemy_unit(1, fake.UNIT_MARINE, x=1, y=1)]
    reset_step = fake.make_timestep(
        units=marines, minerals=0, food_cap=15, total_value_units=config.reward.economic_value_cap,
    )
    bad_step = fake.make_timestep(units=marines + enemy_at_home, minerals=0, food_cap=15, total_value_units=0)
    win_step = fake.make_timestep(
        units=marines, minerals=0, food_cap=15, reward=1.0, step_type="LAST", total_value_units=0,
    )
    timesteps = [reset_step] + [bad_step] * 100 + [win_step]
    env, _ = make_env(timesteps, config, keep_time_penalty=True)
    env.reset()

    total = 0.0
    terminated = False
    for _ in range(101):
        _, reward, terminated, _, _ = env.step(FixedAction.NO_OP)
        total += reward
    assert terminated
    assert total > 0.0
