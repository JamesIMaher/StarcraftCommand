"""Exercises SC2FightEnv's step()/reset() contract against a stubbed SC2Env
(no live StarCraft II client involved) via the env_factory injection seam."""

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


def make_env(timesteps, config: EnvConfig | None = None) -> tuple[SC2FightEnv, StubSC2Env]:
    stub = StubSC2Env(timesteps)
    env = SC2FightEnv(config or EnvConfig(), env_factory=lambda cfg: stub)
    return env, stub


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
    env, _ = make_env([ts0, ts1])
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 1.0
    assert terminated is True
    assert truncated is False


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
    config.reward.concentration_threshold = 4
    env, _ = make_env([ts0, ts1], config)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(FixedAction.NO_OP)
    assert reward == 40.0


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
