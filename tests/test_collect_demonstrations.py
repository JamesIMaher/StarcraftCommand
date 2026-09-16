"""Validates the demonstration-collection loop against a stubbed SC2Env (no
live StarCraft II client involved), confirming it produces correctly-shaped,
correctly-typed data before ever running it against a real game.
"""

from sc2rl.config import Config
from sc2rl.training.collect_demonstrations import collect
from tests.fakes import fake_pysc2 as fake
from tests.test_env_wrapper import StubSC2Env


def test_collect_produces_matching_length_arrays_with_correct_dtypes():
    timesteps = [
        fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=150, food_cap=15),
        fake.make_timestep(units=[fake.scv(1, x=5, y=5), fake.marine(2, x=10, y=10)],
                            minerals=100, food_cap=15),
        fake.make_timestep(units=[fake.marine(2, x=10, y=10)], minerals=50, food_cap=15,
                            reward=1.0, step_type="LAST"),
    ]

    def env_factory(cfg):
        return StubSC2Env(timesteps)

    config = Config()
    observations, actions, masks, episode_ids = collect(config, episodes=2, env_factory=env_factory)

    assert len(observations) == len(actions) == len(masks) == len(episode_ids) > 0
    assert observations.dtype.kind == "f"
    assert actions.dtype.kind == "i"
    assert masks.dtype == bool
    assert episode_ids.dtype.kind == "i"
    assert observations.shape[1] > 0
    assert masks.shape[1] > 0
    assert set(episode_ids) == {0, 1}  # exactly the 2 requested episodes
    assert list(episode_ids) == sorted(episode_ids)  # non-decreasing: collection order


def test_every_recorded_action_is_legal_under_its_own_recorded_mask():
    # Regression test: env.action_masks() additionally applies build
    # cooldowns on top of the scripted policy's own raw legality check, so
    # they can disagree right after a build order fires. A stub with an SCV
    # and plenty of minerals every step (but never actually showing a built
    # supply depot, since the stub replays a fixed observation sequence
    # regardless of the action taken) makes the scripted policy want
    # build_supply_depot on every single step -- exactly the scenario where
    # the mismatch showed up live (blew up BC's loss to ~1.25 million on
    # the real dataset, since a masked-illegal action's log_prob is
    # astronomically negative).
    always_can_build = fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=150, food_cap=15)
    timesteps = [always_can_build] * 10 + [
        fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=150, food_cap=15,
                            reward=1.0, step_type="LAST"),
    ]

    def env_factory(cfg):
        return StubSC2Env(timesteps)

    config = Config()
    observations, actions, masks, episode_ids = collect(config, episodes=1, env_factory=env_factory)

    for i in range(len(actions)):
        assert masks[i][actions[i]], f"step {i}: action {actions[i]} illegal under its own recorded mask"
