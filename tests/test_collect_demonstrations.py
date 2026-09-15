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
    observations, actions, masks = collect(config, episodes=2, env_factory=env_factory)

    assert len(observations) == len(actions) == len(masks) > 0
    assert observations.dtype.kind == "f"
    assert actions.dtype.kind == "i"
    assert masks.dtype == bool
    assert observations.shape[1] > 0
    assert masks.shape[1] > 0
