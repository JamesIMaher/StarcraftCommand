"""Validates --resume-from actually works: save a checkpoint, load it back
with MaskablePPO.load(), and confirm training continues (num_timesteps keeps
advancing rather than resetting) instead of just asserting this by reading
the code.
"""

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sc2rl.config import EnvConfig
from sc2rl.env.sc2_env_wrapper import SC2FightEnv
from tests.fakes import fake_pysc2 as fake
from tests.test_env_wrapper import StubSC2Env


def _make_env():
    def _init():
        timesteps = [
            fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=150, food_cap=15),
            fake.make_timestep(units=[fake.scv(1, x=5, y=5), fake.marine(2, x=10, y=10)],
                                minerals=100, food_cap=15),
            fake.make_timestep(units=[fake.marine(2, x=10, y=10)], minerals=50, food_cap=15,
                                reward=1.0, step_type="LAST"),
        ]
        stub = StubSC2Env(timesteps)
        env = SC2FightEnv(EnvConfig(), env_factory=lambda cfg: stub)
        return Monitor(env)

    return _init


def test_resume_from_checkpoint_continues_training(tmp_path):
    vec_env = DummyVecEnv([_make_env()])
    model = MaskablePPO("MlpPolicy", vec_env, n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    model.learn(total_timesteps=16)
    assert model.num_timesteps >= 16

    checkpoint_path = tmp_path / "checkpoint.zip"
    model.save(str(checkpoint_path))

    resumed_env = DummyVecEnv([_make_env()])
    resumed_model = MaskablePPO.load(str(checkpoint_path), env=resumed_env)
    timesteps_at_resume = resumed_model.num_timesteps
    assert timesteps_at_resume == model.num_timesteps  # picked up where it left off

    resumed_model.learn(total_timesteps=16, reset_num_timesteps=False)
    assert resumed_model.num_timesteps == timesteps_at_resume + 16  # advanced, not reset to 16
