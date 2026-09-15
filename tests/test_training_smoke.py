"""End-to-end smoke test for the MaskablePPO training wiring (env -> Monitor
-> DummyVecEnv -> MaskablePPO, including action-mask auto-detection) against a
stubbed SC2Env, so the whole training pipeline is validated without needing a
live StarCraft II client. Kept tiny (a handful of PPO update steps) so it
stays fast; it is not asserting learned behavior, only that the pipeline runs.
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
        # Short looping episode: two mid-game steps then a win, then the
        # DummyVecEnv auto-resets (mirroring pysc2's real auto-reset-on-step
        # behavior after StepType.LAST) and the cycle repeats.
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


def test_maskable_ppo_trains_against_stubbed_env_without_error():
    vec_env = DummyVecEnv([_make_env()])
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        verbose=0,
    )

    model.learn(total_timesteps=16)

    obs = vec_env.reset()
    action_masks = vec_env.env_method("action_masks")
    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
    assert action.shape == (1,)
    assert 0 <= int(action[0]) < vec_env.action_space.n
