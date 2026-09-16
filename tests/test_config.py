from sc2rl.config import Config


def test_default_yaml_loads_and_matches_defaults():
    config = Config.from_yaml("configs/default.yaml")
    assert config.env.map_name == "Simple64"
    assert config.env.grid.cols == 6
    assert config.env.grid.rows == 6
    assert config.env.reward.shaping_enabled is True
    assert config.training.ppo.total_timesteps == 200000


def test_partial_override_falls_back_to_defaults():
    config = Config.from_dict({"env": {"map_name": "CustomMap"}})
    assert config.env.map_name == "CustomMap"
    assert config.env.opponent_race == "zerg"  # untouched default
    assert config.env.grid.cols == 4  # nested default preserved


def test_empty_dict_gives_full_defaults():
    config = Config.from_dict({})
    assert config.env.map_name == "Simple64"
    assert config.training.checkpoint_dir == "checkpoints"


def test_train_fast_yaml_loads():
    config = Config.from_yaml("configs/train_fast.yaml")
    assert config.training.ppo.total_timesteps == 2000
    assert config.training.checkpoint_dir == "checkpoints/fast"


def test_realtime_defaults_off_and_is_overridable():
    # Off by default: training, evaluation, and demonstration collection all
    # want the game stepped as fast as the client can simulate, not paced to
    # true StarCraft II speed. interactive_play.py opts back in via its own
    # CLI default, not this config default -- see its --no-realtime flag.
    assert Config.from_dict({}).env.realtime is False
    assert Config.from_dict({"env": {"realtime": True}}).env.realtime is True
