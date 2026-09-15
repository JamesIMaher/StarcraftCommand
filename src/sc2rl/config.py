"""Typed configuration loaded from YAML. Each level has an explicit `from_dict`
rather than generic reflection-based deserialization -- the config is small
enough that being explicit is more robust (and easier to read) than a generic
recursive dataclass loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GridConfig:
    cols: int = 4
    rows: int = 4

    @staticmethod
    def from_dict(data: dict) -> "GridConfig":
        return GridConfig(cols=data.get("cols", 4), rows=data.get("rows", 4))


@dataclass
class MaskConfig:
    max_supply_depots: int = 8
    max_barracks: int = 4
    supply_depot_minerals: int = 100
    barracks_minerals: int = 150
    marine_minerals: int = 50
    # Concentration of Force: don't legalize move_army_to_sector_i actions
    # until at least this many marines exist. Newly trained marines spawn
    # near the home base, so while this gate is active they simply stay
    # clustered at home (passive defense) rather than being sent out
    # piecemeal -- this directly targets attacking/exploring with too few
    # units to survive contact.
    min_marines_to_move: int = 4

    @staticmethod
    def from_dict(data: dict) -> "MaskConfig":
        defaults = MaskConfig()
        return MaskConfig(**{**defaults.__dict__, **data})


@dataclass
class RewardConfig:
    shaping_enabled: bool = False
    # total_value_units / killed_value_units / killed_value_structures are
    # raw mineral-equivalent values -- they can accumulate into the hundreds
    # or more over a full episode as marines get built and enemies get
    # killed. This coefficient is scaled down accordingly so the summed
    # shaping reward over an episode stays roughly comparable to the
    # terminal +-1 win/loss reward rather than swamping it; treat it as a
    # starting point to tune, not a tuned value.
    shaping_coefficient: float = 0.001
    # Mass / Concentration of Force: the killed_value portion of the shaping
    # reward is scaled by min(1, marine_count / concentration_threshold), so
    # a kill landed with a large army earns full credit while a kill landed
    # with a tiny, exposed squad earns much less -- discourages treating
    # opportunistic small-squad kills as a winning strategy on their own.
    concentration_threshold: int = 4
    # killed_value_units/killed_value_structures only ever increase (kills
    # aren't "undone"), so unlike economic value it is NOT offset by an
    # eventual loss -- observed in practice: a losing episode's ep_rew_mean
    # went UP, because kills traded during a losing fight outweighed the
    # terminal penalty and the (comparatively small) economic-collapse
    # penalty. This is a separate, additional discount on top of
    # shaping_coefficient specifically for the killed-value portion, so
    # "traded some kills before losing" stays a minor bonus rather than
    # something that can rival actually winning.
    kill_value_scale: float = 0.1
    # Economy of Force / Security: per-step penalty while the home sector has
    # enemy units present and no friendly marines there to respond.
    home_defense_penalty: float = 0.05
    # OODA loop (Observe): one-time reward the first time an enemy unit is
    # ever seen in a given sector during an episode -- rewards scouting
    # itself, separate from combat outcomes.
    scouting_bonus: float = 0.02
    # Multiplies PySC2's own terminal win/loss reward (+-1). 1.0 = untouched.
    # Raise this if a full episode's cumulative shaping still rivals or
    # exceeds the terminal signal in magnitude -- shaping now applies on
    # every step including the terminal one (see sc2_env_wrapper.py), so a
    # losing episode's final collapse in economic value is captured; this
    # knob is for further tuning the balance if that alone isn't enough.
    terminal_reward_scale: float = 1.0

    @staticmethod
    def from_dict(data: dict) -> "RewardConfig":
        defaults = RewardConfig()
        return RewardConfig(**{**defaults.__dict__, **data})


@dataclass
class EnvConfig:
    map_name: str = "Simple64"
    opponent_race: str = "zerg"
    difficulty: str = "very_easy"
    map_size: int = 64
    step_mul: int = 8
    build_cooldown_steps: int = 3
    max_game_loop_norm: int = 20000
    visualize: bool = False
    grid: GridConfig = field(default_factory=GridConfig)
    masking: MaskConfig = field(default_factory=MaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @staticmethod
    def from_dict(data: dict) -> "EnvConfig":
        data = dict(data)
        grid = GridConfig.from_dict(data.pop("grid", {}) or {})
        masking = MaskConfig.from_dict(data.pop("masking", {}) or {})
        reward = RewardConfig.from_dict(data.pop("reward", {}) or {})
        defaults = EnvConfig()
        scalar_fields = {
            "map_name": defaults.map_name,
            "opponent_race": defaults.opponent_race,
            "difficulty": defaults.difficulty,
            "map_size": defaults.map_size,
            "step_mul": defaults.step_mul,
            "build_cooldown_steps": defaults.build_cooldown_steps,
            "max_game_loop_norm": defaults.max_game_loop_norm,
            "visualize": defaults.visualize,
        }
        scalar_fields.update(data)
        return EnvConfig(grid=grid, masking=masking, reward=reward, **scalar_fields)


@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    seed: int | None = None

    @staticmethod
    def from_dict(data: dict) -> "PPOConfig":
        defaults = PPOConfig()
        return PPOConfig(**{**defaults.__dict__, **data})


@dataclass
class TrainingConfig:
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 5000
    tensorboard_log: str = "runs"
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @staticmethod
    def from_dict(data: dict) -> "TrainingConfig":
        data = dict(data)
        ppo = PPOConfig.from_dict(data.pop("ppo", {}) or {})
        defaults = TrainingConfig()
        scalar_fields = {
            "checkpoint_dir": defaults.checkpoint_dir,
            "checkpoint_freq": defaults.checkpoint_freq,
            "tensorboard_log": defaults.tensorboard_log,
        }
        scalar_fields.update(data)
        return TrainingConfig(ppo=ppo, **scalar_fields)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @staticmethod
    def from_dict(data: dict) -> "Config":
        data = data or {}
        env = EnvConfig.from_dict(data.get("env", {}) or {})
        training = TrainingConfig.from_dict(data.get("training", {}) or {})
        return Config(env=env, training=training)

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return Config.from_dict(data)
