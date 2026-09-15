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

    @staticmethod
    def from_dict(data: dict) -> "MaskConfig":
        defaults = MaskConfig()
        return MaskConfig(**{**defaults.__dict__, **data})


@dataclass
class RewardConfig:
    shaping_enabled: bool = False
    shaping_coefficient: float = 0.01

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
