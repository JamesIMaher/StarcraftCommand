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
    # Concentration of Force: don't legalize even a move/recall to the HOME
    # sector until at least this many marines exist. Newly trained marines
    # spawn near the home base, so while this gate is active they simply
    # stay clustered at home (passive defense). Deliberately low -- just a
    # "can the army move at all" floor, not "ready to go on offense"; see
    # min_marines_to_advance for that.
    min_marines_to_move: int = 4
    # Separate, higher bar for moving to any sector other than home --
    # matches ScriptedPolicyConfig.attack_threshold, the BC teacher's own
    # threshold for committing to a search-and-destroy offensive rather than
    # holding position. A hard mask, not just a reward incentive: without
    # it, nothing stopped the RL policy from sending the whole army into
    # unexplored territory with only min_marines_to_move marines -- observed
    # live as marines exploring too early with too few marines and dying
    # immediately, which then taught the policy to avoid moving altogether
    # rather than to wait for mass.
    min_marines_to_advance: int = 20

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
    # Ceiling on (total_value_units + total_value_structures) used for the
    # economic-value reward -- growing past this earns no further reward.
    # Structures are already capped by masking.max_supply_depots/
    # max_barracks, but marine count has no upper limit, so without this a
    # policy can farm reward indefinitely by hoarding marines it never risks
    # in combat (confirmed live: a losing episode earned +2.85 in economic
    # reward alone). ~4000 generously covers a fully-built base (~400 for 8
    # SCVs, ~2000 for maxed-out depots/barracks/command center) plus a real
    # ~25-marine fighting force (~1250) -- comfortably enough to reward
    # building a real force, not enough to reward hoarding past one.
    economic_value_cap: float = 4000.0
    # Mass / Concentration of Force: the killed_value portion of the shaping
    # reward is scaled by min(1, marine_count / concentration_threshold), so
    # a kill landed with a large army earns full credit while a kill landed
    # with a tiny, exposed squad earns much less -- discourages treating
    # opportunistic small-squad kills as a winning strategy on their own.
    # Matches MaskConfig.min_marines_to_advance -- both express "this is what
    # counts as a real, committed fighting force" for this scenario.
    concentration_threshold: int = 20
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
    # Ceiling on (killed_value_units + killed_value_structures) used for the
    # kill-value reward. killed_value only ever increases (kills aren't
    # "undone"), so like economic value it needs a cap: a long, grindy fight
    # trading kills against a continuously-spawning bot can otherwise run
    # this into the thousands with no natural ceiling the way
    # economic_value_cap gives total_value. ~2000 is generous (most of a
    # real enemy army/base) without being unbounded.
    kill_value_cap: float = 2000.0
    # Economy of Force / Security: per-step penalty while the home sector has
    # enemy units present and no friendly marines there to respond.
    home_defense_penalty: float = 0.05
    # Episodes have no step limit (only PySC2's own game-end conditions), so
    # an uncapped per-step penalty can accumulate for hundreds or thousands
    # of steps in an unusually long episode -- confirmed live: an early,
    # untrained episode's ep_rew_mean reached -46.4 after only 18,000
    # timesteps, far below anything the "any win beats any loss" analysis on
    # terminal_reward_scale accounted for, because that analysis only ever
    # reasoned about a single-step transition, never an episode-length
    # accumulation of an uncapped per-step penalty. Total home_defense
    # penalty within a single episode is clamped to this.
    home_defense_penalty_cap: float = 1.0
    # OODA loop (Observe): one-time reward the first time an enemy unit is
    # ever seen in a given sector during an episode -- rewards scouting
    # itself, separate from combat outcomes.
    scouting_bonus: float = 0.02
    # Counterweight to home_defense_penalty: that penalty fires every step
    # the home sector is undefended against a present enemy, for as long as
    # that holds, with no cap -- but scouting_bonus only pays once per
    # sector and economic reward stops entirely once economic_value_cap is
    # hit. Without this, once those one-off bonuses are exhausted the net
    # marginal reward of staying forward can be zero or negative while
    # returning home is strictly zero-or-better, which pulls the policy back
    # to base with nothing pulling it back out -- observed live as marines
    # oscillating near home instead of continuing to search. One-time reward
    # the first time a friendly marine is present in a given sector during
    # an episode (excluding the home sector, already "visited" at spawn),
    # independent of whether an enemy is there -- so covering new ground
    # itself has an ongoing payoff, not just finding something in it.
    exploration_bonus: float = 0.02
    # Further counterweight to the same stalling problem exploration_bonus
    # addresses: that bonus is one-time per sector, so once the army has
    # visited what it's going to visit for a while, there is still nothing
    # actively pulling it to keep moving -- observed live as a large army
    # parking in one sector indefinitely once assembled ("a huge pile of
    # marines in one location"). Flat per-step penalty once the army has
    # gone stale_search_patience steps without entering a sector it hasn't
    # been in before, applied only once movement is actually legal
    # (min_marines_to_move marines) so standing at home during the early
    # economy-building phase is never penalized.
    stale_search_penalty: float = 0.01
    stale_search_patience: int = 30
    # Same unbounded-episode-length reasoning as home_defense_penalty_cap --
    # total stale_search penalty within a single episode is clamped to this.
    stale_search_penalty_cap: float = 0.5
    # Multiplies PySC2's own terminal win/loss reward (+-1). Set well above
    # 1.0 deliberately: even with economic_value_cap and kill_value_cap in
    # place, a fully-built economy plus a long fight can still sum to a few
    # points of shaping reward regardless of outcome -- confirmed live, a
    # losing episode's ep_rew_mean stayed above +2. Capping individual
    # components bounds each one, but doesn't guarantee winning always beats
    # losing on its own; only the terminal term does that reliably. With the
    # current caps, worst-case POSITIVE shaping per episode is roughly
    # shaping_coefficient * (economic_value_cap + kill_value_scale *
    # kill_value_cap) + (scouting_bonus + exploration_bonus) * num_sectors
    # =~ 0.001 * (4000 + 0.1 * 2000) + 0.04 * 36 =~ 5.6, and worst-case
    # NEGATIVE shaping is roughly -(shaping_coefficient * economic_value_cap
    # + home_defense_penalty_cap + stale_search_penalty_cap) =~ -(4.0 + 1.0 +
    # 0.5) = -5.5 (economic_value_cap covers the worst case of the delta
    # collapsing from the cap to zero; kill_value never decreases so it has
    # no negative side). 10x here (+-10) comfortably dominates both
    # directions with margin, so a win's total reward is always positive and
    # a loss's is always negative, regardless of how much shaping either
    # episode racked up.
    terminal_reward_scale: float = 10.0

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
