"""Typed per-step snapshot of the game, parsed once from a PySC2 timestep's
`raw_units`/`player` observation. Both observation featurization and action
translation/masking read from this single snapshot instead of each re-scanning
`raw_units`.

`from_observation` only relies on attribute access on `obs.observation.raw_units`
/ `.player` / `.game_loop` — the same shape PySC2's real TimeStep exposes and the
fakes in tests/fakes/fake_pysc2.py mimic — so this module never imports pysc2
and works identically against a live game or a test fake.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_ALLIANCE_SELF = 1
_ALLIANCE_ALLY = 2
_ALLIANCE_ENEMY = 4

_UNIT_SCV = 45
_UNIT_COMMAND_CENTER = 18
_UNIT_SUPPLY_DEPOT = 19
_UNIT_BARRACKS = 21
_UNIT_MARINE = 48
_UNIT_NEXUS = 59
_UNIT_HATCHERY = 86

_BUILD_COMPLETE_PROGRESS = 1.0

# Indices into obs.observation.score_cumulative (pysc2's
# features.ScoreCumulative enum) -- the game engine maintains these itself
# every step, so reward shaping can read them directly instead of us
# re-deriving something weaker (e.g. summing unit health ourselves misses
# kills and economy entirely).
_SCORE_TOTAL_VALUE_UNITS = 3
_SCORE_TOTAL_VALUE_STRUCTURES = 4
_SCORE_KILLED_VALUE_UNITS = 5
_SCORE_KILLED_VALUE_STRUCTURES = 6


@dataclass(frozen=True)
class UnitInfo:
    tag: int
    unit_type: int
    x: float
    y: float
    health: float
    # PySC2's raw_units exposes no health_max field -- only absolute `health`
    # and `health_ratio`, an int(health / health_max * 255) already computed
    # by the game client (see pysc2 features.py). health_fraction below just
    # rescales that back to [0, 1].
    health_ratio: int
    build_progress: float
    order_length: int

    @property
    def is_idle(self) -> bool:
        return self.order_length == 0

    @property
    def is_complete(self) -> bool:
        return self.build_progress >= _BUILD_COMPLETE_PROGRESS

    @property
    def health_fraction(self) -> float:
        return self.health_ratio / 255.0


@dataclass
class GameState:
    game_loop: int
    minerals: int
    food_used: int
    food_cap: int
    total_value_units: int = 0
    total_value_structures: int = 0
    killed_value_units: int = 0
    killed_value_structures: int = 0

    scvs: list[UnitInfo] = field(default_factory=list)
    marines: list[UnitInfo] = field(default_factory=list)
    command_centers: list[UnitInfo] = field(default_factory=list)
    supply_depots: list[UnitInfo] = field(default_factory=list)
    barracks: list[UnitInfo] = field(default_factory=list)
    enemies: list[UnitInfo] = field(default_factory=list)

    enemy_race_terran: bool = False
    enemy_race_protoss: bool = False
    enemy_race_zerg: bool = False

    @property
    def idle_scvs(self) -> list[UnitInfo]:
        return [u for u in self.scvs if u.is_idle]

    @property
    def complete_supply_depots(self) -> list[UnitInfo]:
        return [u for u in self.supply_depots if u.is_complete]

    @property
    def complete_barracks(self) -> list[UnitInfo]:
        return [u for u in self.barracks if u.is_complete]

    @property
    def command_center_pos(self) -> tuple[float, float] | None:
        if not self.command_centers:
            return None
        cc = self.command_centers[0]
        return cc.x, cc.y

    @property
    def supply_headroom(self) -> int:
        return self.food_cap - self.food_used

    @classmethod
    def from_observation(cls, obs) -> "GameState":
        player = obs.observation.player
        score = obs.observation.score_cumulative
        state = cls(
            game_loop=int(obs.observation.game_loop[0]),
            minerals=int(player.minerals),
            food_used=int(player.food_used),
            food_cap=int(player.food_cap),
            total_value_units=int(score[_SCORE_TOTAL_VALUE_UNITS]),
            total_value_structures=int(score[_SCORE_TOTAL_VALUE_STRUCTURES]),
            killed_value_units=int(score[_SCORE_KILLED_VALUE_UNITS]),
            killed_value_structures=int(score[_SCORE_KILLED_VALUE_STRUCTURES]),
        )
        for unit in obs.observation.raw_units:
            info = UnitInfo(
                tag=int(unit.tag),
                unit_type=int(unit.unit_type),
                x=float(unit.x),
                y=float(unit.y),
                health=float(unit.health),
                health_ratio=int(unit.health_ratio),
                build_progress=float(unit.build_progress),
                order_length=int(unit.order_length),
            )
            if unit.alliance == _ALLIANCE_ENEMY:
                state.enemies.append(info)
                if unit.unit_type == _UNIT_COMMAND_CENTER:
                    state.enemy_race_terran = True
                elif unit.unit_type == _UNIT_NEXUS:
                    state.enemy_race_protoss = True
                elif unit.unit_type == _UNIT_HATCHERY:
                    state.enemy_race_zerg = True
            elif unit.alliance in (_ALLIANCE_SELF, _ALLIANCE_ALLY):
                if unit.unit_type == _UNIT_SCV:
                    state.scvs.append(info)
                elif unit.unit_type == _UNIT_MARINE:
                    state.marines.append(info)
                elif unit.unit_type == _UNIT_COMMAND_CENTER:
                    state.command_centers.append(info)
                elif unit.unit_type == _UNIT_SUPPLY_DEPOT:
                    state.supply_depots.append(info)
                elif unit.unit_type == _UNIT_BARRACKS:
                    state.barracks.append(info)
        return state
