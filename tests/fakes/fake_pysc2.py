"""Fakes that mimic the attribute-access shape of a real PySC2 TimeStep, so
GameState.from_observation() and higher-level code can be exercised without a
live StarCraft II client. Only the fields sc2rl actually reads are modeled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sc2rl.env.game_state import is_structure_type

ALLIANCE_SELF = 1
ALLIANCE_ENEMY = 4

UNIT_SCV = 45
UNIT_COMMAND_CENTER = 18
UNIT_SUPPLY_DEPOT = 19
UNIT_BARRACKS = 21
UNIT_MARINE = 48
UNIT_NEXUS = 59
UNIT_HATCHERY = 86


@dataclass
class FakeUnit:
    """Doubles as raw-observation input to GameState.from_observation() (via
    its pysc2-shaped fields) AND as a drop-in UnitInfo stand-in for tests that
    populate a GameState's unit lists directly -- hence the UnitInfo-mirrored
    is_idle/is_complete/health_fraction properties below.
    """

    tag: int
    unit_type: int
    alliance: int = ALLIANCE_SELF
    x: float = 0.0
    y: float = 0.0
    health: float = 45.0
    # Matches real raw_units, which has no health_max field -- only absolute
    # `health` and `health_ratio` (int(health / health_max * 255)). 255 = full health.
    health_ratio: int = 255
    build_progress: float = 1.0
    order_length: int = 0

    @property
    def is_idle(self) -> bool:
        return self.order_length == 0

    @property
    def is_complete(self) -> bool:
        return self.build_progress >= 1.0

    @property
    def health_fraction(self) -> float:
        return self.health_ratio / 255.0

    @property
    def is_structure(self) -> bool:
        return is_structure_type(self.unit_type)


@dataclass
class FakePlayer:
    minerals: int = 0
    food_used: int = 0
    food_cap: int = 15


_SCORE_CUMULATIVE_LEN = 13  # matches pysc2 features.ScoreCumulative


@dataclass
class FakeObservation:
    raw_units: list = field(default_factory=list)
    player: FakePlayer = field(default_factory=FakePlayer)
    game_loop: list = field(default_factory=lambda: [0])
    score_cumulative: list = field(default_factory=lambda: [0] * _SCORE_CUMULATIVE_LEN)
    # Real timesteps expose feature_minimap indexable by layer index; only the
    # `pathable` layer is modeled, keyed by pysc2's own index for it. None
    # (the default) mimics an observation without feature layers, which
    # makes the env fall back to "everything is pathable".
    feature_minimap: object = None


@dataclass
class FakeTimeStep:
    observation: FakeObservation = field(default_factory=FakeObservation)
    reward: float = 0.0
    step_type: str = "MID"

    def last(self) -> bool:
        return self.step_type == "LAST"

    def first(self) -> bool:
        return self.step_type == "FIRST"


def make_timestep(
    units: list[FakeUnit] | None = None,
    minerals: int = 0,
    food_used: int = 0,
    food_cap: int = 15,
    game_loop: int = 0,
    reward: float = 0.0,
    step_type: str = "MID",
    total_value_units: int = 0,
    total_value_structures: int = 0,
    killed_value_units: int = 0,
    killed_value_structures: int = 0,
    pathable=None,
    height=None,
) -> FakeTimeStep:
    score = [0] * _SCORE_CUMULATIVE_LEN
    score[3] = total_value_units
    score[4] = total_value_structures
    score[5] = killed_value_units
    score[6] = killed_value_structures
    feature_minimap = None
    if pathable is not None:
        from pysc2.lib import features

        feature_minimap = {features.MINIMAP_FEATURES.pathable.index: pathable}
        if height is not None:
            feature_minimap[features.MINIMAP_FEATURES.height_map.index] = height
    return FakeTimeStep(
        observation=FakeObservation(
            raw_units=units or [],
            player=FakePlayer(minerals=minerals, food_used=food_used, food_cap=food_cap),
            game_loop=[game_loop],
            score_cumulative=score,
            feature_minimap=feature_minimap,
        ),
        reward=reward,
        step_type=step_type,
    )


def scv(tag: int, x: float = 0.0, y: float = 0.0, idle: bool = True) -> FakeUnit:
    return FakeUnit(
        tag=tag, unit_type=UNIT_SCV, alliance=ALLIANCE_SELF, x=x, y=y,
        order_length=0 if idle else 1,
    )


def marine(tag: int, x: float = 0.0, y: float = 0.0, health: float = 45.0,
           health_ratio: int = 255, idle: bool = True) -> FakeUnit:
    return FakeUnit(
        tag=tag, unit_type=UNIT_MARINE, alliance=ALLIANCE_SELF, x=x, y=y,
        health=health, health_ratio=health_ratio, order_length=0 if idle else 1,
    )


def enemy_unit(tag: int, unit_type: int, x: float = 0.0, y: float = 0.0,
                health: float = 45.0, health_ratio: int = 255) -> FakeUnit:
    return FakeUnit(
        tag=tag, unit_type=unit_type, alliance=ALLIANCE_ENEMY, x=x, y=y,
        health=health, health_ratio=health_ratio,
    )


def command_center(tag: int, x: float = 0.0, y: float = 0.0) -> FakeUnit:
    return FakeUnit(tag=tag, unit_type=UNIT_COMMAND_CENTER, alliance=ALLIANCE_SELF, x=x, y=y,
                     health=1500.0, health_ratio=255)


def supply_depot(tag: int, x: float = 0.0, y: float = 0.0, complete: bool = True) -> FakeUnit:
    return FakeUnit(tag=tag, unit_type=UNIT_SUPPLY_DEPOT, alliance=ALLIANCE_SELF, x=x, y=y,
                     build_progress=1.0 if complete else 0.5)


def barracks(tag: int, x: float = 0.0, y: float = 0.0, complete: bool = True) -> FakeUnit:
    return FakeUnit(tag=tag, unit_type=UNIT_BARRACKS, alliance=ALLIANCE_SELF, x=x, y=y,
                     health=1000.0, health_ratio=255,
                     build_progress=1.0 if complete else 0.5)
