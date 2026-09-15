"""Converts a GameState snapshot into the flat feature vector fed to the policy.

Expands on the old repo's 9 global scalars with per-sector spatial density
(friendly/enemy count and health per grid cell), using the same SectorGrid as
the movement action space, so the network actually has the spatial context its
movement decisions need -- the old repo's action space was spatial but its
inputs were not.

Two of the per-sector features are "minimap memory" rather than a snapshot
of the current step:
  - known enemy structures: buildings the player has seen. PySC2 keeps a
    previously-seen structure in raw_units as a display_type=Snapshot entry
    while it's in fog, so this persists until the area is re-seen without it
    -- exactly what a human reads off the minimap.
  - explored: whether a friendly marine has been in that sector this
    episode. Tracked by the env (it's episode state, not part of any single
    GameState) and passed in. Without it the policy has no way to tell an
    empty sector it has already swept from one it has never looked at, so it
    can't do a systematic search -- observed live as the army missing enemy
    buildings in sectors it simply never visited.
"""

from __future__ import annotations

from typing import Collection

import numpy as np

from .game_state import GameState
from .sector_grid import SectorGrid, SpawnOrientation

_MINERALS_NORM = 1000.0
_SUPPLY_CAP_NORM = 200.0
_MARINE_COUNT_NORM = 50.0
_SCV_COUNT_NORM = 30.0
_SUPPLY_DEPOT_COUNT_NORM = 8.0
_BARRACKS_COUNT_NORM = 8.0
_ENEMY_COUNT_NORM = 50.0
_SECTOR_COUNT_NORM = 10.0
_SECTOR_STRUCTURE_NORM = 5.0

FEATURES_PER_SECTOR = 6


def featurize(
    state: GameState,
    grid: SectorGrid,
    max_game_loop: int,
    orientation: SpawnOrientation,
    explored_sectors: Collection[int] = (),
) -> np.ndarray:
    features: list[float] = []

    features.append(min(state.minerals / _MINERALS_NORM, 1.0))
    food_cap = max(state.food_cap, 1)
    features.append(min(state.food_used / food_cap, 1.0))
    features.append(min(state.food_cap / _SUPPLY_CAP_NORM, 1.0))
    features.append(max(0.0, state.supply_headroom / food_cap))

    num_marines = len(state.marines)
    features.append(min(num_marines / _MARINE_COUNT_NORM, 1.0))
    avg_marine_health = (
        sum(u.health_fraction for u in state.marines) / num_marines if num_marines else 0.0
    )
    features.append(avg_marine_health)

    num_scvs = len(state.scvs)
    features.append(min(num_scvs / _SCV_COUNT_NORM, 1.0))
    features.append(min(len(state.idle_scvs) / _SCV_COUNT_NORM, 1.0))

    features.append(min(len(state.supply_depots) / _SUPPLY_DEPOT_COUNT_NORM, 1.0))
    features.append(min(len(state.complete_supply_depots) / _SUPPLY_DEPOT_COUNT_NORM, 1.0))
    features.append(min(len(state.barracks) / _BARRACKS_COUNT_NORM, 1.0))
    features.append(min(len(state.complete_barracks) / _BARRACKS_COUNT_NORM, 1.0))

    num_enemies = len(state.enemies)
    features.append(min(num_enemies / _ENEMY_COUNT_NORM, 1.0))
    avg_enemy_health = (
        sum(u.health_fraction for u in state.enemies) / num_enemies if num_enemies else 0.0
    )
    features.append(avg_enemy_health)

    features.append(1.0 if state.enemy_race_terran else 0.0)
    features.append(1.0 if state.enemy_race_protoss else 0.0)
    features.append(1.0 if state.enemy_race_zerg else 0.0)
    known_race = state.enemy_race_terran or state.enemy_race_protoss or state.enemy_race_zerg
    features.append(0.0 if known_race else 1.0)

    features.append(min(state.game_loop / max(max_game_loop, 1), 1.0))

    sector_friendly_count = [0] * grid.num_sectors
    sector_friendly_health = [0.0] * grid.num_sectors
    sector_enemy_count = [0] * grid.num_sectors
    sector_enemy_health = [0.0] * grid.num_sectors
    sector_enemy_structures = [0] * grid.num_sectors

    for unit in state.marines:
        cx, cy = orientation.to_canonical(unit.x, unit.y)
        s = grid.sector_of(cx, cy)
        sector_friendly_count[s] += 1
        sector_friendly_health[s] += unit.health_fraction

    for unit in state.enemies:
        cx, cy = orientation.to_canonical(unit.x, unit.y)
        s = grid.sector_of(cx, cy)
        sector_enemy_count[s] += 1
        sector_enemy_health[s] += unit.health_fraction
        if unit.is_structure:
            sector_enemy_structures[s] += 1

    explored = set(explored_sectors)
    for s in range(grid.num_sectors):
        features.append(min(sector_friendly_count[s] / _SECTOR_COUNT_NORM, 1.0))
        features.append(min(sector_friendly_health[s] / _SECTOR_COUNT_NORM, 1.0))
        features.append(min(sector_enemy_count[s] / _SECTOR_COUNT_NORM, 1.0))
        features.append(min(sector_enemy_health[s] / _SECTOR_COUNT_NORM, 1.0))
        features.append(min(sector_enemy_structures[s] / _SECTOR_STRUCTURE_NORM, 1.0))
        features.append(1.0 if s in explored else 0.0)

    return np.array(features, dtype=np.float32)


def observation_length(grid: SectorGrid) -> int:
    """Computed rather than hardcoded, so it can never drift out of sync with
    featurize() as fields are added. Orientation only affects values, never
    the vector length, so an identity orientation is fine here."""
    empty_state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=1)
    identity = SpawnOrientation(map_size=grid.map_size, mirror_x=False, mirror_y=False)
    return len(featurize(empty_state, grid, max_game_loop=1, orientation=identity))
