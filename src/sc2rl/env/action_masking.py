"""Computes the hard game-state legality mask MaskablePPO consumes via
`action_masks()`. Pure function of (GameState, spec, config) -- no PySC2
imports, no mutable/episode-tracked state (cooldowns for in-flight build
orders are tracked separately by the env wrapper and ANDed with this mask,
keeping this function fully unit-testable in isolation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_space import ActionSpaceSpec, FixedAction
from .game_state import GameState


@dataclass(frozen=True)
class MaskingConfig:
    max_supply_depots: int = 8
    max_barracks: int = 4
    supply_depot_minerals: int = 100
    barracks_minerals: int = 150
    marine_minerals: int = 50
    # Concentration of Force: below this many marines, movement/attack
    # actions are illegal entirely. Newly trained marines spawn near home, so
    # while blocked they simply stay clustered there (passive defense)
    # instead of being sent out piecemeal.
    min_marines_to_move: int = 4


def compute_action_masks(state: GameState, spec: ActionSpaceSpec, config: MaskingConfig) -> np.ndarray:
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[FixedAction.NO_OP] = True

    # Deliberately NOT gated on supply headroom being "low enough to justify
    # it" -- an earlier version required headroom <= a threshold, which
    # deadlocked the whole economy: at game start headroom is already above
    # any reasonable threshold and nothing else can lower it (marines need a
    # barracks, a barracks needs a depot), so BUILD_SUPPLY_DEPOT never became
    # legal. When to build is exactly the kind of timing decision the policy
    # is meant to learn -- the mask should only rule out what's illegal
    # (unaffordable, no worker, at the cap), not what's merely unwise yet.
    can_build_depot = (
        state.minerals >= config.supply_depot_minerals
        and len(state.scvs) > 0
        and len(state.supply_depots) < config.max_supply_depots
    )
    mask[FixedAction.BUILD_SUPPLY_DEPOT] = can_build_depot

    can_build_barracks = (
        state.minerals >= config.barracks_minerals
        and len(state.supply_depots) > 0
        and len(state.scvs) > 0
        and len(state.barracks) < config.max_barracks
    )
    mask[FixedAction.BUILD_BARRACKS] = can_build_barracks

    can_train_marine = (
        state.minerals >= config.marine_minerals
        and len(state.complete_barracks) > 0
        and state.supply_headroom > 0
    )
    mask[FixedAction.TRAIN_MARINE] = can_train_marine

    can_move = len(state.marines) >= config.min_marines_to_move
    for sector in range(spec.grid.num_sectors):
        mask[spec.move_action_for_sector(sector)] = can_move

    return mask
