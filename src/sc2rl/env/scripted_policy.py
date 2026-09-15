"""A simple, hand-written policy used as a behavior-cloning teacher
(imitation-learning warm start) -- NOT intended to play well on its own,
just to give the RL policy a reasonable starting point instead of random
weights. It operates in the exact same action space as the trained policy
(see action_space.py), so its choices are directly usable as supervised
training targets: build a small base, keep training marines, then either
defend home (if under attack and undefended) or advance on the enemy once
mobilized. Same spirit as AlphaStar's imitation-learning warm start from
human replays -- imitating our own scripted teacher instead, since our
action space is a simplified custom abstraction that doesn't correspond to
raw replay actions the way AlphaStar's full-interface action space did.
"""

from __future__ import annotations

from dataclasses import dataclass

from .action_masking import MaskingConfig, compute_action_masks
from .action_space import ActionSpaceSpec, FixedAction
from .game_state import GameState
from .sector_grid import SpawnOrientation

_HOME_SECTOR = 0


@dataclass(frozen=True)
class ScriptedPolicyConfig:
    target_supply_depots: int = 4
    target_barracks: int = 2


def scripted_action(
    state: GameState,
    spec: ActionSpaceSpec,
    masking_config: MaskingConfig,
    orientation: SpawnOrientation,
    policy_config: ScriptedPolicyConfig | None = None,
) -> int:
    policy_config = policy_config or ScriptedPolicyConfig()
    mask = compute_action_masks(state, spec, masking_config)

    if mask[FixedAction.BUILD_SUPPLY_DEPOT] and len(state.supply_depots) < policy_config.target_supply_depots:
        return int(FixedAction.BUILD_SUPPLY_DEPOT)
    if mask[FixedAction.BUILD_BARRACKS] and len(state.barracks) < policy_config.target_barracks:
        return int(FixedAction.BUILD_BARRACKS)
    if mask[FixedAction.TRAIN_MARINE]:
        return int(FixedAction.TRAIN_MARINE)

    can_move = any(mask[spec.move_action_for_sector(s)] for s in range(spec.grid.num_sectors))
    if can_move:
        home_action = spec.move_action_for_sector(_HOME_SECTOR)
        if mask[home_action] and _home_under_threat(state, spec, orientation):
            return home_action
        far_sector = spec.grid.num_sectors - 1
        return spec.move_action_for_sector(far_sector)

    return int(FixedAction.NO_OP)


def _home_under_threat(state: GameState, spec: ActionSpaceSpec, orientation: SpawnOrientation) -> bool:
    def sector_of(x: float, y: float) -> int:
        cx, cy = orientation.to_canonical(x, y)
        return spec.grid.sector_of(cx, cy)

    return any(sector_of(u.x, u.y) == _HOME_SECTOR for u in state.enemies)
