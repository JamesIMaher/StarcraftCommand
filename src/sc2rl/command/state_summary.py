"""Turns a GameState snapshot into a short, human-readable description for
an LLM -- deliberately separate from env/observation.py's featurize(),
which produces normalized floats for the neural network and means nothing
to a language model. Only what a human glancing at the game would know:
resources, army composition, and which sectors currently hold something
worth mentioning.
"""

from __future__ import annotations

from ..env.game_state import GameState
from ..env.sector_grid import SectorGrid, SpawnOrientation, home_sector, sectors_of


def describe_state_for_llm(
    state: GameState,
    grid: SectorGrid,
    orientation: SpawnOrientation,
    mobilized: bool = False,
    garrison_size: int = 0,
) -> str:
    home = home_sector(state.command_center_pos, grid, orientation)
    lines = [
        f"Sectors are numbered 0..{grid.num_sectors - 1} in row-major order over a "
        f"{grid.rows}x{grid.cols} grid (index = row * {grid.cols} + column); sector {home} "
        "is the HOME base.",
        f"Minerals: {state.minerals}. Supply: {state.food_used}/{state.food_cap}.",
        f"Marines: {len(state.marines)} total"
        + (f", including {garrison_size} permanently held at home as a garrison" if garrison_size else "")
        + ".",
        f"The army is {'' if mobilized else 'NOT '}currently mobilized for an offensive.",
        f"SCVs: {len(state.scvs)}. Supply depots: {len(state.complete_supply_depots)} complete, "
        f"{len(state.supply_depots) - len(state.complete_supply_depots)} building. "
        f"Barracks: {len(state.complete_barracks)} complete, "
        f"{len(state.barracks) - len(state.complete_barracks)} building.",
    ]

    if state.enemies:
        enemy_sectors = sorted(sectors_of(state.enemies, grid, orientation))
        structure_sectors = sorted(sectors_of(state.enemy_structures, grid, orientation))
        lines.append(f"Known enemy units visible in sectors: {enemy_sectors}.")
        if structure_sectors:
            lines.append(f"Known enemy structures (buildings) in sectors: {structure_sectors}.")
    else:
        lines.append("No enemy units or structures currently known.")

    return "\n".join(lines)
