"""Turns one natural-language player command into one legal action index,
via a single forced Claude tool call. The `client` (an anthropic.Anthropic
instance, or anything with the same `.messages.create(**kwargs)` shape) is
passed in rather than constructed here -- dependency injection, same as
`env_factory` elsewhere in this codebase, so this is testable with a fake
client and no network/API key.

Reuses action_space.py's existing name<->index mapping (`spec.name()`,
`spec.index_for_name()`) exactly as it was designed for: this module is the
"LLM-driven command layer" that module's docstring already anticipated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.action_space import ActionSpaceSpec
from ..env.game_state import GameState
from ..env.sector_grid import SpawnOrientation
from .state_summary import describe_state_for_llm

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# The one escape hatch outside the real action vocabulary: lets Claude
# decline instead of being forced into a legal-but-wrong action just
# because the schema requires picking something from the enum.
CANNOT_COMPLY = "cannot_comply"

_SYSTEM_PROMPT = """You are the tactical command interpreter for a StarCraft II \
bot fighting a Zerg opponent on Simple64. The player gives you a short, informal \
instruction. Translate it into EXACTLY ONE action from the legal list by calling \
choose_action, or call it with action="cannot_comply" if nothing in the legal \
list matches what they're asking for (explain briefly in `message`).

Action name reference:
- "no_op": do nothing this step.
- "build_supply_depot" / "build_barracks" / "train_marine": economy actions.
- "move_army_to_sector_N": send every marine NOT held in the home garrison to \
sector N. Moving to the HOME sector recalls the army to base -- use this for \
"return to base", "come home", "retreat", "defend the base", etc.
- To attack or search a specific area, pick the sector number nearest what the \
player described (prefer a sector already listed as holding a known enemy).

Always call choose_action exactly once. Keep `message` to one short sentence \
addressed to the player, confirming what you're doing (or explaining why you \
can't)."""


@dataclass(frozen=True)
class CommandResult:
    action_index: int | None  # None means declined (action_name == CANNOT_COMPLY)
    action_name: str
    message: str


def interpret_command(
    client,
    text: str,
    state: GameState,
    spec: ActionSpaceSpec,
    legal_mask: np.ndarray,
    orientation: SpawnOrientation,
    mobilized: bool = False,
    garrison_size: int = 0,
    model: str = DEFAULT_MODEL,
) -> CommandResult:
    legal_names = [spec.name(i) for i in range(spec.num_actions) if legal_mask[i]]
    state_description = describe_state_for_llm(state, spec.grid, orientation, mobilized, garrison_size)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Current game state:\n{state_description}\n\nPlayer command: {text!r}",
            }],
            tools=[{
                "name": "choose_action",
                "description": "Choose the one legal action that best carries out the player's command.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": legal_names + [CANNOT_COMPLY],
                            "description": "The action to take, or cannot_comply if none of the legal actions match.",
                        },
                        "message": {
                            "type": "string",
                            "description": "One short sentence to show the player confirming or explaining.",
                        },
                    },
                    "required": ["action", "message"],
                },
            }],
            tool_choice={"type": "tool", "name": "choose_action"},
        )
    except Exception as exc:  # network/API failure -- never crash the game loop over this
        return CommandResult(
            action_index=None, action_name=CANNOT_COMPLY,
            message=f"Sorry, I couldn't reach the AI service: {exc}",
        )

    tool_use = next(block for block in response.content if block.type == "tool_use")
    chosen = tool_use.input["action"]
    message = tool_use.input.get("message", "")

    if chosen == CANNOT_COMPLY or chosen not in legal_names:
        return CommandResult(action_index=None, action_name=CANNOT_COMPLY, message=message)
    return CommandResult(action_index=spec.index_for_name(chosen), action_name=chosen, message=message)
