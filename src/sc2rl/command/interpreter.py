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
choose_action.

The `action` you choose MUST be copied EXACTLY from the enum list in the tool \
schema for THIS turn -- that list is the complete set of actions actually \
available right now, and it changes turn to turn. An action name you recall from \
the reference below but that is NOT in this turn's enum is NOT available. Do not \
guess or assume it will work; call choose_action with action="cannot_comply" \
instead, and explain the likely reason using the game state you were given (e.g. \
"you don't have a supply depot yet" or "the army isn't mobilized -- you have 4 \
marines, but need at least 20 to launch an offensive").

Action name reference (for recognizing intent only -- always verify against this \
turn's actual enum before choosing):
- "no_op": do nothing this step.
- "build_supply_depot" / "build_barracks" / "train_marine": economy actions.
- "move_army_to_sector_N": send every marine NOT held in the home garrison to \
sector N. Moving to the HOME sector recalls the army to base -- use this for \
"return to base", "come home", "retreat", "defend the base", etc.
- To attack or search a specific area, pick the sector number nearest what the \
player described (prefer a sector already listed as holding a known enemy).

Always call choose_action exactly once. Keep `message` to one short sentence \
addressed to the player: confirming what you're doing if you found a legal match, \
or stating the SPECIFIC reason if you're declining -- never a sentence that reads \
like confirmation for something you didn't actually do."""


@dataclass(frozen=True)
class CommandResult:
    action_index: int | None  # None means declined (action_name == CANNOT_COMPLY)
    action_name: str
    message: str


def _explain_unavailable(action_name: str, state: GameState) -> str:
    """A ground-truth reason `action_name` isn't currently legal, computed
    from the actual game state rather than trusted from the model. Needed
    because a forced tool schema's `enum` is not strictly enforced server
    side: a small, fast model can still name an action outside this turn's
    actual legal list, and its own `message` for that case was written
    assuming the (illegal) choice would be honored -- showing it to the
    player reads as a false confirmation of something that didn't happen.
    Mirrors action_masking.py's legality checks in prose, not logic; keep
    the two in sync if the masking rules change."""
    if action_name == "build_supply_depot":
        if not state.scvs:
            return "there's no SCV available to build it"
        return "you can't afford it right now, or you're at the supply depot cap"
    if action_name == "build_barracks":
        if not state.supply_depots:
            return "you need a supply depot built first"
        if not state.scvs:
            return "there's no SCV available to build it"
        return "you can't afford it right now, or you're at the barracks cap"
    if action_name == "train_marine":
        if not state.complete_barracks:
            return "you don't have a completed barracks yet"
        if state.supply_headroom <= 0:
            return "you're at your supply cap"
        return "you can't afford it right now"
    if action_name.startswith("move_army_to_sector_"):
        return f"the army isn't mobilized enough for that yet -- you have {len(state.marines)} marines"
    return "that's not available right now"


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

    if chosen == CANNOT_COMPLY:
        return CommandResult(action_index=None, action_name=CANNOT_COMPLY, message=message)
    if chosen not in legal_names:
        # The model named something outside this turn's actual enum -- its
        # `message` was written assuming that choice would be honored, so
        # showing it would read as a false confirmation. Explain the real
        # reason from the game state instead of trusting the model's text.
        return CommandResult(
            action_index=None, action_name=CANNOT_COMPLY,
            message=f"Can't do that yet -- {_explain_unavailable(chosen, state)}.",
        )
    return CommandResult(action_index=spec.index_for_name(chosen), action_name=chosen, message=message)
