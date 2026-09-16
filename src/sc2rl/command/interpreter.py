"""Turns one natural-language player command into a structured intent, via a
single forced Claude tool call chosen from several ("tool_choice": "any").
The `client` (an anthropic.Anthropic instance, or anything with the same
`.messages.create(**kwargs)` shape) is passed in rather than constructed
here -- dependency injection, same as `env_factory` elsewhere in this
codebase, so this is testable with a fake client and no network/API key.

Five intents, one tool each, routed by which tool Claude calls:
- choose_action: execute one currently-legal action right now (the original
  v1 behavior).
- dispatch_units: pull specific marines out of autonomous control and send
  them somewhere under the player's own persistent control.
- release_units: return player-controlled marines to autonomous control.
- set_directive: ban or re-allow an action name for the autonomous policy,
  for the rest of the episode -- never restricts the player's own future
  explicit commands.
- decline: nothing else fits.

Reuses action_space.py's existing name<->index mapping (`spec.name()`,
`spec.index_for_name()`, `spec.all_names()`) exactly as it was designed for:
this module is the "LLM-driven command layer" that module's docstring
already anticipated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..env.action_space import ActionSpaceSpec
from ..env.game_state import GameState
from ..env.sector_grid import SpawnOrientation
from .state_summary import describe_state_for_llm

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class ActionResult:
    action_index: int
    action_name: str
    message: str


@dataclass(frozen=True)
class DispatchResult:
    # None means "every marine not already reserved" -- resolved by
    # SC2FightEnv.assign_to_player_control's own `count=None` meaning.
    unit_count: int | None
    target_sector: int
    message: str


@dataclass(frozen=True)
class ReleaseResult:
    # Exactly one of these describes the scope; all False/None means "all".
    sector: int | None
    most_recent: bool
    message: str


@dataclass(frozen=True)
class DirectiveResult:
    action_name: str
    allowed: bool  # False = ban, True = lift an existing ban
    message: str


@dataclass(frozen=True)
class DeclineResult:
    message: str


CommandResult = ActionResult | DispatchResult | ReleaseResult | DirectiveResult | DeclineResult


_SYSTEM_PROMPT = """You are the tactical command interpreter for a StarCraft II \
bot fighting a Zerg opponent on Simple64. The player gives you a short, informal \
instruction. Call EXACTLY ONE of the five tools available to you, based on what \
kind of instruction it is:

- choose_action: do ONE specific thing right now (build, train, or move/attack \
with the whole autonomous army). The `action` you choose MUST be copied EXACTLY \
from THAT tool's enum -- the complete set of actions actually available this \
turn, which changes turn to turn. An action name you recall from the reference \
below but that is NOT in this turn's enum is NOT available -- don't guess; use \
decline instead.

- dispatch_units: send SPECIFIC marines somewhere under the player's OWN \
persistent control, separate from the autonomous army -- use this whenever the \
player names or implies a subset of the force rather than "the army" as a whole \
(e.g. "send a marine to...", "take a few marines and...", "send everyone to..."). \
Infer unit_count from the wording: "a marine"/"one marine" -> "1", "a few"/"some" \
-> "3", "all"/"every marine"/"the whole army" -> "all". These units are EXCLUDED \
from the autonomous policy's own moves until released -- they hold position (and \
fight anything that comes into range) once they arrive, so one dispatch order is \
enough; you don't need to keep re-issuing it. Sector numbers are home-relative, NOT \
laid out like the screen -- when the player names an actual screen corner or edge \
("the bottom right corner of the map", "the top edge"), you MUST read the sector \
number off the state description's corner legend rather than computing it from the \
row/col index math; only use the row/col math for relative phrasing like "toward \
the enemy" or "near home".

- release_units: return player-controlled marines to autonomous AI control (e.g. \
"return all units to AI control", "let the AI handle everything again", "release \
the marine I just sent", "release whoever's at sector 12"). Use `scope="all"` for \
everyone, `scope="most_recent"` for the most recently dispatched group, or \
`scope="sector"` (with `target_sector`) for whichever player-controlled units are \
currently in that sector. Releasing one arbitrarily-described unit by identity \
(not by sector or recency) isn't supported by voice -- the console's own per-unit \
button handles that case; use decline and say so if that's clearly what's meant.

- set_directive: set or lift a STANDING rule that bans (or re-allows) the \
AUTONOMOUS policy from choosing a specific action for the rest of the game (e.g. \
"don't build any more supply depots" -> ban build_supply_depot; "okay you can \
build depots again" -> allow it). This NEVER restricts the player's own future \
one-off choose_action commands -- if the player later explicitly asks for the \
banned action anyway, honor it as a one-time exception via choose_action; a \
standing ban only ever constrains the autonomous policy's own choices.

- decline: nothing else fits, or a choose_action's intended target isn't actually \
available this turn. Give the SPECIFIC reason using the game state you were given \
(e.g. "you don't have a supply depot yet", "the army isn't mobilized -- you have \
4 marines, but need at least 20").

Action name reference (for recognizing intent only -- always verify against the \
CURRENT tool's actual enum before choosing):
- "no_op": do nothing this step.
- "build_supply_depot" / "build_barracks" / "train_marine": economy actions.
- "move_army_to_sector_N": the autonomous army (everyone not held back in the \
garrison or by the player) attack-moves to sector N. Moving to the HOME sector \
recalls it to base.

Always call exactly one tool. Keep `message` to one short sentence addressed to \
the player: confirming what you're doing, or -- for decline and any choose_action \
you had to fall back on -- the SPECIFIC reason, never a sentence that reads like \
confirmation for something you didn't actually do."""


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


def _safe_int(text: str) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _tool_schemas(legal_names: list[str], bannable_names: list[str], num_sectors: int) -> list[dict]:
    return [
        {
            "name": "choose_action",
            "description": "Execute one legal action right now.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": legal_names, "description": "The action to take."},
                    "message": {"type": "string", "description": "One short sentence confirming what you're doing."},
                },
                "required": ["action", "message"],
            },
        },
        {
            "name": "dispatch_units",
            "description": "Send specific marines to a sector under the player's own persistent control.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "unit_count": {
                        "type": "string",
                        "description": 'How many marines, inferred from wording: a plain integer as a '
                                       'string (e.g. "1", "3"), or "all" for every marine not already reserved.',
                    },
                    "target_sector": {
                        "type": "integer",
                        "description": f"Sector number (0..{num_sectors - 1}) to send them to.",
                    },
                    "message": {"type": "string", "description": "One short sentence confirming what you're doing."},
                },
                "required": ["unit_count", "target_sector", "message"],
            },
        },
        {
            "name": "release_units",
            "description": "Return player-controlled marines to autonomous AI control.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["all", "most_recent", "sector"],
                        "description": '"all", "most_recent" (the last dispatch), or "sector" '
                                       "(with target_sector).",
                    },
                    "target_sector": {
                        "type": "integer",
                        "description": 'Required when scope="sector": which sector to release units from.',
                    },
                    "message": {"type": "string", "description": "One short sentence confirming what you're doing."},
                },
                "required": ["scope", "message"],
            },
        },
        {
            "name": "set_directive",
            "description": "Ban or re-allow an action for the AUTONOMOUS policy for the rest of the game. "
                            "Never restricts the player's own future explicit commands.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action_name": {"type": "string", "enum": bannable_names, "description": "Which action."},
                    "allowed": {"type": "boolean", "description": "false to ban it, true to lift an existing ban."},
                    "message": {"type": "string", "description": "One short sentence confirming what you're doing."},
                },
                "required": ["action_name", "allowed", "message"],
            },
        },
        {
            "name": "decline",
            "description": "Nothing else fits, or a requested action isn't currently available.",
            "input_schema": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "The specific reason."}},
                "required": ["message"],
            },
        },
    ]


def interpret_command(
    client,
    text: str,
    state: GameState,
    spec: ActionSpaceSpec,
    legal_mask: np.ndarray,
    orientation: SpawnOrientation,
    mobilized: bool = False,
    garrison_size: int = 0,
    player_controlled_count: int = 0,
    banned_actions: tuple[str, ...] = (),
    model: str = DEFAULT_MODEL,
) -> CommandResult:
    legal_names = [spec.name(i) for i in range(spec.num_actions) if legal_mask[i]]
    bannable_names = [n for n in spec.all_names() if n != "no_op"]
    state_description = describe_state_for_llm(
        state, spec.grid, orientation, mobilized, garrison_size, player_controlled_count, banned_actions,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Current game state:\n{state_description}\n\nPlayer command: {text!r}",
            }],
            tools=_tool_schemas(legal_names, bannable_names, spec.grid.num_sectors),
            tool_choice={"type": "any"},
        )
    except Exception as exc:  # network/API failure -- never crash the game loop over this
        return DeclineResult(message=f"Sorry, I couldn't reach the AI service: {exc}")

    tool_use = next(block for block in response.content if block.type == "tool_use")
    inp = tool_use.input
    message = inp.get("message", "")

    if tool_use.name == "choose_action":
        chosen = inp.get("action")
        if chosen not in legal_names:
            # The model named something outside this turn's actual enum --
            # its `message` was written assuming that choice would be
            # honored, so showing it would read as a false confirmation.
            # Explain the real reason from the game state instead.
            return DeclineResult(message=f"Can't do that yet -- {_explain_unavailable(chosen, state)}.")
        return ActionResult(action_index=spec.index_for_name(chosen), action_name=chosen, message=message)

    if tool_use.name == "dispatch_units":
        raw_count = str(inp.get("unit_count", "")).strip().lower()
        if raw_count == "all":
            unit_count = None
        else:
            unit_count = _safe_int(raw_count)
            if unit_count is None or unit_count <= 0:
                return DeclineResult(message="I couldn't tell how many marines you meant.")
        target_sector = inp.get("target_sector")
        if target_sector is None or not (0 <= int(target_sector) < spec.grid.num_sectors):
            return DeclineResult(message="That sector doesn't exist.")
        return DispatchResult(unit_count=unit_count, target_sector=int(target_sector), message=message)

    if tool_use.name == "release_units":
        scope = inp.get("scope")
        if scope == "sector":
            target_sector = inp.get("target_sector")
            if target_sector is None or not (0 <= int(target_sector) < spec.grid.num_sectors):
                return DeclineResult(message="That sector doesn't exist.")
            return ReleaseResult(sector=int(target_sector), most_recent=False, message=message)
        if scope == "most_recent":
            return ReleaseResult(sector=None, most_recent=True, message=message)
        return ReleaseResult(sector=None, most_recent=False, message=message)  # "all" or unrecognized -> all

    if tool_use.name == "set_directive":
        action_name = inp.get("action_name")
        if action_name not in bannable_names:
            return DeclineResult(message=f'"{action_name}" isn\'t a recognized action.')
        return DirectiveResult(action_name=action_name, allowed=bool(inp.get("allowed", True)), message=message)

    return DeclineResult(message=message or "I can't do that.")
