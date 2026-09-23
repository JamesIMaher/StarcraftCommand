import json
from types import SimpleNamespace

import numpy as np

from sc2rl.command.interpreter import (
    ActionResult,
    DeclineResult,
    DirectiveResult,
    DispatchResult,
    ReleaseResult,
    interpret_command,
)
from sc2rl.env.action_space import ActionSpaceSpec
from sc2rl.env.game_state import GameState
from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation
from tests.fakes import fake_pysc2 as fake


def make_spec() -> ActionSpaceSpec:
    return ActionSpaceSpec(grid=SectorGrid(map_size=64, cols=4, rows=4))


def identity_orientation(spec: ActionSpaceSpec) -> SpawnOrientation:
    return SpawnOrientation(map_size=spec.grid.map_size, mirror_x=False, mirror_y=False)


def fake_client(tool_name: str, **tool_input):
    """Mimics anthropic.Anthropic's client.messages.create(**kwargs) shape,
    returning a canned forced-tool-use response naming `tool_name`. Records
    every call's kwargs on client.calls for assertions."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        tool_use = SimpleNamespace(type="tool_use", name=tool_name, input=dict(tool_input))
        return SimpleNamespace(content=[tool_use])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    client.calls = calls
    return client


def find_tool(client, name: str) -> dict:
    return next(t for t in client.calls[0]["tools"] if t["name"] == name)


# --- choose_action ----------------------------------------------------------

def test_choose_action_resolves_a_legal_action():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("choose_action", action="build_supply_depot", message="Building a supply depot.")

    result = interpret_command(client, "build a depot", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, ActionResult)
    assert result.action_index == spec.index_for_name("build_supply_depot")
    assert result.action_name == "build_supply_depot"
    assert result.message == "Building a supply depot."


def test_choose_action_tool_only_offers_currently_legal_actions():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True  # only no_op legal
    client = fake_client("choose_action", action="no_op", message="Waiting.")

    interpret_command(client, "attack the enemy base", state, spec, mask, identity_orientation(spec))

    assert find_tool(client, "choose_action")["input_schema"]["properties"]["action"]["enum"] == ["no_op"]


def test_all_five_tools_are_always_offered_with_tool_choice_any():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("decline", message="n/a")

    interpret_command(client, "anything", state, spec, mask, identity_orientation(spec))

    offered = {t["name"] for t in client.calls[0]["tools"]}
    assert offered == {"choose_action", "dispatch_units", "release_units", "set_directive", "decline"}
    assert client.calls[0]["tool_choice"] == {"type": "any"}


def test_choose_action_declines_with_a_ground_truth_reason_when_model_ignores_the_enum():
    # A forced tool schema's enum is not strictly enforced server side --
    # confirmed live: a fast model can still name an action outside this
    # turn's actual legal list, with a `message` written assuming the choice
    # would be honored ("Training a marine.") -- that text must never reach
    # the player, since it would read as a false confirmation.
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    client = fake_client("choose_action", action="train_marine", message="Training a marine.")

    result = interpret_command(client, "make a marine", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)
    assert "Training a marine." not in result.message
    assert "completed barracks" in result.message


def test_choose_action_explains_missing_supply_depot_for_a_named_but_illegal_barracks():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=500, food_used=0, food_cap=15)  # no supply_depots
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    client = fake_client("choose_action", action="build_barracks", message="Building a barracks now.")

    result = interpret_command(client, "build a barracks", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)
    assert "Building a barracks now." not in result.message
    assert "supply depot" in result.message


def test_choose_action_explains_army_not_mobilized_for_a_named_but_illegal_move():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for i in range(4):
        state.marines.append(fake.marine(i))
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    far_sector_name = spec.name(spec.move_action_for_sector(spec.grid.num_sectors - 1))
    client = fake_client("choose_action", action=far_sector_name, message="Attacking now.")

    result = interpret_command(client, "attack the enemy base", state, spec, mask, identity_orientation(spec))

    assert "Attacking now." not in result.message
    assert "4 marines" in result.message


# --- dispatch_units ----------------------------------------------------------

def test_dispatch_units_resolves_an_explicit_count():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("dispatch_units", unit_count="1", target_sector=5, message="Sending a marine.")

    result = interpret_command(client, "send a marine to sector 5", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DispatchResult)
    assert result.unit_count == 1
    assert result.target_sector == 5
    assert result.message == "Sending a marine."


def test_dispatch_units_resolves_all_as_none():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("dispatch_units", unit_count="all", target_sector=3, message="Sending everyone.")

    result = interpret_command(client, "send all marines to sector 3", state, spec, mask, identity_orientation(spec))

    assert result.unit_count is None


def test_dispatch_units_declines_an_unparseable_count():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("dispatch_units", unit_count="a few maybe", target_sector=3, message="Sending marines.")

    result = interpret_command(client, "send some marines", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)


def test_dispatch_units_declines_an_out_of_range_sector():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("dispatch_units", unit_count="1", target_sector=999, message="Sending a marine.")

    result = interpret_command(client, "send a marine to sector 999", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)


# --- release_units ------------------------------------------------------------

def test_release_units_all():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("release_units", scope="all", message="Returning everyone to AI control.")

    result = interpret_command(client, "return all units to AI control", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, ReleaseResult)
    assert result.sector is None
    assert result.most_recent is False


def test_release_units_most_recent():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("release_units", scope="most_recent", message="Releasing that marine.")

    result = interpret_command(client, "release the marine I just sent", state, spec, mask, identity_orientation(spec))

    assert result.most_recent is True
    assert result.sector is None


def test_release_units_by_sector():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("release_units", scope="sector", target_sector=12, message="Releasing units at sector 12.")

    result = interpret_command(client, "release whoever's at sector 12", state, spec, mask, identity_orientation(spec))

    assert result.sector == 12
    assert result.most_recent is False


# --- set_directive -------------------------------------------------------------

def test_set_directive_bans_an_action():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client(
        "set_directive", action_name="build_supply_depot", allowed=False,
        message="No more supply depots.",
    )

    result = interpret_command(
        client, "do not build any more supply depots", state, spec, mask, identity_orientation(spec),
    )

    assert isinstance(result, DirectiveResult)
    assert result.action_name == "build_supply_depot"
    assert result.allowed is False


def test_set_directive_offers_the_full_vocabulary_minus_no_op():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.zeros(spec.num_actions, dtype=bool)  # even with almost nothing legal right now
    mask[0] = True
    client = fake_client("set_directive", action_name="build_barracks", allowed=False, message="Banned.")

    interpret_command(client, "never build barracks", state, spec, mask, identity_orientation(spec))

    enum = find_tool(client, "set_directive")["input_schema"]["properties"]["action_name"]["enum"]
    assert "no_op" not in enum
    assert "build_barracks" in enum
    assert len(enum) == spec.num_actions - 1  # every real action, regardless of current legality


def test_set_directive_declines_an_unrecognized_action_name():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("set_directive", action_name="nuke_everything", allowed=False, message="Banning it.")

    result = interpret_command(client, "no nukes", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)


# --- decline / errors ----------------------------------------------------------

def test_decline_tool():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client("decline", message="Nothing matches that request.")

    result = interpret_command(client, "do a backflip", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)
    assert result.message == "Nothing matches that request."


def test_interpret_command_returns_decline_on_api_error_instead_of_raising():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)

    def raise_error(**kwargs):
        raise RuntimeError("network down")

    client = SimpleNamespace(messages=SimpleNamespace(create=raise_error))

    result = interpret_command(client, "return to base", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)
    assert "network down" in result.message


# --- OpenAI-compatible clients (e.g. gpt-oss-120b on Kamiwaza) -------------------

def fake_openai_client(tool_name: str | None, content: str | None = None, **tool_input):
    """Mimics openai.OpenAI's client.chat.completions.create(**kwargs) shape.
    `tool_name=None` returns a prose-only reply with no tool call."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        tool_calls = None
        if tool_name is not None:
            function = SimpleNamespace(name=tool_name, arguments=json.dumps(tool_input))
            tool_calls = [SimpleNamespace(type="function", function=function)]
        message = SimpleNamespace(tool_calls=tool_calls, content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


def test_openai_client_gets_function_tools_with_tool_choice_required():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    client = fake_openai_client("decline", message="n/a")

    interpret_command(client, "anything", state, spec, mask, identity_orientation(spec), model="gpt-oss-120b")

    call = client.calls[0]
    assert call["model"] == "gpt-oss-120b"
    assert call["tool_choice"] == "required"
    assert call["messages"][0]["role"] == "system"
    assert "Player command: 'anything'" in call["messages"][1]["content"]
    offered = {t["function"]["name"]: t for t in call["tools"]}
    assert set(offered) == {"choose_action", "dispatch_units", "release_units", "set_directive", "decline"}
    assert all(t["type"] == "function" for t in call["tools"])
    assert offered["choose_action"]["function"]["parameters"]["properties"]["action"]["enum"] == ["no_op"]


def test_openai_client_tool_call_arguments_are_parsed_like_claude_input():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_openai_client("dispatch_units", unit_count="3", target_sector=5, message="Sending three.")

    result = interpret_command(client, "take a few marines to 5", state, spec, mask, identity_orientation(spec))

    assert result == DispatchResult(unit_count=3, target_sector=5, message="Sending three.")


def test_openai_client_enum_violation_still_declines_with_a_ground_truth_reason():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    client = fake_openai_client("choose_action", action="train_marine", message="Training a marine.")

    result = interpret_command(client, "make a marine", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)
    assert "barracks" in result.message


def test_openai_client_prose_reply_without_a_tool_call_declines():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_openai_client(None, content="I'm not sure what you mean.")

    result = interpret_command(client, "blorp", state, spec, mask, identity_orientation(spec))

    assert result == DeclineResult(message="I'm not sure what you mean.")


def test_openai_client_api_error_declines_instead_of_raising():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)

    def raise_error(**kwargs):
        raise RuntimeError("proxy unreachable")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=raise_error)))

    result = interpret_command(client, "return to base", state, spec, mask, identity_orientation(spec))

    assert isinstance(result, DeclineResult)
    assert "proxy unreachable" in result.message
