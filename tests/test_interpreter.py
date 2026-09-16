from types import SimpleNamespace

import numpy as np

from sc2rl.command.interpreter import CANNOT_COMPLY, interpret_command
from sc2rl.env.action_space import ActionSpaceSpec
from sc2rl.env.game_state import GameState
from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation
from tests.fakes import fake_pysc2 as fake


def make_spec() -> ActionSpaceSpec:
    return ActionSpaceSpec(grid=SectorGrid(map_size=64, cols=4, rows=4))


def identity_orientation(spec: ActionSpaceSpec) -> SpawnOrientation:
    return SpawnOrientation(map_size=spec.grid.map_size, mirror_x=False, mirror_y=False)


def fake_client(action: str, message: str = "ok"):
    """Mimics anthropic.Anthropic's client.messages.create(**kwargs) shape,
    returning a canned forced-tool-use response. Records every call's kwargs
    on client.calls for assertions."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        tool_use = SimpleNamespace(type="tool_use", input={"action": action, "message": message})
        return SimpleNamespace(content=[tool_use])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    client.calls = calls
    return client


def test_interpret_command_resolves_a_legal_action():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=200, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client(action="build_supply_depot", message="Building a supply depot.")

    result = interpret_command(client, "build a depot", state, spec, mask, identity_orientation(spec))

    assert result.action_index == spec.index_for_name("build_supply_depot")
    assert result.action_name == "build_supply_depot"
    assert result.message == "Building a supply depot."


def test_interpret_command_only_offers_currently_legal_actions_to_the_model():
    # Regression guard for the actual point of this function: Claude must
    # never even be offered an action the mask currently forbids.
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True  # only no_op legal
    client = fake_client(action="no_op", message="Can't do that yet.")

    interpret_command(client, "attack the enemy base", state, spec, mask, identity_orientation(spec))

    offered = client.calls[0]["tools"][0]["input_schema"]["properties"]["action"]["enum"]
    assert offered == ["no_op", CANNOT_COMPLY]


def test_interpret_command_forces_the_choose_action_tool():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client(action="no_op")

    interpret_command(client, "wait", state, spec, mask, identity_orientation(spec))

    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "choose_action"}


def test_interpret_command_handles_decline():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)
    client = fake_client(action=CANNOT_COMPLY, message="Nothing matches that request.")

    result = interpret_command(client, "do a backflip", state, spec, mask, identity_orientation(spec))

    assert result.action_index is None
    assert result.action_name == CANNOT_COMPLY
    assert result.message == "Nothing matches that request."


def test_interpret_command_treats_a_hallucinated_action_name_as_decline():
    # A forced tool schema's enum is not strictly enforced server side --
    # confirmed live: a fast model can still name an action outside this
    # turn's actual legal list. The model's own `message` for that case was
    # written assuming the choice would be honored ("Training a marine."),
    # which would read as a false confirmation of something that didn't
    # happen -- the ground-truth reason must replace it, not the model's text.
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    client = fake_client(action="train_marine", message="Training a marine.")  # illegal under this mask

    result = interpret_command(client, "make a marine", state, spec, mask, identity_orientation(spec))

    assert result.action_index is None
    assert result.action_name == CANNOT_COMPLY
    assert "Training a marine." not in result.message
    assert "completed barracks" in result.message


def test_interpret_command_explains_missing_supply_depot_when_model_names_barracks_anyway():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=500, food_used=0, food_cap=15)  # no supply_depots
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    client = fake_client(action="build_barracks", message="Building a barracks now.")

    result = interpret_command(client, "build a barracks", state, spec, mask, identity_orientation(spec))

    assert result.action_index is None
    assert "Building a barracks now." not in result.message
    assert "supply depot" in result.message


def test_interpret_command_explains_missing_scv_for_a_named_but_illegal_build():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=500, food_used=0, food_cap=15)  # no scvs
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    client = fake_client(action="build_supply_depot", message="Building a supply depot now.")

    result = interpret_command(client, "build a depot", state, spec, mask, identity_orientation(spec))

    assert "SCV" in result.message


def test_interpret_command_explains_army_not_mobilized_for_a_named_but_illegal_move():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    for i in range(4):
        state.marines.append(fake.marine(i))
    mask = np.zeros(spec.num_actions, dtype=bool)
    mask[0] = True
    far_sector_name = spec.name(spec.move_action_for_sector(spec.grid.num_sectors - 1))
    client = fake_client(action=far_sector_name, message="Attacking now.")

    result = interpret_command(client, "attack the enemy base", state, spec, mask, identity_orientation(spec))

    assert "Attacking now." not in result.message
    assert "4 marines" in result.message


def test_interpret_command_returns_decline_on_api_error_instead_of_raising():
    spec = make_spec()
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    mask = np.ones(spec.num_actions, dtype=bool)

    def raise_error(**kwargs):
        raise RuntimeError("network down")

    client = SimpleNamespace(messages=SimpleNamespace(create=raise_error))

    result = interpret_command(client, "return to base", state, spec, mask, identity_orientation(spec))

    assert result.action_index is None
    assert result.action_name == CANNOT_COMPLY
    assert "network down" in result.message
