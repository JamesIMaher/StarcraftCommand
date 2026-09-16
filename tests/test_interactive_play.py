"""Exercises interactive_play.py's own routing logic (_apply_command_result,
_autonomous_action) against the real SC2FightEnv over a stubbed SC2Env --
same fixture test_env_wrapper.py itself uses -- rather than a hand-rolled
fake, since these functions' whole job is to call real env methods correctly.
"""

import pytest

import sc2rl.command.server as server_module
from sc2rl.command.interpreter import ActionResult, DeclineResult, DirectiveResult, DispatchResult, ReleaseResult
from sc2rl.command.server import EventLog, PendingCommand
from sc2rl.config import EnvConfig
from sc2rl.env.action_space import FixedAction
from sc2rl.inference.interactive_play import _apply_command_result, _autonomous_action, _build_client
from tests.fakes import fake_pysc2 as fake
from tests.test_env_wrapper import make_env


# --- _build_client ---------------------------------------------------------------

def test_claude_client_fails_fast_instead_of_freezing_the_game_loop():
    # Regression test: the SDK's own default (600s read timeout, 2 retries)
    # let one slow/hung API call block the main loop -- and so every
    # env.step() with it -- for minutes, long after server.py's own 30s
    # wait had already given up and reported a timeout to the browser.
    client = _build_client()
    assert client.timeout == pytest.approx(12.0)
    assert client.max_retries == 0
    # The whole point: even a client that maxes out its own timeout once
    # (no retries) must still come back well inside the console's request
    # timeout, so interpret_command's exception handler gets a chance to
    # return a DeclineResult before the browser's own wait expires.
    assert client.timeout < server_module._COMMAND_TIMEOUT_SECONDS


class FakeModel:
    """Returns its canned responses in call order -- interactive_play.py
    always calls predict() unconstrained-then-constrained when a directive
    is active, so the test controls exactly what each call returns."""

    def __init__(self, responses):
        self._responses = list(responses)

    def predict(self, obs, action_masks=None, deterministic=True):
        return self._responses.pop(0), None


def _pending(text: str = "cmd") -> PendingCommand:
    return PendingCommand(text=text)


# --- _apply_command_result -----------------------------------------------------

def test_apply_command_result_action():
    env, _ = make_env([fake.make_timestep(minerals=0, food_cap=15)], EnvConfig())
    env.reset()
    events = EventLog()
    pending = _pending()
    result = ActionResult(action_index=2, action_name="build_barracks", message="Building it.")

    action, extra_calls = _apply_command_result(env, result, pending, events)

    assert action == 2
    assert extra_calls is None
    assert pending.result == {"kind": "action", "action_name": "build_barracks", "message": "Building it."}
    assert events.since(0) == []


def test_apply_command_result_dispatch_assigns_units_and_logs_event():
    ts = fake.make_timestep(
        units=[fake.command_center(1, x=8, y=8), fake.marine(10, x=8, y=8)], minerals=0, food_cap=15,
    )
    env, _ = make_env([ts, ts], EnvConfig())
    env.reset()
    events = EventLog()
    pending = _pending()
    result = DispatchResult(unit_count=1, target_sector=5, message="Sending a marine.")

    action, extra_calls = _apply_command_result(env, result, pending, events)

    assert action == int(FixedAction.NO_OP)
    assert extra_calls  # the raw Attack_pt order(s) to actually send
    assert env.player_controlled_tags == frozenset({10})
    assert pending.result == {"kind": "dispatch", "message": "Sending a marine."}
    assert [e["message"] for e in events.since(0)] == ["Sending a marine."]


def test_apply_command_result_release_returns_no_action_and_logs():
    ts = fake.make_timestep(
        units=[fake.command_center(1, x=8, y=8), fake.marine(10, x=8, y=8)], minerals=0, food_cap=15,
    )
    env, _ = make_env([ts, ts], EnvConfig())
    env.reset()
    env.assign_to_player_control(target_sector=0, tags={10})
    events = EventLog()
    pending = _pending()
    result = ReleaseResult(sector=None, most_recent=False, message="Releasing everyone.")

    action, extra_calls = _apply_command_result(env, result, pending, events)

    assert action is None
    assert extra_calls is None
    assert env.player_controlled_tags == frozenset()
    assert pending.result == {"kind": "release", "released": [10], "message": "Releasing everyone."}
    assert [e["message"] for e in events.since(0)] == ["Releasing everyone."]


def test_apply_command_result_directive_ban_then_allow():
    env, _ = make_env([fake.make_timestep(minerals=0, food_cap=15)], EnvConfig())
    env.reset()
    events = EventLog()

    action, extra_calls = _apply_command_result(
        env, DirectiveResult(action_name="build_supply_depot", allowed=False, message="Banning it."),
        _pending(), events,
    )
    assert action is None and extra_calls is None
    assert env.banned_actions == frozenset({"build_supply_depot"})

    _apply_command_result(
        env, DirectiveResult(action_name="build_supply_depot", allowed=True, message="Allowing it again."),
        _pending(), events,
    )
    assert env.banned_actions == frozenset()
    assert [e["message"] for e in events.since(0)] == ["Banning it.", "Allowing it again."]


def test_apply_command_result_decline_is_never_broadcast_as_an_event():
    # A decline is private feedback to whoever sent the command, not
    # something everyone else watching the console needs to see.
    env, _ = make_env([fake.make_timestep(minerals=0, food_cap=15)], EnvConfig())
    env.reset()
    events = EventLog()
    pending = _pending()

    action, extra_calls = _apply_command_result(env, DeclineResult(message="Can't do that."), pending, events)

    assert action is None and extra_calls is None
    assert pending.result == {"kind": "decline", "message": "Can't do that."}
    assert events.since(0) == []


# --- _autonomous_action ---------------------------------------------------------

def test_autonomous_action_predicts_once_with_no_directives():
    env, _ = make_env([fake.make_timestep(minerals=0, food_cap=15)], EnvConfig())
    env.reset()
    events = EventLog()
    model = FakeModel([3])

    action = _autonomous_action(
        env, model, obs=None, action_masks=env.action_masks(), deterministic=True, events=events,
    )

    assert action == 3
    assert model._responses == []  # exactly one predict() call consumed -- zero overhead when unbanned
    assert events.since(0) == []


def test_autonomous_action_logs_a_denial_when_the_ban_changes_the_choice():
    ts = fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=500, food_cap=15)
    env, _ = make_env([ts, ts], EnvConfig())
    env.reset()
    env.ban_action("build_supply_depot")
    events = EventLog()
    # First predict() call is unconstrained (wants the now-banned action),
    # second is against the ban-narrowed mask (falls back to no_op).
    model = FakeModel([int(FixedAction.BUILD_SUPPLY_DEPOT), int(FixedAction.NO_OP)])

    action = _autonomous_action(
        env, model, obs=None, action_masks=env.action_masks(), deterministic=True, events=events,
    )

    assert action == int(FixedAction.NO_OP)
    messages = [e["message"] for e in events.since(0)]
    assert messages == ["Wanted to build_supply_depot, but that's currently banned by a player directive."]


def test_autonomous_action_stays_silent_when_the_ban_does_not_change_the_choice():
    ts = fake.make_timestep(units=[fake.scv(1, x=5, y=5)], minerals=500, food_cap=15)
    env, _ = make_env([ts, ts], EnvConfig())
    env.reset()
    env.ban_action("build_barracks")  # not what the model wants either way
    events = EventLog()
    model = FakeModel([int(FixedAction.NO_OP), int(FixedAction.NO_OP)])

    action = _autonomous_action(
        env, model, obs=None, action_masks=env.action_masks(), deterministic=True, events=events,
    )

    assert action == int(FixedAction.NO_OP)
    assert events.since(0) == []
