"""Exercises the Flask endpoints' queue/event handoffs without any live game
or SC2FightEnv involved -- a fake "game loop" thread pulls from the same
queues the real one would and fulfils requests directly.
"""

from __future__ import annotations

import queue
import threading

import sc2rl.command.server as server_module
from sc2rl.command.server import EventLog, create_app


def make_app(state_snapshot=None):
    command_queue: "queue.Queue" = queue.Queue()
    control_queue: "queue.Queue" = queue.Queue()
    events = EventLog()
    app = create_app(command_queue, control_queue, events, state_snapshot or (lambda: {}))
    return app, command_queue, control_queue, events


# --- POST /command -------------------------------------------------------------

def test_command_endpoint_returns_what_the_game_loop_reports():
    app, command_queue, _, _ = make_app()
    client = app.test_client()

    def fake_game_loop():
        pending = command_queue.get(timeout=2)
        assert pending.text == "attack sector 1"
        pending.result = {"kind": "action", "action_name": "move_army_to_sector_1", "message": "On it."}
        pending.done.set()

    threading.Thread(target=fake_game_loop).start()
    response = client.post("/command", json={"text": "attack sector 1"})

    assert response.status_code == 200
    assert response.get_json()["message"] == "On it."


def test_command_endpoint_rejects_empty_text_without_touching_the_queue():
    app, command_queue, _, _ = make_app()
    client = app.test_client()

    response = client.post("/command", json={"text": "   "})

    assert response.status_code == 400
    assert command_queue.empty()


def test_command_endpoint_times_out_if_nothing_processes_it():
    app, _, _, _ = make_app()
    client = app.test_client()

    original = server_module._COMMAND_TIMEOUT_SECONDS
    server_module._COMMAND_TIMEOUT_SECONDS = 0.05
    try:
        response = client.post("/command", json={"text": "hello"})
    finally:
        server_module._COMMAND_TIMEOUT_SECONDS = original

    assert response.status_code == 504


# --- POST /units/release --------------------------------------------------------

def test_release_endpoint_uses_the_control_queue_not_the_command_queue():
    app, command_queue, control_queue, _ = make_app()
    client = app.test_client()

    def fake_game_loop():
        pending = control_queue.get(timeout=2)
        assert pending.tags == [10, 11]
        pending.result = {"released": [10, 11]}
        pending.done.set()

    threading.Thread(target=fake_game_loop).start()
    response = client.post("/units/release", json={"tags": [10, 11]})

    assert response.status_code == 200
    assert response.get_json() == {"released": [10, 11]}
    assert command_queue.empty()


def test_release_endpoint_release_all_sends_no_scope_fields():
    app, _, control_queue, _ = make_app()
    client = app.test_client()

    def fake_game_loop():
        pending = control_queue.get(timeout=2)
        assert pending.tags is None
        assert pending.sector is None
        pending.result = {"released": [10, 11]}
        pending.done.set()

    threading.Thread(target=fake_game_loop).start()
    response = client.post("/units/release", json={})

    assert response.status_code == 200


def test_release_endpoint_times_out_quickly_if_nothing_processes_it():
    # Deliberately a much shorter timeout than /command -- a button click
    # should never make the player wait anywhere near 30s.
    app, _, _, _ = make_app()
    client = app.test_client()

    original = server_module._CONTROL_TIMEOUT_SECONDS
    server_module._CONTROL_TIMEOUT_SECONDS = 0.05
    try:
        response = client.post("/units/release", json={})
    finally:
        server_module._CONTROL_TIMEOUT_SECONDS = original

    assert response.status_code == 504


# --- GET /state ------------------------------------------------------------------

def test_state_endpoint_returns_the_live_snapshot():
    snapshot = {"player_controlled_units": [10, 11], "banned_actions": ["build_supply_depot"]}
    app, _, _, _ = make_app(state_snapshot=lambda: snapshot)
    client = app.test_client()

    response = client.get("/state")

    assert response.get_json() == snapshot


# --- GET /events -----------------------------------------------------------------

def test_events_endpoint_returns_only_entries_after_the_given_cursor():
    app, _, _, events = make_app()
    client = app.test_client()
    events.append("denied", "wanted to build_supply_depot, but that's banned")
    events.append("dispatch", "sent 1 marine to sector 5")

    all_events = client.get("/events").get_json()["events"]
    assert len(all_events) == 2

    since_first = client.get(f"/events?since={all_events[0]['id']}").get_json()["events"]
    assert len(since_first) == 1
    assert since_first[0]["message"] == "sent 1 marine to sector 5"


def test_event_log_is_bounded():
    events = EventLog(capacity=3)
    for i in range(5):
        events.append("info", f"event {i}")
    remaining = events.since(0)
    assert len(remaining) == 3
    assert [e["message"] for e in remaining] == ["event 2", "event 3", "event 4"]


# --- GET / -------------------------------------------------------------------------

def test_index_serves_the_command_page():
    app, _, _, _ = make_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Command Console" in response.data
