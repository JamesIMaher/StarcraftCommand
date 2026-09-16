"""Exercises the Flask command endpoint's queue/event handoff without any
live game or SC2FightEnv involved -- a fake "game loop" thread pulls from
the same queue the real one would and fulfils the request directly.
"""

from __future__ import annotations

import queue
import threading

import sc2rl.command.server as server_module
from sc2rl.command.server import create_app


def test_command_endpoint_returns_what_the_game_loop_reports():
    command_queue: "queue.Queue" = queue.Queue()
    app = create_app(command_queue)
    client = app.test_client()

    def fake_game_loop():
        pending = command_queue.get(timeout=2)
        assert pending.text == "attack sector 1"
        pending.result = {"action_index": 5, "action_name": "move_army_to_sector_1", "message": "On it."}
        pending.done.set()

    threading.Thread(target=fake_game_loop).start()

    response = client.post("/command", json={"text": "attack sector 1"})

    assert response.status_code == 200
    assert response.get_json() == {"action_index": 5, "action_name": "move_army_to_sector_1", "message": "On it."}


def test_command_endpoint_reports_a_decline():
    command_queue: "queue.Queue" = queue.Queue()
    app = create_app(command_queue)
    client = app.test_client()

    def fake_game_loop():
        pending = command_queue.get(timeout=2)
        pending.result = {"action_index": None, "action_name": "cannot_comply", "message": "Can't do that."}
        pending.done.set()

    threading.Thread(target=fake_game_loop).start()

    response = client.post("/command", json={"text": "do a backflip"})

    assert response.status_code == 200
    assert response.get_json()["action_index"] is None


def test_command_endpoint_rejects_empty_text_without_touching_the_queue():
    command_queue: "queue.Queue" = queue.Queue()
    app = create_app(command_queue)
    client = app.test_client()

    response = client.post("/command", json={"text": "   "})

    assert response.status_code == 400
    assert command_queue.empty()


def test_command_endpoint_times_out_if_nothing_processes_it():
    command_queue: "queue.Queue" = queue.Queue()
    app = create_app(command_queue)
    client = app.test_client()

    original_timeout = server_module._COMMAND_TIMEOUT_SECONDS
    server_module._COMMAND_TIMEOUT_SECONDS = 0.05
    try:
        response = client.post("/command", json={"text": "hello"})
    finally:
        server_module._COMMAND_TIMEOUT_SECONDS = original_timeout

    assert response.status_code == 504


def test_index_serves_the_command_page():
    command_queue: "queue.Queue" = queue.Queue()
    app = create_app(command_queue)
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Command Console" in response.data
