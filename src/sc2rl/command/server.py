"""Local web dialog for human commands: a tiny Flask app running in a
background thread alongside the main SC2 game loop. `POST /command` hands
the text off to the game loop via a queue and blocks until it's processed,
so the browser gets back exactly what happened (or why it didn't).

The game loop and the Flask request thread are different threads -- pysc2's
SC2Env is not safe to step from multiple threads, so the queue + per-request
threading.Event handoff is what keeps all `env.step()` calls on the one main
thread while still letting the HTTP handler wait synchronously for a result.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

from flask import Flask, jsonify, render_template, request

_COMMAND_TIMEOUT_SECONDS = 30.0


@dataclass
class PendingCommand:
    text: str
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None  # set by the game loop before calling done.set()


def create_app(command_queue: "queue.Queue[PendingCommand]") -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("command.html")

    @app.post("/command")
    def command():
        text = (request.get_json(silent=True) or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "empty command"}), 400
        pending = PendingCommand(text=text)
        command_queue.put(pending)
        if not pending.done.wait(timeout=_COMMAND_TIMEOUT_SECONDS):
            return jsonify({"error": "timed out waiting for the game loop"}), 504
        return jsonify(pending.result)

    return app


def run_server(command_queue: "queue.Queue[PendingCommand]", port: int) -> threading.Thread:
    app = create_app(command_queue)
    thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread
