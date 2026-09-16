"""Local web dialog for human commands: a tiny Flask app running in a
background thread alongside the main SC2 game loop.

Two queues, both drained by the main loop, never by Flask itself -- pysc2's
SC2Env is not safe to step from multiple threads, so every queue here is
just a thread-safe handoff, and the actual env.step()/bookkeeping calls all
happen on the one main thread:

- `command_queue` (`POST /command`): text that needs Claude to interpret.
  Blocks the HTTP response (via a per-request threading.Event) until the
  main loop has processed it, since the browser wants to show what actually
  happened, not just "queued."
- `control_queue` (`POST /units/release`): a button click, not a sentence --
  no LLM round trip, so no reason to share the Claude-bound queue's latency
  or its 30s timeout. Same Event-based blocking, but near-instant.

`EventLog` is a small append-only, bounded log both Flask's `GET /events`
route and the main loop write to/read from -- denials, dispatches, releases,
and directive changes all land here so the page can show them live via
polling instead of needing SSE/WebSockets for what's a single local user.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from itertools import count

from flask import Flask, jsonify, render_template, request

_COMMAND_TIMEOUT_SECONDS = 30.0
_CONTROL_TIMEOUT_SECONDS = 5.0
_EVENT_LOG_CAPACITY = 200


@dataclass
class PendingCommand:
    text: str
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None  # set by the game loop before calling done.set()


@dataclass
class PendingRelease:
    """A direct (non-LLM) 'release these units' request from a UI button.
    `tags`/`sector`/`most_recent` mirror interpreter.ReleaseResult's scope --
    exactly one set, or none of them for "release all"."""
    tags: list[int] | None = None
    sector: int | None = None
    most_recent: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class EventLog:
    """Thread-safe, bounded, append-only log with monotonic ids so the page
    can poll `GET /events?since=<id>` and only ever receive what's new."""

    def __init__(self, capacity: int = _EVENT_LOG_CAPACITY):
        self._lock = threading.Lock()
        self._entries: deque[dict] = deque(maxlen=capacity)
        self._next_id = count(1)

    def append(self, kind: str, message: str) -> None:
        with self._lock:
            self._entries.append({"id": next(self._next_id), "kind": kind, "message": message})

    def since(self, after_id: int) -> list[dict]:
        with self._lock:
            return [e for e in self._entries if e["id"] > after_id]


def create_app(
    command_queue: "queue.Queue[PendingCommand]",
    control_queue: "queue.Queue[PendingRelease]",
    events: EventLog,
    state_snapshot,
) -> Flask:
    """`state_snapshot` is a zero-arg callable returning the current
    `{"player_controlled_units": [...], "banned_actions": [...]}` dict for
    `GET /state` -- a callable (not a fixed dict) so the page always reads
    live state off the main loop's thread-owned objects at request time."""
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

    @app.post("/units/release")
    def release_units():
        body = request.get_json(silent=True) or {}
        pending = PendingRelease(tags=body.get("tags"), sector=body.get("sector"))
        control_queue.put(pending)
        if not pending.done.wait(timeout=_CONTROL_TIMEOUT_SECONDS):
            return jsonify({"error": "timed out waiting for the game loop"}), 504
        return jsonify(pending.result)

    @app.get("/state")
    def state():
        return jsonify(state_snapshot())

    @app.get("/events")
    def get_events():
        since = request.args.get("since", 0, type=int)
        return jsonify({"events": events.since(since)})

    return app


def run_server(
    command_queue: "queue.Queue[PendingCommand]",
    control_queue: "queue.Queue[PendingRelease]",
    events: EventLog,
    state_snapshot,
    port: int,
) -> threading.Thread:
    app = create_app(command_queue, control_queue, events, state_snapshot)
    thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    thread.start()
    return thread
