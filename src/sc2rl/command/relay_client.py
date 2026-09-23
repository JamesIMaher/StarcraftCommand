"""Connects the local game loop to the command console hosted as a Kamiwaza
app (kamiwaza-app/ in this repo), in place of the local Flask server.

The Kamiwaza pod can't reach this machine, so everything here is outbound:

- a command thread long-polls `GET /api/game/next` for what the player typed
  or said in the browser, puts it on the SAME `command_queue` /
  `control_queue` the local Flask console uses (server.PendingCommand /
  PendingRelease), waits for the main loop to process it exactly as before,
  and posts the result back to `POST /api/game/result`;
- a state thread posts the console snapshot plus any new EventLog entries to
  `POST /api/game/state` once a second, which doubles as the heartbeat the
  page uses to show whether a game is connected.

The main loop is untouched: it still drains the two queues on its own
thread, so env.step() stays single-threaded.
"""

from __future__ import annotations

import queue
import threading
import time

import httpx

from .server import EventLog, PendingCommand, PendingRelease, _COMMAND_TIMEOUT_SECONDS, _CONTROL_TIMEOUT_SECONDS

_POLL_WAIT_SECONDS = 15.0  # the app caps its long poll at this
_STATE_INTERVAL_SECONDS = 1.0
_RETRY_SECONDS = 3.0


class RelayClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        command_queue: "queue.Queue[PendingCommand]",
        control_queue: "queue.Queue[PendingRelease]",
        events: EventLog,
        state_snapshot,
        verify: bool = True,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._verify = verify
        self._command_queue = command_queue
        self._control_queue = control_queue
        self._events = events
        self._state_snapshot = state_snapshot
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for target in (self._command_loop, self._state_loop):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(base_url=self._base_url, headers=self._headers, verify=self._verify, timeout=timeout)

    def handle(self, item: dict) -> dict:
        """Hand one relayed request to the main loop and wait for its result,
        exactly like the local Flask routes do."""
        if item.get("kind") == "release":
            pending = PendingRelease(tags=item.get("tags"), sector=item.get("sector"))
            self._control_queue.put(pending)
            timeout = _CONTROL_TIMEOUT_SECONDS
        else:
            pending = PendingCommand(text=str(item.get("text", "")))
            self._command_queue.put(pending)
            timeout = _COMMAND_TIMEOUT_SECONDS
        if not pending.done.wait(timeout=timeout):
            return {"error": "timed out waiting for the game loop"}
        return pending.result or {}

    def _command_loop(self) -> None:
        warned = False
        with self._client(timeout=_POLL_WAIT_SECONDS + 15.0) as client:
            while not self._stop.is_set():
                try:
                    response = client.get("/api/game/next", params={"wait": _POLL_WAIT_SECONDS})
                    if response.status_code == 204:
                        warned = False
                        continue
                    response.raise_for_status()
                    warned = False
                    item = response.json()
                    result = self.handle(item)
                    client.post("/api/game/result", json={"id": item["id"], "result": result}).raise_for_status()
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    if not warned:  # once per outage, not once per retry
                        print(f"[relay] command channel error, retrying: {exc}")
                        warned = True
                    self._stop.wait(_RETRY_SECONDS)

    def _state_loop(self) -> None:
        last_event_id = 0
        warned = False
        with self._client(timeout=10.0) as client:
            while not self._stop.is_set():
                started = time.monotonic()
                new_events = self._events.since(last_event_id)
                try:
                    client.post("/api/game/state", json={
                        "snapshot": self._state_snapshot(),
                        "events": [{"kind": e["kind"], "message": e["message"]} for e in new_events],
                    }).raise_for_status()
                    if new_events:
                        last_event_id = new_events[-1]["id"]
                    if warned:
                        print("[relay] reconnected to the Kamiwaza console")
                    warned = False
                except httpx.HTTPError as exc:
                    if not warned:
                        print(f"[relay] state channel error, retrying: {exc}")
                        warned = True
                self._stop.wait(max(0.0, _STATE_INTERVAL_SECONDS - (time.monotonic() - started)))
