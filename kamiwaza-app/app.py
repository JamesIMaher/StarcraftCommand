"""SC2 Command Console, served from the Kamiwaza App Garden.

The game itself (StarCraft II, the PySC2 env and the trained policy) runs on
the player's own machine -- see `sc2rl.inference.interactive_play
--kamiwaza-url`. The pod can't reach that machine, so this app is a relay the
game connects OUT to:

    browser --POST /api/command--> [this app] <--GET /api/game/next (long poll)-- game
    browser <-----result----------  [this app] <--POST /api/game/result---------- game
    browser --GET /api/state, /api/events--> [this app] <--POST /api/game/state-- game

Plus `POST /v1/chat/completions`, a narrow OpenAI-compatible proxy to the
platform's LLM (gpt-oss-120b on the in-mesh vLLM service), so the game's
command interpreter needs one URL and one token -- this app's -- instead of
tracking the model deployment's own id and route.

Each authenticated user gets their own channel, keyed by platform user id:
the browser (session cookie) and the game client (PAT) authenticate as the
same user, so they meet in the same channel, and two players never see each
other's games. All state is in memory -- run exactly one replica/worker.
"""

import asyncio
import itertools
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from kamiwaza_extensions_lib import Identity, create_session_router, require_auth

# Browser-facing waits mirror the local Flask console (sc2rl.command.server).
# The game side's own interpreter timeout (12s) fits inside COMMAND_TIMEOUT.
COMMAND_TIMEOUT_SECONDS = 30.0
CONTROL_TIMEOUT_SECONDS = 10.0
# Long-poll window for the game's GET /api/game/next. Kept well under typical
# ingress/proxy idle timeouts (often 30-60s).
MAX_POLL_SECONDS = 15.0
# The game client posts state every second; after this long without hearing
# from it, the page shows "no game connected" and commands fail fast.
CONNECTED_WITHIN_SECONDS = 10.0
EVENT_LOG_CAPACITY = 200

LLM_BASE_URL = (os.environ.get("KAMIWAZA_BASE_URL") or "").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME") or "gpt-oss-120b"
LLM_MAX_TOKENS_CAP = 2000
LLM_TIMEOUT_SECONDS = 30.0

PAGE = (Path(__file__).parent / "static" / "command.html").read_text()


@dataclass
class _Item:
    """One browser request waiting for the game loop."""
    id: int
    kind: str  # "command" | "release"
    payload: dict
    future: asyncio.Future
    cancelled: bool = False


@dataclass
class _Channel:
    pending: asyncio.Queue = field(default_factory=asyncio.Queue)
    snapshot: dict = field(default_factory=dict)
    events: deque = field(default_factory=lambda: deque(maxlen=EVENT_LOG_CAPACITY))
    in_flight: dict = field(default_factory=dict)  # item id -> _Item handed to the game
    last_seen: float = 0.0

    def connected(self) -> bool:
        return time.monotonic() - self.last_seen < CONNECTED_WITHIN_SECONDS


_channels: dict[str, _Channel] = {}
_item_ids = itertools.count(1)
# Event ids are assigned here, not by the game client, so the page's
# `since` cursor stays monotonic across game restarts.
_event_ids = itertools.count(1)


def _channel(identity: Identity) -> _Channel:
    key = identity.user_id or "anonymous"
    if key not in _channels:
        _channels[key] = _Channel()
    return _channels[key]


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


app = FastAPI(title="SC2 Command Console")
app.include_router(create_session_router())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    # The ingress doesn't strip the app's mount path (see
    # _StripAppPathMiddleware), so the page's own fetch() calls need it.
    app_path = os.environ.get("KAMIWAZA_APP_PATH", "").rstrip("/")
    return PAGE.replace("<head>", f"<head>\n<script>window.__APP_PATH__ = {app_path!r};</script>", 1)


# --- browser-facing ---------------------------------------------------------------

async def _send_to_game(channel: _Channel, kind: str, payload: dict, timeout: float) -> JSONResponse:
    if not channel.connected():
        return _error(503, "No game is connected. Start interactive_play with --kamiwaza-url on your game PC.")
    item = _Item(id=next(_item_ids), kind=kind, payload=payload,
                 future=asyncio.get_running_loop().create_future())
    await channel.pending.put(item)
    try:
        result = await asyncio.wait_for(asyncio.shield(item.future), timeout)
    except asyncio.TimeoutError:
        item.cancelled = True  # the game skips it if it hasn't picked it up yet
        return _error(504, "Timed out waiting for the game loop.")
    if "error" in result:
        return _error(504, result["error"])
    return JSONResponse(result)


@app.post("/api/command")
async def command(request: Request, identity: Identity = Depends(require_auth)):
    body = await request.json()
    text = str(body.get("text", "")).strip()
    if not text:
        return _error(400, "empty command")
    return await _send_to_game(_channel(identity), "command", {"text": text}, COMMAND_TIMEOUT_SECONDS)


@app.post("/api/units/release")
async def release_units(request: Request, identity: Identity = Depends(require_auth)):
    body = await request.json()
    payload = {"tags": body.get("tags"), "sector": body.get("sector")}
    return await _send_to_game(_channel(identity), "release", payload, CONTROL_TIMEOUT_SECONDS)


@app.get("/api/state")
def state(identity: Identity = Depends(require_auth)):
    channel = _channel(identity)
    connected = channel.connected()
    return {"connected": connected, **(channel.snapshot if connected else {})}


@app.get("/api/events")
def events(since: int = 0, identity: Identity = Depends(require_auth)):
    return {"events": [e for e in _channel(identity).events if e["id"] > since]}


# --- game-facing ------------------------------------------------------------------

@app.get("/api/game/next")
async def game_next(wait: float = MAX_POLL_SECONDS, identity: Identity = Depends(require_auth)):
    """Long poll: the next browser request for this user's game, or 204."""
    channel = _channel(identity)
    channel.last_seen = time.monotonic()
    deadline = time.monotonic() + max(0.0, min(wait, MAX_POLL_SECONDS))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Response(status_code=204)
        try:
            item = await asyncio.wait_for(channel.pending.get(), remaining)
        except asyncio.TimeoutError:
            return Response(status_code=204)
        if not item.cancelled:
            channel.in_flight[item.id] = item
            return {"id": item.id, "kind": item.kind, **item.payload}


@app.post("/api/game/result")
async def game_result(request: Request, identity: Identity = Depends(require_auth)):
    body = await request.json()
    item = _channel(identity).in_flight.pop(int(body.get("id", 0)), None)
    if item is not None and not item.future.done():
        item.future.set_result(body.get("result") or {"error": "The game returned no result."})
    return {"ok": True}


@app.post("/api/game/state")
async def game_state(request: Request, identity: Identity = Depends(require_auth)):
    """Heartbeat from the game: the console snapshot plus any new events."""
    body = await request.json()
    channel = _channel(identity)
    channel.last_seen = time.monotonic()
    channel.snapshot = body.get("snapshot") or {}
    for event in body.get("events") or []:
        channel.events.append({"id": next(_event_ids), "kind": str(event.get("kind", "")),
                               "message": str(event.get("message", ""))})
    return {"ok": True}


# --- LLM proxy ----------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, identity: Identity = Depends(require_auth)):
    """OpenAI-compatible, non-streaming, pinned to MODEL_NAME. The in-mesh
    vLLM service needs no auth; the caller already authenticated to us."""
    if not LLM_BASE_URL:
        return _error(503, "KAMIWAZA_BASE_URL is not configured.")
    body = await request.json()
    if body.get("stream"):
        return _error(400, "streaming is not supported by this proxy")
    body["model"] = MODEL_NAME
    body["max_tokens"] = min(int(body.get("max_tokens") or LLM_MAX_TOKENS_CAP), LLM_MAX_TOKENS_CAP)
    body.pop("max_completion_tokens", None)
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            upstream = await client.post(f"{LLM_BASE_URL}/chat/completions", json=body)
    except httpx.HTTPError as exc:
        return _error(502, f"LLM backend unreachable: {exc}")
    return Response(upstream.content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"))


class _StripAppPathMiddleware:
    """The platform ingress forwards requests at their full external path
    (/runtime/apps/<name>/...) and hands the prefix over as
    KAMIWAZA_APP_PATH. Strip it so routes match; a no-op for health probes
    and local runs, which hit the container unprefixed."""

    def __init__(self, app, prefix):
        self.app = app
        self.prefix = prefix.rstrip("/")

    async def __call__(self, scope, receive, send):
        if self.prefix and scope["type"] == "http" and scope["path"].startswith(self.prefix):
            scope = dict(scope)
            scope["path"] = scope["path"][len(self.prefix):] or "/"
            scope["root_path"] = self.prefix
        await self.app(scope, receive, send)


_app_path = os.environ.get("KAMIWAZA_APP_PATH", "")
if _app_path:
    app = _StripAppPathMiddleware(app, _app_path)
