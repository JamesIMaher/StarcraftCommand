"""End-to-end over real HTTP: the console app under uvicorn, the game-side
RelayClient (src/sc2rl/command/relay_client.py) connected to it, a stand-in
for interactive_play's main loop draining the queues, and a fake vLLM behind
the app's /v1 proxy. Auth is off (KAMIWAZA_USE_AUTH=false), so every caller
shares the anonymous user's channel -- same as one real user with a browser
session and a PAT.

Run from the repo root:  pytest kamiwaza-app/tests
"""

import importlib
import json
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI, Request

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parent
sys.path[:0] = [str(APP_DIR), str(REPO_ROOT / "src")]

from sc2rl.command.interpreter import DispatchResult, interpret_command  # noqa: E402
from sc2rl.command.relay_client import RelayClient  # noqa: E402
from sc2rl.command.server import EventLog  # noqa: E402
from sc2rl.env.action_space import ActionSpaceSpec  # noqa: E402
from sc2rl.env.game_state import GameState  # noqa: E402
from sc2rl.env.sector_grid import SectorGrid, SpawnOrientation  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app) -> str:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


upstream_requests: list[dict] = []


def _fake_vllm() -> FastAPI:
    fake = FastAPI()

    @fake.post("/v1/chat/completions")
    async def completions(request: Request):
        body = await request.json()
        upstream_requests.append(body)
        args = {"unit_count": "2", "target_sector": 3, "message": "Sending two marines."}
        return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "dispatch_units", "arguments": json.dumps(args)}}],
        }}]}

    return fake


@pytest.fixture(scope="module")
def app_url(tmp_path_factory):
    llm_url = _serve(_fake_vllm())
    mp = pytest.MonkeyPatch()
    mp.setenv("KAMIWAZA_USE_AUTH", "false")
    mp.setenv("KAMIWAZA_BASE_URL", f"{llm_url}/v1")
    mp.setenv("MODEL_NAME", "gpt-oss-120b")
    mp.delenv("KAMIWAZA_APP_PATH", raising=False)
    module = importlib.import_module("app")
    importlib.reload(module)  # read the env above, whatever imported it first
    yield _serve(module.app)
    mp.undo()


def test_commands_fail_fast_when_no_game_is_connected(app_url):
    response = httpx.post(f"{app_url}/api/command", json={"text": "build a depot"})
    assert response.status_code == 503
    assert "No game is connected" in response.json()["error"]
    assert httpx.get(f"{app_url}/api/state").json() == {"connected": False}


def test_page_is_served_with_the_app_path_injected(app_url):
    page = httpx.get(f"{app_url}/").text
    assert "window.__APP_PATH__ = ''" in page
    assert "API_BASE + '/api/command'" in page


def test_browser_command_round_trips_through_the_game_loop(app_url):
    command_queue, control_queue, events = queue.Queue(), queue.Queue(), EventLog()
    snapshot = {"player_controlled_units": [7], "banned_actions": ["build_barracks"]}
    relay = RelayClient(app_url, "unused-with-auth-off", command_queue, control_queue, events, lambda: snapshot)

    def main_loop():  # what interactive_play.play() does with each queue
        while True:
            try:
                pending = command_queue.get(timeout=0.05)
                pending.result = {"kind": "action", "action_name": "train_marine", "message": f"ok: {pending.text}"}
                events.append("dispatch", "Two marines headed out.")
                pending.done.set()
            except queue.Empty:
                pass
            try:
                release = control_queue.get_nowait()
                release.result = {"released": release.tags or []}
                release.done.set()
            except queue.Empty:
                pass

    threading.Thread(target=main_loop, daemon=True).start()
    relay.start()
    try:
        deadline = time.time() + 5
        while not httpx.get(f"{app_url}/api/state").json()["connected"]:
            assert time.time() < deadline, "game never showed as connected"
            time.sleep(0.1)

        state = httpx.get(f"{app_url}/api/state").json()
        assert state == {"connected": True, **snapshot}

        response = httpx.post(f"{app_url}/api/command", json={"text": "make a marine"}, timeout=35)
        assert response.status_code == 200
        assert response.json() == {"kind": "action", "action_name": "train_marine", "message": "ok: make a marine"}

        response = httpx.post(f"{app_url}/api/units/release", json={"tags": [7]}, timeout=15)
        assert response.json() == {"released": [7]}

        deadline = time.time() + 5
        while True:
            logged = httpx.get(f"{app_url}/api/events", params={"since": 0}).json()["events"]
            if logged:
                break
            assert time.time() < deadline, "event never reached the page"
            time.sleep(0.1)
        assert logged[0]["kind"] == "dispatch"
        assert logged[0]["message"] == "Two marines headed out."
        later = httpx.get(f"{app_url}/api/events", params={"since": logged[-1]["id"]}).json()["events"]
        assert later == []
    finally:
        relay.stop()


def test_interpreter_reaches_the_llm_through_the_app_proxy(app_url):
    from sc2rl.inference.interactive_play import _build_openai_client

    upstream_requests.clear()
    client = _build_openai_client(f"{app_url}/v1", "unused-with-auth-off")
    spec = ActionSpaceSpec(grid=SectorGrid(map_size=64, cols=4, rows=4))
    state = GameState(game_loop=0, minerals=0, food_used=0, food_cap=15)
    orientation = SpawnOrientation(map_size=64, mirror_x=False, mirror_y=False)

    result = interpret_command(
        client, "send two marines to sector 3", state, spec, np.ones(spec.num_actions, dtype=bool),
        orientation, model="whatever-the-client-asks-for",
    )

    assert result == DispatchResult(unit_count=2, target_sector=3, message="Sending two marines.")
    sent = upstream_requests[0]
    assert sent["model"] == "gpt-oss-120b"  # pinned by the app, not the caller
    assert sent["tool_choice"] == "required"
    assert sent["max_tokens"] <= 2000


def test_proxy_rejects_streaming(app_url):
    response = httpx.post(f"{app_url}/v1/chat/completions", json={"stream": True, "messages": []})
    assert response.status_code == 400
