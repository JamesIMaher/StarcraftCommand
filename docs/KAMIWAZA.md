# Running the command console on Kamiwaza

The [command console](COMMAND_CONSOLE.md) can be hosted as an app in the
Kamiwaza App Garden instead of on `127.0.0.1`. Commands are then interpreted
by the platform's own LLM (gpt-oss-120b) instead of Claude. StarCraft II, the
PySC2 environment and the trained policy still run on your game PC, so you
still watch the game there.

## How it fits together

```
 your game PC                                   Kamiwaza (srt.gaifederal.net)
┌───────────────────────────────┐              ┌────────────────────────────────────┐
│ StarCraft II + SC2FightEnv    │              │ sc2-command app (kamiwaza-app/)    │
│ MaskablePPO policy            │  long poll   │                                    │
│ interactive_play main loop  ◀─┼── commands ──┤  POST /api/command   ◀── browser   │
│   (unchanged: drains the same │── results ──▶│  (page, mic, unit list, log)       │
│    command/control queues)    │── state ────▶│                                    │
│ interpreter.py ───────────────┼── /v1 ──────▶│  LLM proxy ──▶ gpt-oss-120b (vLLM) │
└───────────────────────────────┘   (HTTPS,    └────────────────────────────────────┘
                                   PAT auth)
```

- **Every connection starts on the game PC.** The pod can't reach your PC, so
  the game connects to the app instead (`src/sc2rl/command/relay_client.py`).
  One thread long-polls for commands; another posts the unit/directive
  snapshot and new log events once a second, which also tells the page a game
  is connected.
- **The main game loop is unchanged.** Relayed commands go on the same
  queues the local Flask console used, so `env.step()` still runs on one
  thread and a command still pauses the policy for exactly one step.
- **Interpreting a command is unchanged too.** `interpreter.py` accepts
  either an Anthropic client or an OpenAI-compatible one. Same five tools,
  same enum-violation guard. The OpenAI client points at the app's
  `/v1/chat/completions` proxy. The proxy pins the model to `MODEL_NAME` and
  forwards to the in-cluster vLLM service, so the game PC needs no model
  deployment id.
- **Each user gets their own channel**, keyed by Kamiwaza user id. Your
  browser session and your game's personal access token (PAT) are the same
  user, so they meet in the same channel.
- gpt-oss-120b stays loaded: the app only calls the existing deployment.
  Measured on the live instance: about 1–1.4 s per interpretation, about 2–2.6 s
  from the browser round trip.

## Running a game against it

1. Create a personal access token in the Kamiwaza UI, or with
   `POST /api/auth/pats` and `{"name": "sc2", "scope": "read", "ttl_seconds": 2592000}`.
   A `read` scope is enough: the token only has to get through the platform
   ingress to this app.
2. On the game PC, put it in `.env` (see `.env.example`):
   ```
   KAMIWAZA_API_KEY=<the token>
   KAMIWAZA_VERIFY_SSL=false   # srt.gaifederal.net uses a self-signed certificate
   ```
3. Start the game:
   ```powershell
   python -m sc2rl.inference.interactive_play --checkpoint checkpoints/final_model `
       --kamiwaza-url https://srt.gaifederal.net/runtime/apps/sc2-command
   ```
4. Open `https://srt.gaifederal.net/runtime/apps/sc2-command/` in Chrome or
   Edge, logged in as the same user that owns the token. The badge next to
   the title turns green ("game connected") within a second or two.

Without `--kamiwaza-url`, everything behaves exactly as before: local console,
Claude, `ANTHROPIC_API_KEY`.

## Voice

The mic button uses the browser's built-in Web Speech API, same as the local
console. In Chrome and Edge, that sends the audio to Google or Microsoft for
transcription. The audio never passes through Kamiwaza. That works on a
network with internet access, but not on an air-gapped one, and it means
voice leaves the platform's boundary. Keeping it inside Kamiwaza means
transcribing in the pod, e.g. `faster-whisper` on CPU. That isn't built yet.

## Deploying and releasing

Everything is in `kamiwaza-app/deploy.sh`. Run it on the Kamiwaza node, where
`localhost:5001` is the registry the extension operator pulls from. It asks
for the Kamiwaza password unless `KAMIWAZA_PASS` is set.

```bash
cd kamiwaza-app
TAG=0.1.1 ./deploy.sh build      # build + push -- always a new tag
TAG=0.1.1 ./deploy.sh patch      # move the running app to it
TAG=0.1.1 ./deploy.sh template   # keep the App Garden catalog entry in step
```

`TAG=... ./deploy.sh create` is only for a first deploy, or after the
extension has been deleted. The app's `KAMIWAZA_BASE_URL` points at the
in-cluster vLLM service of the current gpt-oss-120b deployment
(`vllm-<first 8 chars of its id>`). If the model is redeployed, that id
changes. Run `./deploy.sh model` to get the new URL, then `template` again,
and restart the app from the App Garden.

Tests for the app and relay (real HTTP, fake vLLM, auth off):

```bash
pip install -r kamiwaza-app/requirements.txt
pytest kamiwaza-app/tests
```

## Limits

- **One app replica.** Channels, queues and waiting requests are in memory.
  A pod restart drops them. The game client reconnects on its own, and the page
  shows "no game connected" until it does.
- **One game per user.** A second game started with the same user's token
  would share the channel and take commands meant for the first.
- The `/v1` proxy is open to any authenticated platform user, like the
  platform's own model routes. It only does non-streaming chat completions,
  with `max_tokens` capped at 2000.
