# Live command console (Generative-AI layer)

`interactive_play.py` runs the same trained policy as `play.py` (see
[TRAINING.md](TRAINING.md#after-training)) but adds a local web page where
you can type -- or speak, via the browser's built-in speech recognition
(Chrome/Edge) -- an instruction like "return the marines to base" or "build
a supply depot."

```powershell
copy .env.example .env
notepad .env    # paste your key in place of sk-ant-...
python -m sc2rl.inference.interactive_play --checkpoint checkpoints/final_model --visualize
```

## Where the API key lives

`interactive_play.py` calls `load_dotenv()` on startup, which reads a local
`.env` file (gitignored, never committed) into the process environment --
copy `.env.example` to `.env` and fill in a real key from
[console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
This is the recommended way on Windows: the key stays scoped to this one
project instead of a machine-wide environment variable every other program
on the PC can also read, and there's nothing to re-set every time you open a
new PowerShell window.

Two alternatives if you'd rather not add a `.env` file:

- **Current session only:** `$env:ANTHROPIC_API_KEY = "sk-ant-..."` --
  forgotten the moment you close that PowerShell window.
- **Persistent, machine-wide:** `setx ANTHROPIC_API_KEY "sk-ant-..."` --
  survives across sessions and reboots (stored in the Windows registry
  under your user account), but takes effect only in *new* terminals opened
  after running it, and every program running as you can read it, not just
  this project. The Environment Variables entry in Windows' System
  Properties dialog does the same thing through the GUI.

## The five tools

Open the printed `http://127.0.0.1:8765` (`--command-port` to change it).
Under the hood, a command is routed to one of **five Claude tools** in a
single call (`command/interpreter.py`, `tool_choice={"type": "any"}` --
forces *a* tool call, lets Claude pick which):

- `choose_action` -- the original v1 behavior: one legal action now, `enum`
  restricted to whatever `env.action_masks()` says is currently legal, plus
  the same ground-truth-backed illegal-choice handling described below.
- `dispatch_units` -- send a specific number of marines (Claude infers the
  count from wording -- "a marine" -> 1, "a few" -> 3, "all" -> every
  unreserved marine) to a named sector, under **persistent player control**
  until released.
- `release_units` -- return player-controlled marines to AI control, scoped
  to "all", the "most recent" dispatch, or whichever ones are currently in a
  named sector (see [Per-unit player control](#per-unit-player-control-eg-send-a-marine-to-the-bottom-right-corner)
  below for why these three scopes specifically).
- `set_directive` -- ban or re-allow a named action for the *autonomous*
  policy for the rest of the episode (see
  [Standing directives](#standing-directives-eg-do-not-build-any-more-supply-depots)
  below).
- `decline` -- anything that doesn't fit the above (replaces v1's inline
  `cannot_comply` enum value).

For `choose_action` specifically, the schema's `enum` is a strong hint, not
a hard guarantee:

- Confirmed live, a fast model can still occasionally name something
  outside this turn's actual enum (e.g. `build_barracks` when no supply
  depot exists yet) -- so `interpret_command()` never trusts the model's
  own explanation for that case.
- An illegal choice always gets `env.step()` blocked (never executes) *and*
  has its message replaced with a ground-truth reason computed straight
  from `GameState` (`interpreter.py`'s `_explain_unavailable()`), never the
  model's original text, which was written assuming the choice would be
  honored and would otherwise read as a false confirmation.
- This reuses `action_space.py`'s existing `name()`/`index_for_name()`
  exactly as its docstring always said a future "human- or LLM-driven
  command layer" would: no changes to the action space, the environment, or
  the trained model were needed for this feature.

## Standing directives (e.g. "do not build any more supply depots")

- v1 had no memory: "do not build any more supply depots" executed as
  nothing (there's no legal action matching a prohibition), and the next
  time the autonomous policy wanted one, it just built it.
- A `set_directive` call now adds the action's name to `env.banned_actions`,
  which persists for the rest of the episode until a later "you can build
  supply depots again" command lifts it (`env.allow_action`).
- **The naive way to enforce a ban** -- let the policy pick freely, then
  veto and substitute `no_op` after the fact -- risks a real soft-lock:
  with `deterministic=True` action selection and a near-static
  observation, a vetoed top choice can get re-picked (and re-vetoed) every
  single step forever, stalling that whole category of play for as long as
  the ban is active.
- **Instead**, a directive is applied as an extra masking layer
  (`env.apply_bans()`, a copy of `action_masks()` with the banned index
  forced `False`) that only ever applies to the *autonomous* policy's own
  selection -- never to a player's explicit one-shot command (if you say
  "no more depots" and then later say "okay, build one now," that one-shot
  command is honored without lifting the standing rule first), and never
  during training or `play.py`. Masked out this way, the policy naturally
  falls through to its next-best *legal* alternative, exactly like every
  other mask condition already works, instead of grinding on a dead choice.
- To still show "the model tried to build a supply depot and was denied" on
  the page, `interactive_play.py`'s `_autonomous_action()` predicts
  **twice** whenever any directive is active: once against the full mask
  (what the policy would have picked) and once against the
  directive-narrowed mask (what actually executes). If the two differ and
  the unconstrained pick names a banned action, a `denied` event is
  appended to the live feed. This is two extra forward passes through a
  2x64 MLP -- free -- and only runs at all while a directive is active, so
  it costs nothing when the feature isn't in use.

## Per-unit player control (e.g. "send a marine to the bottom right corner")

- v1 also had no concept of "this specific marine is under player control
  right now" -- every marine not in the garrison was fair game for the
  autonomous policy's next group-move, so a marine sent to a corner got
  swept right back home on the model's very next move order.
- The fix reuses the **garrison** pattern (already "a persistent set of
  marine tags excluded from the autonomous policy's group-move, managed
  outside the RL action space") for a second pool: `dispatch_units` adds
  the assigned tags to `env._player_controlled_tags` via
  `assign_to_player_control()`, and `step()` passes `garrison_tags |
  player_controlled_tags` as the reserved set to the translator -- both
  pools are equally off-limits to the autonomous group-move.
- A dispatch order rides to the game as raw `Attack_pt` calls merged into
  that turn's `env.step(action, extra_calls=...)`, reusing the exact same
  pathing-aware, known-structure-aware, cliff-avoiding target resolution as
  the autonomous move (`action_translation.py`'s `move_specific()`,
  factored out of `_move_army` for this purpose).
- **Critically**, a player-issued dispatch is a *different action kind*
  than the autonomous policy's group-move, so it does **not** go through
  `min_marines_to_move`/`_advance`/`_continue` -- those thresholds exist to
  gate the trained model's single "move everyone" action, not as a
  physical law that a marine can't move alone. Asking to move fewer
  marines than the model's own threshold (e.g. "move these 2 marines" with
  only 4 at home) just works.

**Sector numbering is home-relative, not screen-relative -- and that broke
"send a marine to the bottom right corner."**

- Every sector number the rest of the system uses (action names, a
  dispatch's `target_sector`) is in *canonical* space (`sector_grid.py`'s
  `SpawnOrientation`): mirrored per episode so sector 0 is always near
  home, regardless of which real map corner the player actually spawned in
  -- necessary for the trained policy to learn a stable meaning for
  "toward the enemy," since Simple64 randomizes spawn corners.
- Confirmed live, this silently broke directional player commands: with
  both axes mirrored (home spawned toward the map's real bottom-right),
  asking for "the bottom right corner" resolved to the *diagonally
  opposite* real corner, and the marine only landed where intended by
  asking for "bottom left" instead.
- **Fix**: the state description Claude sees now includes a real corner
  legend (`sector_grid.py`'s `real_corner_sectors()`, e.g. `top-left=15,
  top-right=12, ...`) computed by round-tripping each actual screen
  corner's center point through the same `orientation.to_canonical()`
  transform every other sector number already goes through -- so Claude
  looks the right sector number up by name instead of re-deriving it (and
  getting the mirroring backwards) from the row/col index math itself. The
  interpreter's system prompt tells Claude explicitly to prefer this
  legend whenever the player names an actual screen corner or edge, and to
  fall back to the row/col math only for home-relative phrasing like
  "toward the enemy."

**Releasing a unit back to AI control**, three ways, none of which need
Claude to disambiguate which marine you mean from a loose description:

- The page's per-unit **Release** button or its **Release all** button
  (`POST /units/release`, near-instant -- its own `control_queue`,
  separate from the Claude-bound `command_queue`, so a button click never
  shares the 30-second LLM-round-trip timeout).
- A voice/text command like "return all units to AI control"
  (`release_units` with `scope: "all"`).
- "return the marines in sector 3" (`scope`d by sector), or "let the AI
  take back the marine I just sent" (`scope: "most_recent"`).
- Releasing one arbitrarily-described unit by name is deliberately out of
  scope -- the per-unit button is the reliable path for that; asking
  Claude to guess which marine you mean without a stable player-facing
  identifier was left for a future round.

## New endpoints and the live activity feed

- `GET /state` -- current player-controlled unit tags and active banned
  actions, polled every 2s by the page's "Units under your control" and
  "Standing directives" panels.
- `GET /events?since=<id>` -- polled every 2s for anything new in
  `command/server.py`'s `EventLog` (a bounded, thread-safe, append-only
  deque with monotonic ids): denials, executed dispatches, releases, and
  directive changes all land here so the page shows them live without
  needing SSE/WebSockets for what's a single local user.
- `POST /units/release` -- body `{"tags": [...]}` / `{"sector": N}` / `{}`
  (release all); see [Per-unit player control](#per-unit-player-control-eg-send-a-marine-to-the-bottom-right-corner)
  above for why this has its own queue and its own short timeout instead of
  sharing `/command`'s.

## Timing and responsiveness

- **The game does not advance while a command is being interpreted**,
  because nothing calls `env.step()` until the Claude round-trip resolves
  -- the bot-mode client simply waits for the next step request, so a
  command always executes against the state the player actually saw. A
  declined command doesn't consume a game step either; the loop just falls
  through to `model.predict()` on the next iteration as if nothing
  happened.
- **A slow/hung Claude call must not freeze the whole game.** Because
  nothing else runs while a command is being interpreted (previous
  bullet), the Claude API call is squarely on the game loop's own critical
  path -- and the `anthropic` SDK's own defaults (600s read timeout, 2
  retries) mean an unbounded call there can stall *every* future
  `env.step()`, autonomous or commanded, for minutes: confirmed live, a
  command timed out in the browser (`server.py`'s own 30s wait gave up)
  while the main loop was still blocked inside `client.messages.create()`,
  so the autonomous policy stopped issuing any new orders at all until
  that call finally returned (SCVs kept auto-mining regardless -- that's
  an already-issued SC2 order, not something the loop re-triggers).
  `interactive_play.py` now builds the client via `_build_client()` with
  an explicit `timeout=12.0, max_retries=0` -- comfortably under
  `_COMMAND_TIMEOUT_SECONDS` and with no silent retry doubling that
  further, so a bad network moment fails fast into
  `interpret_command()`'s existing exception handler (a `DeclineResult`)
  instead of freezing play past the point the page already reported a
  timeout.
- **Startup order.** There's one command, not two -- `interactive_play.py`
  starts the command console (a background thread) and *then* launches
  StarCraft II itself (the first `env.reset()`, inside the same process),
  so the console is reachable slightly before the game has actually
  finished loading. Wait for the game to visibly be playing (the
  visualized window, or episode output in the terminal) before sending
  your first command: a command sent too early just sits queued until the
  main loop starts pulling from it, and if that wait passes the command
  endpoint's 30-second timeout (`command/server.py`'s
  `_COMMAND_TIMEOUT_SECONDS`) the browser gets a timeout error rather than
  an answer. There's currently no "game not ready yet" status shown on the
  page itself -- a reasonable follow-up if this turns out to matter in
  practice.
- **How the browser reaches the game.** There is no REST service inside
  StarCraft II, and the page never talks to the game directly. Everything
  -- the environment, the trained model, the Claude client, and the Flask
  server -- runs in the one `interactive_play.py` process; the browser
  only ever talks to that process's own local Flask endpoint
  (`http://127.0.0.1:8765`), which hands commands to the game loop through
  an in-process queue (`command/server.py`'s `PendingCommand` +
  `queue.Queue`). PySC2 talks to the actual SC2 executable over its own
  separate local protocol, untouched by any of this.
- **Realtime pacing.** By default, `SC2FightEnv` steps the game as fast as
  the client can simulate -- the right choice for training, evaluation, and
  demonstration collection, where wall-clock speed doesn't matter. A human
  typing or speaking a command needs the opposite: true StarCraft II
  pacing (22.4 game loops/second), which is pysc2's own `realtime` mode
  (`env.realtime` in `config.py`, plumbed straight through to
  `sc2_env.SC2Env`). `interactive_play.py` turns this **on by default** for
  exactly that reason -- pass `--no-realtime` to fall back to turbo speed
  if you want to stress-test the console without waiting on the clock.
  `play.py` keeps it off by default (`--realtime` to turn it on there
  too), since that entrypoint is normally used for fast bulk evaluation.

## Which model, and where that's set

- Commands are interpreted by `claude-haiku-4-5-20251001` -- the fastest,
  cheapest model in the current Claude lineup, and deliberately chosen:
  this call is a single forced tool use picking one item off a short,
  explicit menu (the current legal-action list), not open-ended reasoning,
  so a larger model buys nothing here.
- `command/interpreter.py`'s `DEFAULT_MODEL` constant is the single source
  of truth for it; override per-run without touching code via
  `--command-model claude-sonnet-5` (or any other model ID) on
  `interactive_play.py`.
- `state_summary.py` builds the natural-language state description Claude
  sees (minerals, army size, mobilized/garrison status, which sectors hold
  known enemies) -- deliberately separate from `observation.py`'s
  `featurize()`, whose normalized float vector means nothing to an LLM.

## Scope, by design

Each command still resolves to one discrete step of intent -- one action,
one dispatch, one release, or one directive change -- never a multi-step or
conditional plan (e.g. "stop building depots once I have 3 barracks" is out
of scope; directives are a flat ban/allow on an action name for the rest of
the episode).
