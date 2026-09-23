# StarcraftCommand

A PyTorch/PPO rewrite of [`StarcraftFightMission`](https://github.com/) -- a
StarCraft II agent that learns to both manage its economy (supply depots,
barracks, marine production) and command its army (where to move/attack),
instead of scripting the economy and only learning a 4-way movement choice
like the original did.

## Why this exists

The original project used a hand-rolled Keras net and a crude reward trick
(win/loss split evenly across every step of an episode, regressed with MAE
loss -- no discounting, no advantage estimation, no action masking). This
rewrite:

- Uses **PyTorch** via **Stable-Baselines3 + sb3-contrib's `MaskablePPO`**
  instead of a hand-rolled training loop -- proper GAE-based credit
  assignment, and invalid-action masking instead of silently no-op'ing
  illegal moves.
- Lets the network decide **economy actions** (build supply depot, build
  barracks, train marine) **and** army movement in one unified action space,
  instead of scripting the economy.
- Is architected so a future generative-AI command layer can sit on top:
  `src/sc2rl/env/action_translation.py` and `action_space.py` expose a
  **named** command interface (`"build_barracks"`, `"move_army_to_sector_3"`,
  ...), not just numeric action indices, so a higher-level (human- or
  LLM-driven) layer can issue the same commands the trained policy does --
  see [docs/COMMAND_CONSOLE.md](docs/COMMAND_CONSOLE.md), which builds
  exactly that.

See `src/sc2rl/env/` for the core modules; every file there except
`sc2_env_wrapper.py` is plain Python/dataclasses/numpy with no PySC2
dependency, which is what makes the whole thing unit-testable without a
running StarCraft II client (see "Status" below).

## Documentation

| Doc | Covers |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Installing Python, dependencies, and StarCraft II itself |
| [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) | Observation/action space design, garrison, movement targeting, reward shaping, PPO settings -- the "why," with the live-debugging stories behind each choice |
| [docs/TRAINING.md](docs/TRAINING.md) | Running training, checkpoints/resuming, the behavior-cloning warm start |
| [docs/COMMAND_CONSOLE.md](docs/COMMAND_CONSOLE.md) | The natural-language web console: standing directives, per-unit player control, all five Claude tools |
| [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Every `env:` config knob, defaults, and the per-episode reward breakdown |
| [docs/KAMIWAZA.md](docs/KAMIWAZA.md) | Hosting the command console on Kamiwaza, with gpt-oss-120b interpreting commands |

## Repository layout

```
configs/                  YAML configs (default.yaml, train_fast.yaml)
docs/                      Setup, design rationale, training, command console, config reference
src/sc2rl/
  env/
    sc2_env_wrapper.py     gymnasium.Env wrapping pysc2 -- the only file that touches a live client
    sector_grid.py         grid/sector math shared by the action space and observation
    pathing.py             pathable attack targets from the game's minimap pathable layer
    game_state.py           per-step snapshot parsed from raw_units/player
    observation.py          GameState -> feature vector
    action_space.py         named Discrete(N) action registry
    action_masking.py       legality mask (MaskablePPO's action_masks() source)
    action_translation.py   action index/name -> raw PySC2 FunctionCalls
  training/train.py         MaskablePPO training entrypoint (supports --resume-from)
  inference/play.py         run a trained checkpoint against a live game
  inference/interactive_play.py  same, plus a local web command console (see docs/COMMAND_CONSOLE.md)
  command/
    state_summary.py        GameState -> short human/LLM-readable description
    interpreter.py           one command + legal actions -> one forced tool call (Claude or OpenAI-compatible) -> one action
    server.py                 Flask app: the command console's backend
    relay_client.py           connects the game loop to the Kamiwaza-hosted console instead
    templates/command.html   the command console's page (text box + mic button)
  config.py                 YAML -> typed config
kamiwaza-app/              the command console as a Kamiwaza App Garden app (see docs/KAMIWAZA.md)
tests/                     pytest suite, entirely against fakes/stubs (see tests/fakes/fake_pysc2.py)
```

## Quick start

1. [Set up the environment](docs/SETUP.md) (Python 3.10, dependencies,
   StarCraft II + maps).
2. [Train a policy](docs/TRAINING.md):
   ```powershell
   python -m sc2rl.training.train --config configs/train_fast.yaml
   ```
3. Run it:
   ```powershell
   python -m sc2rl.inference.play --checkpoint checkpoints/final_model --episodes 5 --visualize
   ```
   or, for the natural-language command console, see
   [docs/COMMAND_CONSOLE.md](docs/COMMAND_CONSOLE.md).

## Status

- Fully verified end-to-end against a live StarCraft II client, including
  several bugs that only surfaced once a real game was actually running
  (they were invisible against the test fakes, which had modeled the
  observation schema on assumption rather than the live one). See the git
  log for specifics, including:
  - PySC2 needing `absl` flags parsed before use.
  - `raw_units` having no `health_max` field (only `health` +
    `health_ratio`).
  - A `protobuf` version conflict between `pysc2` and `tensorboard`.
  - Build actions silently resolving to no-ops because SCVs are never
    "idle" while auto-mining.
  - A supply-headroom masking heuristic that deadlocked the whole economy
    at game start.
  - Movement sectors being in absolute map coordinates rather than
    home-relative ones (meaning the same action meant opposite things
    across episodes, since spawn corner is randomized -- see
    [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#home-relative-sectors)).
  - All fixed and covered by regression tests.
- Run the test suite with:
  ```powershell
  pytest
  ```
  See [docs/SETUP.md#verifying-the-setup](docs/SETUP.md#verifying-the-setup)
  for what it covers and what it doesn't (the live-client parts of
  `sc2_env_wrapper.py`).
