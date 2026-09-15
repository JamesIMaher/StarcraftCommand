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
  LLM-driven) layer can issue the same commands the trained policy does.

See `src/sc2rl/env/` for the core modules; every file there except
`sc2_env_wrapper.py` is plain Python/dataclasses/numpy with no PySC2
dependency, which is what makes the whole thing unit-testable without a
running StarCraft II client (see "Status" below).

## Repository layout

```
configs/                  YAML configs (default.yaml, train_fast.yaml)
src/sc2rl/
  env/
    sc2_env_wrapper.py     gymnasium.Env wrapping pysc2 -- the only file that touches a live client
    sector_grid.py         grid/sector math shared by the action space and observation
    game_state.py           per-step snapshot parsed from raw_units/player
    observation.py          GameState -> feature vector
    action_space.py         named Discrete(N) action registry
    action_masking.py       legality mask (MaskablePPO's action_masks() source)
    action_translation.py   action index/name -> raw PySC2 FunctionCalls
  training/train.py         MaskablePPO training entrypoint
  inference/play.py         run a trained checkpoint against a live game
  config.py                 YAML -> typed config
tests/                     pytest suite, entirely against fakes/stubs (see tests/fakes/fake_pysc2.py)
```

## Setup

### 1. Python 3.10

PySC2's dependency chain isn't reliably tested past 3.10/3.11. This repo
targets **3.10** specifically.

```powershell
winget install --id Python.Python.3.10 --source winget
```

### 2. Virtual environment

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

PyTorch is deliberately **not** in `requirements.txt` -- install the CPU-only
build first (a plain `pip install torch` pulls a multi-GB CUDA build you
don't need for this MLP-sized policy; there's no GPU-bound work here, PySC2's
game-stepping is the actual bottleneck):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
```

### 4. StarCraft II itself

Not installed on this machine yet (an install attempt errored and needs a
system restart to retry -- ask Claude Code to help debug that when you get to
it). Once it's installed:

- Set `SC2PATH` if StarCraft II isn't in the default location PySC2 expects
  (`~/StarCraftII` on Linux; on Windows PySC2 auto-detects the standard
  Battle.net install path).
- Unzip the map pack containing `Simple64.SC2Map` into
  `<StarCraft II install>/Maps/` -- PySC2 does not fetch maps itself.

## Status: what's verified here vs. what needs a live game

This repo was built without a StarCraft II install available. Everything in
`src/sc2rl/env/` **except** `sc2_env_wrapper.py` (sector math, game-state
parsing, observation featurization, action masking, action translation) is
covered by `pytest` against fakes in `tests/fakes/fake_pysc2.py` -- run:

```powershell
pytest
```

`tests/test_env_wrapper.py` also exercises `SC2FightEnv`'s `step()`/`reset()`
contract against a *stubbed* `SC2Env` (dependency-injected via the
`env_factory` constructor argument), and `tests/test_training_smoke.py` runs
a few real `MaskablePPO.learn()` steps against that same stub -- so the full
env -> Monitor -> DummyVecEnv -> MaskablePPO wiring, including action-mask
auto-detection, is validated end-to-end without a live client.

**Not verified here** (needs the real game once installed): an actual
`SC2Env` connection, a real training run's learning curve, and map/scenario
behavior specific to `Simple64`.

## Training

```powershell
python -m sc2rl.training.train --config configs/default.yaml
```

Run `configs/train_fast.yaml` first once StarCraft II is installed -- it's a
short (2,000-timestep) config meant to confirm the training loop,
checkpointing, and TensorBoard logging all work end-to-end before committing
to a long run:

```powershell
python -m sc2rl.training.train --config configs/train_fast.yaml
tensorboard --logdir runs
```

Checkpoints land in `training.checkpoint_dir` (`checkpoints/` by default,
`.gitignore`d) as `.zip` files, saved every `training.checkpoint_freq`
timesteps plus a `final_model.zip` at the end.

## Running a trained agent

```powershell
python -m sc2rl.inference.play --checkpoint checkpoints/final_model --episodes 5 --visualize
```

## Key defaults (see `configs/default.yaml`)

- Map: `Simple64` vs. `very_easy` Zerg bot (matches the old repo's baseline,
  both overridable).
- Action space: `no_op`, `build_supply_depot`, `build_barracks`,
  `train_marine`, plus one `move_army_to_sector_i` per grid cell (default
  4x4 = 16 cells) -- 20 actions total.
- Reward: PySC2's own terminal win/loss reward. Dense per-step reward shaping
  (friendly/enemy army-value delta) exists but is **off by default** --
  `env.reward.shaping_enabled` in config -- so the first real training run
  validates against a clean sparse-reward baseline first.
- Single environment (`DummyVecEnv` with one `SC2FightEnv`), CPU-only PyTorch.
  StarCraft II's per-step client overhead is the real bottleneck, not GPU
  compute or environment parallelism -- revisit `SubprocVecEnv` only if
  training throughput turns out to matter once a real run is possible.
