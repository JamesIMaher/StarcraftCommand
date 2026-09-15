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

## How it works

**Observation.** Each step, `GameState.from_observation()` parses PySC2's
`raw_units`/`player` fields into a snapshot, and `observation.py` turns that
into a flat `float32` vector: ~19 global scalars (minerals, supply
used/cap/headroom, marine count + average health, SCV count, supply
depot/barracks counts -- in-progress and complete, visible enemy count +
health, enemy race one-hot, episode-progress fraction) plus 4 features per
grid sector (friendly/enemy count and average health in that sector) for a
6x6 grid -- 163 floats total at the default grid size (`env.grid`,
configurable). This is a `gymnasium.spaces.Box(0.0, 1.0, shape=(163,))`.

**Home-relative sectors.** Maps like `Simple64` randomize which corner each
side spawns in between episodes (confirmed empirically: resets within the
same process landed the command center in different quadrants across
consecutive games). Both the observation's per-sector features and the
`move_army_to_sector_i` actions are computed in a *canonical, home-relative*
coordinate frame (`sector_grid.SpawnOrientation`, mirrored once per episode
from the starting command center position) rather than raw map coordinates
-- so "sector 0" always means "near home" and the farthest sector always
means "toward the enemy", in every episode, regardless of which corner you
actually spawned in. Without this, the same action index meant opposite
things in different games and the policy could never learn a stable
explore-toward-the-enemy or return-to-defend behavior.

**Action space.** A flat `Discrete(40)`: `no_op`, `build_supply_depot`,
`build_barracks`, `train_marine`, plus one `move_army_to_sector_i` per grid
cell (36 at the default 6x6 grid). Grid resolution is a real
coverage/thoroughness tradeoff, not just a display detail: on a 64x64 map, a
4x4 grid's 16x16 cells have a half-diagonal (~11.3) beyond a marine's sight
range (~9) -- confirmed live as the cause of enemy buildings tucked in a
cell corner going undetected during search-and-destroy (see
`env/scripted_policy.py`). 6x6's ~10.7x10.7 cells (half-diagonal ~7.5) fit
comfortably within sight range; going finer still improves coverage further
but costs more sectors to sweep within an episode's step budget.
`action_masking.py` computes which actions are legal each step (afford
checks, unit existence, per-type caps) -- illegal actions never get sampled
at all rather than resolving to a silent no-op, because `MaskablePPO` zeroes
out their probability directly in the action distribution before sampling.
Movement/attack actions specifically
stay illegal until `masking.min_marines_to_move` marines exist (**Mass /
Concentration of Force**) -- newly trained marines spawn near home, so while
blocked they default to passive defense there rather than being committed
piecemeal.

**Neural network.** `MaskablePPO`'s default `MlpPolicy`
(`sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy`) is two
small separate PyTorch MLPs reading the same 83-dim input: a **policy head**
(2 hidden layers x 64 units, `Tanh` activation, outputting 20 logits -> the
per-action probabilities after masking) and a **value head** (same shape,
outputting a single scalar -- the estimated value of the current state).
There's no shared trunk and no CNN/spatial convolution -- the input is
already the flat engineered feature vector above, not raw pixels, so a small
MLP is all that's needed.

**Training algorithm (PPO).** Each training iteration: (1) roll out
`n_steps` (256 by default) actions in the live environment using the current
policy, recording observations/actions/rewards/masks/value estimates; (2)
compute advantages via Generalized Advantage Estimation (GAE, `gamma` /
`gae_lambda`); (3) run `n_epochs` passes of minibatch (`batch_size`) gradient
descent over that rollout with the Adam optimizer, on the PPO clipped
surrogate loss (`clip_range` limits how far a single update can move the
policy) plus a value-function loss and an entropy bonus (`ent_coef`,
encourages exploration). This is what actually updates the PyTorch weights
-- `model.learn()` in `src/sc2rl/training/train.py` runs this loop, and it's
the same net_arch/algorithm regardless of CPU or GPU (`sb3-contrib` picks
the device automatically via `device="auto"`).

**Reward.** PySC2's own terminal win/loss reward (+1/-1/0), passed straight
through, plus optional dense per-step shaping (`env.reward.shaping_enabled`,
on by default) built from real PySC2 signals -- `obs.observation.score_cumulative`
and per-sector unit presence -- rather than invented heuristics:

- **Economic value delta** (`total_value_units` + `total_value_structures`,
  capped at `economic_value_cap`, default `4000`): rises as you train
  marines *and* as you complete supply depots/barracks, falls when units
  die. Including structures matters -- without it, building a barracks
  earned no immediate reward (only its eventual marines did), which showed
  up in practice as the policy learning to delay barracks construction. The
  cap matters separately: structures are already bounded by
  `masking.max_supply_depots`/`max_barracks`, but marine count has no upper
  limit, so without a cap a policy can farm this reward indefinitely by
  hoarding marines it never risks in combat -- confirmed live, a losing
  episode earned +2.85 in economic reward alone this way. Growth below the
  cap is fully rewarded; past it, growing the army further earns nothing
  more.
- **Kill value, scaled by Concentration of Force AND discounted**:
  `killed_value_units` + `killed_value_structures` delta, multiplied by both
  `min(1, marine_count / concentration_threshold)` (a kill landed with a
  large army earns full credit; one landed with a small, exposed squad earns
  much less) and the separate `kill_value_scale` (default `0.1`). The extra
  discount matters because killed-value only ever increases -- it is *not*
  offset by an eventual loss the way economic value is -- confirmed in
  practice: a losing episode's `ep_rew_mean` went *up*, because kills traded
  during a losing fight outweighed the terminal penalty and the
  (comparatively small) economic-collapse penalty. `kill_value_scale` keeps
  "traded some kills before losing" a minor bonus, not something that can
  rival actually winning.
- **Home-defense penalty** (Economy of Force / Security): a per-step
  penalty while the home sector has enemy units present and no marines
  there to respond.
- **Scouting bonus** (OODA loop -- Observe): a one-time reward the first
  time an enemy unit is seen in a given sector during an episode, rewarding
  exploration itself rather than only its downstream combat consequences.

All shaping coefficients are deliberately small relative to the terminal
+-1 reward (see the comments in `config.py`) -- shaping nudges the policy
toward useful sub-behaviors faster than sparse win/loss alone could teach
them, but the actual objective stays winning the game, not maximizing the
shaped proxy. Shaping applies on every step, **including the terminal one**
-- a loss typically means the base/army gets wiped out right at the end, so
computing the economic-value delta there too is what actually charges the
agent for that collapse; skipping it (an earlier bug) meant reward already
banked from building an economy earlier in the game was never offset by the
final defeat, so a losing episode's summed reward could still come out
positive. `reward.terminal_reward_scale` (default `1.0`) is an additional
knob to further weight the terminal win/loss signal if needed.

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
  training/train.py         MaskablePPO training entrypoint (supports --resume-from)
  inference/play.py         run a trained checkpoint against a live game
  config.py                 YAML -> typed config
tests/                     pytest suite, entirely against fakes/stubs (see tests/fakes/fake_pysc2.py)
```

## Setup

Needed on **every** machine you run this on (training or inference) --
StarCraft II itself has no cross-machine state, so this is the same on your
work PC and on a second machine (e.g. one with a real GPU).

### 1. Python 3.10

PySC2's dependency chain isn't reliably tested past 3.10/3.11. This repo
targets **3.10** specifically.

```powershell
winget install --id Python.Python.3.10 --source winget
```

### 2. Clone the repo and create a virtual environment

```powershell
git clone git@github.com:<your-username>/StarcraftCommand.git
cd StarcraftCommand
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

(SSH clone needs your SSH key present/loaded on that machine -- `ssh-add` it,
or use the HTTPS clone URL with a personal access token instead.)

### 3. Install dependencies

**PyTorch is deliberately not in `requirements.txt`** because the right
build depends on the machine:

```powershell
# CPU-only (no NVIDIA GPU, or a very old one) -- smaller download, this
# MLP-sized policy doesn't benefit much from a GPU anyway (see "GPU vs CPU"
# below), so this is the simplest default:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU present (e.g. a GTX 10-series / RTX card): install the CUDA
# build instead. Check `nvidia-smi` for your driver's max supported CUDA
# version first, then pick a matching index from
# https://pytorch.org/get-started/locally/ -- cu121 is a safe default for
# most current drivers:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then, on any machine:

```powershell
pip install -r requirements.txt
pip install -e . --no-deps
```

#### GPU vs CPU

`sb3-contrib` auto-selects CUDA if `torch.cuda.is_available()` -- no config
needed. That said, don't expect a big speedup here: the policy/value
networks are tiny (2x64-unit MLPs over an 83-dim vector), so the actual
matrix-multiply work is trivial either way. The real bottleneck is the
StarCraft II client itself stepping the game forward each action -- a
single-threaded, real-time-simulation cost that a GPU doesn't touch at all.
A GPU mainly helps once/if the network grows much larger (e.g. a future CNN
over spatial features) or once training is parallelized across many
concurrent `SC2Env` instances.

### 4. StarCraft II itself

- Install it, then set `SC2PATH` if it's not in the default location PySC2
  expects (`~/StarCraftII` on Linux; on Windows PySC2 auto-detects the
  standard Battle.net install path, normally
  `C:\Program Files (x86)\StarCraft II`).
- Download the official map pack and extract `Simple64.SC2Map` (and friends)
  into `<StarCraft II install>/Maps/` -- PySC2 does not fetch maps itself:

  ```powershell
  # Password-protected; downloading/extracting it means you agree to
  # Blizzard's AI and Machine Learning License.
  Invoke-WebRequest -Uri "https://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip" -OutFile Melee.zip
  Expand-Archive -Path Melee.zip -DestinationPath "C:\Program Files (x86)\StarCraft II\Maps" -Force
  # (Expand-Archive doesn't support the zip's password -- use 7-Zip or
  # `tar`/`unzip -P iagreetotheeula Melee.zip` from Git Bash instead if you
  # hit a password prompt.)
  ```

## Status

Fully verified end-to-end against a live StarCraft II client, including
several bugs that only surfaced once a real game was actually running (they
were invisible against the test fakes, which had modeled the observation
schema on assumption rather than the live one) -- see the git log for
specifics: PySC2 needing `absl` flags parsed before use, `raw_units` having
no `health_max` field (only `health` + `health_ratio`), a `protobuf` version
conflict between `pysc2` and `tensorboard`, build actions silently resolving
to no-ops because SCVs are never "idle" while auto-mining, a supply-headroom
masking heuristic that deadlocked the whole economy at game start, and
movement sectors being in absolute map coordinates rather than home-relative
ones (meaning the same action meant opposite things across episodes, since
spawn corner is randomized -- see "Home-relative sectors" above). All fixed
and covered by regression tests.

`pytest` covers everything in `src/sc2rl/env/` except the live-client parts
of `sc2_env_wrapper.py` against fakes/stubs (`tests/fakes/fake_pysc2.py`);
`tests/test_env_wrapper.py` and `tests/test_training_smoke.py` exercise the
full env -> Monitor -> DummyVecEnv -> MaskablePPO pipeline (including
action-mask auto-detection) against a stubbed `SC2Env`, and
`tests/test_training_resume.py` validates that `--resume-from` actually
continues training (weights, optimizer state, and timestep counter) rather
than restarting:

```powershell
pytest
```

## Training

```powershell
python -m sc2rl.training.train --config configs/default.yaml
```

Run `configs/train_fast.yaml` first if you haven't yet -- it's a short
(2,000-timestep) config meant to confirm the training loop, checkpointing,
and TensorBoard logging all work end-to-end before committing to a long run:

```powershell
python -m sc2rl.training.train --config configs/train_fast.yaml
tensorboard --logdir runs
```

### Checkpoints and resuming

Checkpoints land in `training.checkpoint_dir` (`checkpoints/` by default,
`.gitignore`d) as `.zip` files -- each one is a full snapshot: policy +
value network weights, the optimizer's state, and the hyperparameters used
to train it. One is saved every `training.checkpoint_freq` timesteps, plus a
`final_model.zip` when a run completes normally.

If a run gets interrupted (Ctrl-C, machine restart, corporate-policy
whatever), resume from the latest checkpoint instead of starting over:

```powershell
python -m sc2rl.training.train --config configs/default.yaml --resume-from checkpoints/sc2rl_50000_steps.zip
```

This restores the weights/optimizer exactly and trains for another
`training.ppo.total_timesteps` steps from wherever that checkpoint left off
(not up to an absolute total -- see the docstring in `train.py` if you want
the exact semantics).

### Imitation-learning warm start (behavior cloning)

Same spirit as AlphaStar's imitation-learning bootstrap from human replays,
adapted to what our action space actually supports: AlphaStar imitated raw
replay actions (mouse/camera/hotkeys) because its action space *was* that
full interface. Ours is a simplified custom abstraction with no direct
mapping from human replay data, so the practical equivalent is a **scripted
teacher** (`env/scripted_policy.py`) operating in our own action space:
build a small base, keep training marines, defend home if it's under attack
and undefended, and -- once mobilized to a real attack force
(`ScriptedPolicyConfig.attack_threshold`, default 20 marines, separate from
`masking.min_marines_to_move` which only gates whether movement is legal at
all) -- commit to a **search-and-destroy** pattern: sweep sectors
farthest-from-home first, redirecting immediately to any sector where the
enemy is actually spotted. A new order only fires once the majority of
marines are idle (`UnitInfo.is_idle` -- no active order, so neither
mid-fight nor still traveling); while busy, the teacher issues `no_op`,
which doesn't interrupt existing orders. This matters for two reasons,
both confirmed live: reissuing move orders while marines were still
mid-approach or mid-fight was part of why the group would scatter instead
of staying clustered (an exact-grid-cell "arrival" check also meant the
search could sit idle for a long time on an already-cleared area before
timing out and moving on -- the idle signal reacts immediately instead).
`search_timeout_steps` (default 60) is now just a safety net in case
marines somehow never go idle.

Collect a demonstration dataset from real games with it, then pretrain on
that before RL fine-tuning:

```powershell
python -m sc2rl.training.collect_demonstrations --config configs/default.yaml --episodes 15 --out demonstrations.npz
python -m sc2rl.training.train --config configs/default.yaml --bc-dataset demonstrations.npz
```

`--bc-dataset` initializes the policy's weights via supervised learning
(cross-entropy against the scripted teacher's choices, with action masking
applied so an imitated-but-illegal action never gets credit) before
`.learn()` starts -- mutually exclusive with `--resume-from`, since a
resumed checkpoint already has trained weights.

## Running a trained agent

```powershell
python -m sc2rl.inference.play --checkpoint checkpoints/final_model --episodes 5 --visualize
```

## Key defaults (see `configs/default.yaml`)

- Map: `Simple64` vs. `very_easy` Zerg bot (matches the old repo's baseline,
  both overridable).
- Action space: `no_op`, `build_supply_depot`, `build_barracks`,
  `train_marine`, plus one `move_army_to_sector_i` per grid cell (default
  6x6 = 36 cells) -- 40 actions total.
- Reward: PySC2's own terminal win/loss reward plus dense shaping, on by
  default (see "How it works" above for the individual terms).
- Single environment (`DummyVecEnv` with one `SC2FightEnv`). StarCraft II's
  per-step client overhead is the real bottleneck, not neural-net compute or
  environment parallelism -- revisit `SubprocVecEnv` (multiple concurrent
  `SC2Env` instances) only if training throughput turns out to matter in
  practice.

### Tuning the reward/masking config knobs

All under `env:` in `configs/default.yaml`:

| Key | Default | What it does |
|---|---|---|
| `masking.min_marines_to_move` | `4` | Movement/attack actions illegal below this many marines |
| `reward.shaping_enabled` | `true` | Master on/off switch for everything below |
| `reward.shaping_coefficient` | `0.001` | Scales the army-value and kill-value shaping terms |
| `reward.economic_value_cap` | `4000.0` | Ceiling on economic value used for the reward -- prevents indefinite hoarding |
| `reward.concentration_threshold` | `4` | Marine count for full kill-reward credit; scaled down below it |
| `reward.kill_value_scale` | `0.1` | Additional discount on kill-value credit, on top of concentration scaling |
| `reward.home_defense_penalty` | `0.05` | Per-step penalty while home is undefended and under attack |
| `reward.scouting_bonus` | `0.02` | One-time reward per newly-sighted enemy sector per episode |
| `reward.terminal_reward_scale` | `1.0` | Multiplies PySC2's own terminal win/loss reward |

`SC2FightEnv.step()` also returns each component separately in its `info`
dict (`reward_terminal`, `reward_economic`, `reward_kill`,
`reward_home_defense`, `reward_scouting`, summing to the total reward).
`training/callbacks.py`'s `RewardBreakdownCallback` (wired into every
training run by default) accumulates these per episode and prints a line to
the console the moment each episode ends, e.g.:

```
[episode end] total=-0.847  terminal=-1.000 economic=+0.320 kill=+0.004 home_defense=-0.150 scouting=+0.020
```

It also logs each component to TensorBoard under `reward_breakdown/*`. This
is how to actually see which term is driving an imbalance instead of
reasoning about it from formulas.

`masking.min_marines_to_move` and `reward.concentration_threshold` are
separate knobs on purpose -- one is a hard action-legality gate, the other a
soft reward scaling -- but they default to the same value (4) since they're
both expressing "this is what counts as a real squad" for this scenario;
tune them independently if that stops making sense (e.g. a larger map where
you'd want a bigger minimum force before committing).
