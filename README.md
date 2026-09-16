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
into a flat `float32` vector: 20 global scalars (minerals, supply
used/cap/headroom, marine count + average health + idle fraction, SCV count, supply
depot/barracks counts -- in-progress and complete, visible enemy count +
health, enemy race one-hot, episode-progress fraction) plus 6 features per
grid sector for a 6x6 grid -- 236 floats total at the default grid size
(`env.grid`, configurable). This is a
`gymnasium.spaces.Box(0.0, 1.0, shape=(236,))`. The idle fraction ("has
the army finished what it was told to do") is there because the scripted
teacher's whole reorder rule is majority-idle: without it the imitation
target depended on something the policy couldn't see, a hard floor on the
BC loss, and the RL policy couldn't tell an army mid-move from one standing
around. Four of the per-sector
features are a snapshot of the current step (friendly/enemy count and
average health in that sector); the other two are **minimap memory**:

- *known enemy structures*: enemy buildings the player has seen in that
  sector. PySC2 keeps a previously-seen structure in `raw_units` as a
  `display_type=Snapshot` entry while it's back in fog, so a discovered
  building persists here until the area is re-seen without it -- exactly
  what a human reads off the minimap, not extra information. Structures
  are recognized by unit type (`game_state.py`), since `raw_units` has no
  is-structure flag.
- *explored*: whether a friendly marine has been in that sector this
  episode (the home sector counts from the start). This is episode state
  the env maintains, not part of any single `GameState`. Without it the
  policy had no way to tell an empty sector it had already swept from one
  it had never looked at, so it couldn't do a systematic search -- observed
  live as the army leaving enemy buildings standing in sectors it simply
  never visited.

Changing the observation layout invalidates any previously collected BC
dataset; `train.py` refuses a `--bc-dataset` whose observation width doesn't
match the current environment rather than silently training on garbage.

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
cell (36 at the default 6x6 grid). The grid is laid over the map's
*playable area* (see below), which on Simple64 is a ~43x43 square in the
raw frame. Grid resolution is a real coverage/thoroughness tradeoff, not
just a display detail: the original 4x4 grid over the full 64x64 square had
16x16 cells with a half-diagonal (~11.3) beyond a marine's sight range (~9)
-- confirmed live as the cause of enemy buildings tucked in a cell corner
going undetected during search-and-destroy (see `env/scripted_policy.py`).
6x6 over the playable area gives ~7.1x7.1 cells (half-diagonal ~5), so
standing at a cell's center sees the whole cell; going finer still improves
coverage further but costs more sectors to sweep within an episode.
`action_masking.py` computes which actions are legal each step (afford
checks, unit existence, per-type caps) -- illegal actions never get sampled
at all rather than resolving to a silent no-op, because `MaskablePPO` zeroes
out their probability directly in the action distribution before sampling.
Movement/attack actions specifically
stay illegal until `masking.min_marines_to_move` marines exist (**Mass /
Concentration of Force**) -- newly trained marines spawn near home, so while
blocked they default to passive defense there rather than being committed
piecemeal. Moving to any sector *other than home* needs a separate, higher
`masking.min_marines_to_advance` (default `20`, matching
`scripted_policy.py`'s own `attack_threshold`): without this as a hard mask,
nothing stopped the RL policy from sending the whole army into unexplored
territory with only `min_marines_to_move` marines -- confirmed live as
marines exploring too early with too few marines and dying immediately,
which then taught the policy to avoid moving at all rather than to wait for
mass. Moving *to* home (e.g. recalling a scattered force, or defending)
still only needs the lower `min_marines_to_move` bar.

**Garrison.** Every move order -- whichever sector it targets -- is only
ever given to the army *minus* a small standing garrison
(`env.garrison_size`, default 4): those marines are held at the command
center and simply excluded from the sweep, so an offensive can never strip
the base of every defender. This isn't a policy choice (the action space
didn't change), it's enforced by `SC2FightEnv._update_garrison()`
underneath whatever action gets chosen, so it applies identically to the
scripted teacher and the RL-trained policy without touching the BC dataset
or the observation. Membership is a stable set of marine tags -- nearest to
the command center, topped up from the next-nearest survivor as garrisoned
marines die -- not reassigned from scratch each step, so the same few
marines hold the position instead of churning. This exists because every
RL episode observed before it was added showed the home-defense penalty
maxed out, win or loss: the single "move the whole army" action structurally
could not hedge between offense and defense, so a long game's inevitable
opportunistic raid on an empty base was often what actually decided a
loss the search-and-destroy should have won outright.

**Where a move actually goes.** `action_translation.py` turns
`move_army_to_sector_i` into one attack-move per non-garrisoned marine
(with a little jitter so they don't all path to the identical point). If
the sector holds
a *known enemy structure* (visible, or a fog snapshot of one seen earlier),
the target is the structure itself -- the one nearest the army -- so "go to
the sector with the building" resolves to "go to the building." A move to
the command center's sector targets the command center itself, so a recall
gathers the army at the base rather than at the cell's midpoint. "Home" is
never a hard-coded sector index: the recall sector follows the command
center (`sector_grid.home_sector`), and "the base" for defense purposes is
every sector with one of our buildings in it -- with ~7-unit cells the base
straddles several, and a hard-coded sector 0 silently missed attacks on the
barracks next door (confirmed live as the scripted teacher never returning
to defend, and losing most games, right after the grid was laid over the
playable area). Otherwise it's the *reachable* point
nearest the sector's center: the minimap `pathable` feature layer
(`env/pathing.py`), which at minimap size == raw resolution is in the exact
frame unit positions use, flood-filled from the command center so only
ground actually connected to the base counts. Both halves matter. The
playable rectangle contains unpathable terrain -- cliffs, and on Simple64
the two corners that aren't bases -- and a sector center on a cliff sent
the army to park at the cliff's edge. And *pathable* isn't *reachable*: the
layer marks cliff-top plateaus and isolated pockets as pathable even with
no ramp to them, so "nearest pathable cell" still picked ground the army
could never stand on (both confirmed live as marines "trying to reach
areas they can't reach"). And *reachable* still isn't enough: Simple64 is
two plateaus with cliff lines between them, and the Euclidean-nearest
reachable cell to a point on a plateau is often at the *foot* of the cliff
directly below it -- legal, reachable, wrong level. Marines sent there
stand under the cliff with no vision of the building above, and since that
building never dies the teacher re-targets the sector forever (confirmed
live as the army piling up under a different map edge each game). So a
known enemy structure's target is snapped to the nearest reachable ground
*on the same terrain level* as the structure, using the minimap
`height_map` layer (its own footprint is unpathable), and a sector's sweep
target prefers interior cells over cliff-edge cells, which are ambiguous
about which level they belong to. A sector with no reachable
ground at all is removed from the action space for the episode
(`SC2FightEnv.unreachable_sectors`) and pre-marked explored so no
exploration incentive points at it; the env prints the reachable cell
count and any such sectors once at reset (`[env] pathing: ...`), or a
warning if the observation has no pathable layer. Every target is also clamped strictly inside the game's own
`playable_area` (read from `SC2Env.game_info` at reset), because the
playable area is inset from the nominal 64x64 square: an earlier version biased edge-sector targets all the
way to the literal map corner to reach corner buildings on the old 4x4
grid, and since that point is unpathable the attack-move never completed
-- the whole army would park at the corner cliff for the rest of the game.

One coordinate-frame subtlety worth knowing: `raw_units` positions and the
`world` argument of raw actions are *not* world coordinates. PySC2 scales
them by `raw_resolution / max(map width, height)` and flips the y axis
(`features.Features.init_camera`), and applies the inverse to raw actions,
so observations and actions agree with each other -- but `game_info`
(map size, playable area) is in real world coordinates and must be pushed
through the same transform before it can be compared to anything else.

**The grid is laid over the playable area, not the map square.** Simple64
is actually 88x96 world units with a playable area of (12,12)-(76,76); in
the raw frame that is `x 8.0..50.7, y 13.3..56.0` -- a ~43-unit square
sitting off-center in the 64x64 square. A grid over the full square wasted
the entire last column and first row on sectors nothing could ever reach,
so their `explored` flag stayed 0 forever and the exploration/stale-search
incentives kept pulling the army toward those edges (confirmed live as the
army bunching at a map edge). At reset the env reads the playable area from
`game_info`, converts it to the raw frame, and re-lays the sector grid
over it (`SectorGrid.bounds`) -- the number of sectors, and so the action
and observation space sizes, never change, only the geometry. Home-relative
mirroring reflects about the playable area's center for the same reason.
The env prints the bounds it used once (`[env] sector grid laid over ...`).

**Neural network.** `MaskablePPO`'s default `MlpPolicy`
(`sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy`) is two
small separate PyTorch MLPs reading the same 236-dim input: a **policy head**
(2 hidden layers x 64 units, `Tanh` activation, outputting 40 logits -> the
per-action probabilities after masking) and a **value head** (same shape,
outputting a single scalar -- the estimated value of the current state).
There's no shared trunk and no CNN/spatial convolution -- the input is
already the flat engineered feature vector above, not raw pixels, so a small
MLP is all that's needed.

**Training algorithm (PPO).** Each training iteration: (1) roll out
`n_steps` (2048 by default) actions in the live environment using the current
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

The PPO settings in `configs/default.yaml` deliberately differ from SB3's
defaults because of two things about this problem: episodes have no step
limit and run thousands of steps, and the policy starts from a
behavior-cloned initialization worth preserving. `n_steps` is 2048 (a
256-step rollout almost never contained an episode end, so every update
was bootstrapping from a mid-game value estimate), `gamma` is 0.995 (at
0.99 the ~100-step discount horizon shrank the +-10 terminal reward to
~0.05 by the time credit reached the decisions that decide a game --
smaller than the immediate dense shaping terms, which the policy then
optimized instead), `learning_rate` is 1e-4 (at 3e-4 fine-tuning was
observed to win a few early episodes and then drift steadily worse), and
`ent_coef` is 0.02. That entropy bonus is the built-in "randomization": it
penalizes a peaked action distribution so the policy keeps sampling
alternatives. If a run collapses into a do-nothing local minimum, check
`train/entropy_loss` in the console output -- trending toward 0 means the
policy has become near-deterministic and stopped exploring, and raising
`ent_coef` is the first lever.

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
- **Kill value, scaled by Concentration of Force, discounted, AND capped**:
  `killed_value_units` + `killed_value_structures` delta, multiplied by both
  `min(1, marine_count / concentration_threshold)` (a kill landed with a
  large army earns full credit; one landed with a small, exposed squad earns
  much less) and the separate `kill_value_scale` (default `0.1`), then
  capped at `kill_value_cap` (default `2000.0`). Killed-value only ever
  increases -- it is *not* offset by an eventual loss the way economic value
  is, and has no natural ceiling the way a hard cap gives total_value (a
  long grindy fight against a continuously-spawning bot can otherwise run it
  into the thousands) -- confirmed in practice: a losing episode's
  `ep_rew_mean` went *up* twice, once from kill-value alone and later even
  with the discount in place, from economic value alone before it had a cap.
- **Home-defense penalty** (Economy of Force / Security): a per-step
  penalty while any sector containing one of our buildings has enemy units
  present and no marines there to respond, capped per episode at `home_defense_penalty_cap`
  (default `1.0`). Episodes have no step limit (only PySC2's own game-end
  conditions), so an uncapped version of this could accumulate for
  hundreds or thousands of steps in an unusually long episode -- confirmed
  live: an early, untrained episode's `ep_rew_mean` reached **-46.4** after
  only 18,000 timesteps, far beyond anything the terminal_reward_scale
  analysis below accounted for, because that analysis only ever reasoned
  about a single-step transition, never an episode-length accumulation of
  an uncapped per-step penalty.
- **Scouting bonus** (OODA loop -- Observe): a one-time reward the first
  time an enemy unit is seen in a given sector during an episode, rewarding
  exploration itself rather than only its downstream combat consequences.
- **Exploration bonus** (default `0.1`): a one-time reward the first time
  a friendly marine is present in a given sector during an episode (the
  base's sectors count as already visited at spawn, and so do sectors with
  no pathable ground), independent of whether an enemy is there. Raised
  from 0.02 after the policy settled into staying home with 20+ marines: at
  0.02 a new sector was worth less than half of one marine (+0.05), so
  exploring could never compete with sitting still. This
  exists specifically as a counterweight to the home-defense penalty: that
  penalty fires every step the home sector is undefended, for as long as
  that holds, with no cap, while scouting_bonus only pays once per sector
  and economic reward stops once `economic_value_cap` is hit. Without a
  standing incentive to stay forward, once those one-off bonuses are spent
  the net marginal reward of continuing to search can go to zero or
  negative while returning home is strictly zero-or-better -- observed live
  as marines oscillating back toward home instead of continuing to search.
- **Stale-search penalty**: a further counterweight to the same stalling
  problem -- exploration_bonus is *also* only one-time per sector, so once
  the army has visited what it's going to visit for a while, nothing keeps
  actively pulling it onward. This is a flat per-step penalty
  (`stale_search_penalty`, default `0.02`) once the army has gone
  `stale_search_patience` (default `20`) steps without entering a sector it
  hasn't been in before -- applied only once movement is actually legal, so
  the early economy-building phase (correctly sitting at home) is never
  penalized. Observed live as a large, fully-mobilized army parking in one
  sector indefinitely -- "a huge pile of marines in one location." Also
  capped per episode (`stale_search_penalty_cap`, default `2.0`), same
  unbounded-episode-length reasoning as the home-defense penalty above.
- **Time penalty** (`time_penalty_per_step`, default `0.001`, capped at
  `time_penalty_cap` `2.0` per episode): a flat cost per step from the
  first step, so finishing sooner is worth more than finishing later at all
  -- the only pressure in the reward that says "get on with it" once the
  economy is built and the army is safe at home.
- **Approach reward** (`approach_reward_scale`, default `2.0`): the
  step-to-step change in a potential
  `-scale * distance(army centroid, nearest known enemy structure) / grid diagonal`,
  so closing distance to a known enemy building pays continuously and
  backing away costs the same -- "go attack the base" is rewarded step by
  step instead of only at the terminal win hundreds of steps later, which
  `gamma` discounts to almost nothing. Potential-based, so it sums to at
  most `scale` across the whole map; the positive and negative totals are
  each capped per episode at `approach_reward_cap` (`3.0`). The step on
  which the set of known structures changes (a discovery or a kill) earns
  nothing, otherwise destroying the last building of a base would be
  charged as "the nearest structure just got farther away."

Even with every component capped, their *sum* can still reach a few points
of reward regardless of outcome -- capping bounds each channel, but only the
terminal term actually guarantees a win's total stays positive and a loss's
stays negative. So `reward.terminal_reward_scale` (default `20.0`) is
deliberately set well above 1: with the current caps, worst-case POSITIVE
shaping per episode is roughly `shaping_coefficient * (economic_value_cap +
kill_value_scale * kill_value_cap) + (scouting_bonus + exploration_bonus) *
num_sectors + approach_reward_cap` ~= 11.5, and worst-case NEGATIVE shaping
is roughly `-(shaping_coefficient * economic_value_cap +
home_defense_penalty_cap + stale_search_penalty_cap + time_penalty_cap +
approach_reward_cap)` ~= -12 (economic value has no further downside once
it's dropped to zero; killed value never decreases, so it has no negative
side at all). 20x here (+-20) comfortably dominates both
directions with margin, regardless of how long an episode runs or how much
shaping it racks up either way. This is tested directly against the actual
shipped defaults
(`test_default_config_guarantees_any_win_outscores_a_heavily_shaped_loss`,
plus `test_win_reward_stays_positive_despite_a_long_troubled_episode` for
the specific failure mode of an unbounded-length episode -- the earlier,
single-step version of this test could not have caught `ep_rew_mean`
reaching -46.4 in one long early episode, since home_defense_penalty and
stale_search_penalty are both per-step terms that only overrun their
intended bound across many steps, not within a single one), specifically to
catch this class of imbalance if the caps/scale are ever retuned again.

Shaping applies on every step, **including the terminal one** -- a loss
typically means the base/army gets wiped out right at the end, so computing
the economic-value delta there too is what actually charges the agent for
that collapse; skipping it (an earlier bug) meant reward already banked
from building an economy earlier in the game was never offset by the final
defeat.

## Repository layout

```
configs/                  YAML configs (default.yaml, train_fast.yaml)
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
  inference/interactive_play.py  same, plus a local web command console (see below)
  command/
    state_summary.py        GameState -> short human/LLM-readable description
    interpreter.py           one command + legal actions -> one Claude tool call -> one action
    server.py                 Flask app: the command console's backend
    templates/command.html   the command console's page (text box + mic button)
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
networks are tiny (2x64-unit MLPs over a 236-dim vector), so the actual
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
build a small base, keep training marines, recall the whole offensive
force home only if a sector with one of our buildings is under a threat
bigger than the standing garrison can handle on its own (`garrison_size`,
passed in by the caller -- a single stray unit near home no longer
interrupts the search, which previously made the army destroy a few
enemies elsewhere and then abandon a real, more distant target every time
it happened), and -- once mobilized
to a real attack force (`ScriptedPolicyConfig.attack_threshold`, default 20
marines, matching `masking.min_marines_to_advance`) -- commit to a
**search-and-destroy** pattern. Below the threshold it holds at the base.
Once launched, the offensive has hysteresis: the env remembers the army
reached `masking.min_marines_to_advance` ("mobilized"), the movement mask
then keeps advancing legal down to `masking.min_marines_to_continue`
(default 8), and the teacher keeps pressing as long as the mask allows.
Without that, an army that launched at 20 and lost a few marines was
forbidden from going anywhere but home and walked away from the enemy's
last two buildings (observed live -- that game was only won because a
couple of stragglers never got the recall). Only when the mask closes does
it regroup at the command center, rather than holding mid-map -- holding
there meant getting picked off while reinforcements piled up at home,
observed live as the army parked at the map's center for a very long time. The pattern itself (over legal targets
only -- sectors with no pathable ground are never picked): sweep sectors
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

**Reading the BC loss.** It's the mean negative log-likelihood of the
teacher's action, so `exp(-loss)` is the average probability the policy
gives to what the teacher did: 0.2 means ~82%. It does not go to zero and
shouldn't be expected to -- the teacher's choice depends on its own hidden
state (current search target, cleared-sector memory), so the same
observation can legitimately map to different actions. Each epoch also
prints `held_out_loss` on a ~10% held-out slice of the dataset that was
never trained on (never a random shuffle: consecutive steps of one game
are near-duplicates, so a shuffled split leaks and flatters). That number
answers "should I train more epochs?": if it's still falling alongside the
training loss, yes (`--bc-epochs 30`); if it's flat or rising while the
training loss keeps dropping, the floor has been reached and more epochs
would only memorize the dataset. The loss is a proxy anyway -- what matters
is whether the pretrained policy plays like the teacher, which
`sc2rl.inference.play` on the saved model shows directly.

The held-out slice is *whole trailing episodes*, not a raw slice of the
concatenated array, and spans at least two of them whenever there are
enough to spare (`training/behavior_cloning.py`'s `_episode_validation_split`,
using the `episode_ids` array `collect_demonstrations.py` saves alongside
the observations). This matters more than it sounds like it should:
confirmed live, one unusually long episode by itself exceeded an entire
10%-of-samples target, so a plain tail slice landed inside that one game --
the reported held-out loss was really just "how well does the policy
predict this specific game," and it picked epoch 1 as best while later
epochs' held-out loss climbed from 0.50 to over 1.3, nothing like the clean
bottom-then-rise curve a genuine multi-game split gives. A dataset saved
before `episode_ids` existed (or with fewer than 3 distinct episodes) falls
back to the old plain tail slice automatically.

A dataset is tied to the observation layout it was collected under: any
change to `observation.py`'s features or to `env.grid` changes the vector
width, and `train.py` will refuse the stale dataset with a message saying to
re-collect. Old checkpoints are likewise incompatible after such a change
(the network's input layer is sized to the observation), so start a fresh
run rather than `--resume-from`.

## Running a trained agent

```powershell
python -m sc2rl.inference.play --checkpoint checkpoints/final_model --episodes 5 --visualize
```

### Live command console (Generative-AI layer)

`interactive_play.py` runs the same trained policy but adds a local web page
where you can type -- or speak, via the browser's built-in speech
recognition (Chrome/Edge) -- an instruction like "return the marines to
base" or "build a supply depot." It interrupts autonomous play for exactly
one action, then hands control straight back to the model:

```powershell
copy .env.example .env
notepad .env    # paste your key in place of sk-ant-...
python -m sc2rl.inference.interactive_play --checkpoint checkpoints/final_model --visualize
```

**Where the API key lives.** `interactive_play.py` calls `load_dotenv()` on
startup, which reads a local `.env` file (gitignored, never committed) into
the process environment -- copy `.env.example` to `.env` and fill in a real
key from
[console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
This is the recommended way on Windows: the key stays scoped to this one
project instead of a machine-wide environment variable every other program
on the PC can also read, and there's nothing to re-set every time you open a
new PowerShell window. Two alternatives if you'd rather not add a `.env`
file:
- **Current session only:** `$env:ANTHROPIC_API_KEY = "sk-ant-..."` --
  forgotten the moment you close that PowerShell window.
- **Persistent, machine-wide:** `setx ANTHROPIC_API_KEY "sk-ant-..."` --
  survives across sessions and reboots (stored in the Windows registry
  under your user account), but takes effect only in *new* terminals opened
  after running it, and every program running as you can read it, not just
  this project. The Environment Variables entry in Windows' System
  Properties dialog does the same thing through the GUI.

Open the printed `http://127.0.0.1:8765` (`--command-port` to change it).
Under the hood, a command is turned into one action via a single **forced
Claude tool call** (`command/interpreter.py`): the tool's schema restricts
the choice to an `enum` of whatever `env.action_masks()` says is currently
legal, plus a `cannot_comply` escape hatch. That schema is a strong hint,
not a hard guarantee -- confirmed live, a fast model can still occasionally
name something outside this turn's actual enum (e.g. `build_barracks` when
no supply depot exists yet) -- so `interpret_command()` never trusts the
model's own explanation for that case: an illegal choice always gets
`env.step()` blocked (never executes) *and* has its message replaced with a
ground-truth reason computed straight from `GameState`
(`interpreter.py`'s `_explain_unavailable()`), never the model's original
text, which was written assuming the choice would be honored and would
otherwise read as a false confirmation. This reuses `action_space.py`'s
existing `name()`/`index_for_name()` exactly as its docstring always said a
future "human- or LLM-driven command layer" would: no changes to the action
space, the environment, or the trained model were needed for this feature.

The game does not advance while a command is being interpreted, because
nothing calls `env.step()` until the Claude round-trip resolves -- the
bot-mode client simply waits for the next step request, so a command always
executes against the state the player actually saw. A declined command
doesn't consume a game step either; the loop just falls through to
`model.predict()` on the next iteration as if nothing happened.

**Which model, and where that's set.** Commands are interpreted by
`claude-haiku-4-5-20251001` -- the fastest, cheapest model in the current
Claude lineup, and deliberately chosen: this call is a single forced tool
use picking one item off a short, explicit menu (the current legal-action
list), not open-ended reasoning, so a larger model buys nothing here.
`command/interpreter.py`'s `DEFAULT_MODEL` constant is the single source of
truth for it; override per-run without touching code via
`--command-model claude-sonnet-5` (or any other model ID) on
`interactive_play.py`. `state_summary.py` builds the natural-language state
description Claude sees (minerals, army size, mobilized/garrison status,
which sectors hold known enemies) -- deliberately separate from
`observation.py`'s `featurize()`, whose normalized float vector means
nothing to an LLM.

**v1 scope, by design:** one action per command (matches "return to base"
being a single `move_army_to_sector_N`); no live game-state panel in the
page yet, just a running command log. Both are natural follow-ups once this
is proven out.

**Startup, in order.** There's one command, not two -- `interactive_play.py`
starts the command console (a background thread) and *then* launches
StarCraft II itself (the first `env.reset()`, inside the same process), so
the console is reachable slightly before the game has actually finished
loading. Wait for the game to visibly be playing (the visualized window, or
episode output in the terminal) before sending your first command: a
command sent too early just sits queued until the main loop starts pulling
from it, and if that wait passes the command endpoint's 30-second timeout
(`command/server.py`'s `_COMMAND_TIMEOUT_SECONDS`) the browser gets a
timeout error rather than an answer. There's currently no "game not ready
yet" status shown on the page itself -- a reasonable follow-up if this
turns out to matter in practice.

**How the browser reaches the game.** There is no REST service inside
StarCraft II, and the page never talks to the game directly. Everything --
the environment, the trained model, the Claude client, and the Flask
server -- runs in the one `interactive_play.py` process; the browser only
ever talks to that process's own local Flask endpoint
(`http://127.0.0.1:8765`), which hands commands to the game loop through an
in-process queue (`command/server.py`'s `PendingCommand` + `queue.Queue`).
PySC2 talks to the actual SC2 executable over its own separate local
protocol, untouched by any of this.

**Realtime pacing.** By default, `SC2FightEnv` steps the game as fast as the
client can simulate -- the right choice for training, evaluation, and
demonstration collection, where wall-clock speed doesn't matter. A human
typing or speaking a command needs the opposite: true StarCraft II pacing
(22.4 game loops/second), which is pysc2's own `realtime` mode
(`env.realtime` in `config.py`, plumbed straight through to
`sc2_env.SC2Env`). `interactive_play.py` turns this **on by default** for
exactly that reason -- pass `--no-realtime` to fall back to turbo speed if
you want to stress-test the console without waiting on the clock.
`play.py` keeps it off by default (`--realtime` to turn it on there too),
since that entrypoint is normally used for fast bulk evaluation.

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
| `garrison_size` | `4` | Marines permanently held at the command center, excluded from every move order |
| `masking.min_marines_to_move` | `4` | Movement to the HOME sector illegal below this many marines |
| `masking.min_marines_to_advance` | `20` | Movement to any OTHER sector illegal below this many marines (to launch an offensive) |
| `masking.min_marines_to_continue` | `8` | Once launched ("mobilized"), advancing stays legal down to this many -- hysteresis |
| `reward.shaping_enabled` | `true` | Master on/off switch for everything below |
| `reward.shaping_coefficient` | `0.001` | Scales the army-value and kill-value shaping terms |
| `reward.economic_value_cap` | `4000.0` | Ceiling on economic value used for the reward -- prevents indefinite hoarding |
| `reward.concentration_threshold` | `20` | Marine count for full kill-reward credit; scaled down below it |
| `reward.kill_value_scale` | `0.1` | Additional discount on kill-value credit, on top of concentration scaling |
| `reward.kill_value_cap` | `2000.0` | Ceiling on kill value used for the reward -- kills alone can't grind out unbounded reward |
| `reward.home_defense_penalty` | `0.05` | Per-step penalty while home is undefended and under attack |
| `reward.home_defense_penalty_cap` | `1.0` | Ceiling on total home_defense_penalty accumulated within one episode |
| `reward.scouting_bonus` | `0.02` | One-time reward per newly-sighted enemy sector per episode |
| `reward.exploration_bonus` | `0.1` | One-time reward per sector a marine newly enters per episode -- counterweight to home_defense_penalty |
| `reward.stale_search_penalty` | `0.02` | Per-step penalty once the army stalls without reaching a new sector too long |
| `reward.stale_search_patience` | `20` | Steps of no new-sector progress tolerated before stale_search_penalty kicks in |
| `reward.stale_search_penalty_cap` | `2.0` | Ceiling on total stale_search_penalty accumulated within one episode |
| `reward.time_penalty_per_step` | `0.001` | Flat per-step cost from step one -- finishing sooner is worth more |
| `reward.time_penalty_cap` | `2.0` | Ceiling on total time penalty within one episode |
| `reward.approach_reward_scale` | `2.0` | Potential-based reward for closing distance to the nearest known enemy structure |
| `reward.approach_reward_cap` | `3.0` | Ceiling on the positive and (separately) negative approach totals within one episode |
| `reward.terminal_reward_scale` | `20.0` | Multiplies PySC2's own terminal win/loss reward -- deliberately dominant, see "How it works" |

`SC2FightEnv.step()` also returns each component separately in its `info`
dict (`reward_terminal`, `reward_economic`, `reward_kill`,
`reward_home_defense`, `reward_scouting`, `reward_exploration`,
`reward_stale_search`, summing to the total reward).
`training/callbacks.py`'s `RewardBreakdownCallback` (wired into every
training run by default) accumulates these per episode and prints a line to
the console the moment each episode ends, e.g.:

```
[episode end] total=-20.104  terminal=-20.000 economic=+0.320 kill=+0.004 home_defense=-0.150 scouting=+0.020 exploration=+0.200 stale_search=-0.040 time=-0.850 approach=+0.392
```

It also logs each component to TensorBoard under `reward_breakdown/*`. This
is how to actually see which term is driving an imbalance instead of
reasoning about it from formulas.

`masking.min_marines_to_advance` and `reward.concentration_threshold` are
separate knobs on purpose -- one is a hard action-legality gate, the other a
soft reward scaling -- but they default to the same value (20) since they're
both expressing "this is what counts as a real, committed fighting force"
for this scenario; tune them independently if that stops making sense (e.g.
a larger map where you'd want a bigger minimum force before committing).
`masking.min_marines_to_move` is a separate, lower bar -- it only gates
whether the army can move to the HOME sector at all (e.g. recalling a
scattered force), not whether it's ready to advance elsewhere.
