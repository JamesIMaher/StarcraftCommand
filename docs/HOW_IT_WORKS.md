# How it works

Design rationale for the environment, action space, and training setup --
most of it hard-won from live runs against a real StarCraft II client, not
just from reading the PySC2 docs. See the [README](../README.md) for setup
and quick-start commands, [TRAINING.md](TRAINING.md) for running training
itself, and [COMMAND_CONSOLE.md](COMMAND_CONSOLE.md) for the interactive
natural-language layer.

## Observation

- `GameState.from_observation()` parses PySC2's `raw_units`/`player` fields
  into a per-step snapshot; `observation.py` turns that into a flat
  `float32` vector.
- Layout: 20 global scalars (minerals, supply used/cap/headroom, marine
  count + average health + idle fraction, SCV count, supply depot/barracks
  counts -- in-progress and complete, visible enemy count + health, enemy
  race one-hot, episode-progress fraction) plus 6 features per grid sector
  for a 6x6 grid -- **236 floats total** at the default grid size
  (`env.grid`, configurable). This is a `gymnasium.spaces.Box(0.0, 1.0,
  shape=(236,))`.
- **Idle fraction** ("has the army finished what it was told to do") exists
  because the scripted teacher's reorder rule is majority-idle: without it,
  the imitation target depended on something the policy couldn't see (a
  hard floor on the BC loss), and the RL policy couldn't tell an army
  mid-move from one standing around.
- Four of the six per-sector features are a snapshot of the current step
  (friendly/enemy count and average health in that sector); the other two
  are **minimap memory**:
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
    it had never looked at, so it couldn't do a systematic search --
    observed live as the army leaving enemy buildings standing in sectors
    it simply never visited.
- Changing the observation layout invalidates any previously collected BC
  dataset; `train.py` refuses a `--bc-dataset` whose observation width
  doesn't match the current environment rather than silently training on
  garbage.

## Home-relative sectors

- Maps like `Simple64` randomize which corner each side spawns in between
  episodes (confirmed empirically: resets within the same process landed
  the command center in different quadrants across consecutive games).
- Both the observation's per-sector features and the
  `move_army_to_sector_i` actions are computed in a *canonical,
  home-relative* coordinate frame (`sector_grid.SpawnOrientation`, mirrored
  once per episode from the starting command center position) rather than
  raw map coordinates -- so "sector 0" always means "near home" and the
  farthest sector always means "toward the enemy," in every episode,
  regardless of which corner you actually spawned in.
- Without this, the same action index meant opposite things in different
  games and the policy could never learn a stable explore-toward-the-enemy
  or return-to-defend behavior.

## Action space

- A flat `Discrete(40)`: `no_op`, `build_supply_depot`, `build_barracks`,
  `train_marine`, plus one `move_army_to_sector_i` per grid cell (36 at the
  default 6x6 grid). The grid is laid over the map's *playable area* (see
  below), which on Simple64 is a ~43x43 square in the raw frame.
- **Grid resolution is a real coverage/thoroughness tradeoff, not just a
  display detail**: the original 4x4 grid over the full 64x64 square had
  16x16 cells with a half-diagonal (~11.3) beyond a marine's sight range
  (~9) -- confirmed live as the cause of enemy buildings tucked in a cell
  corner going undetected during search-and-destroy (see
  `env/scripted_policy.py`). 6x6 over the playable area gives ~7.1x7.1
  cells (half-diagonal ~5), so standing at a cell's center sees the whole
  cell; going finer still improves coverage further but costs more sectors
  to sweep within an episode.
- `action_masking.py` computes which actions are legal each step (afford
  checks, unit existence, per-type caps) -- illegal actions never get
  sampled at all rather than resolving to a silent no-op, because
  `MaskablePPO` zeroes out their probability directly in the action
  distribution before sampling.
- **Mass / Concentration of Force**: movement/attack actions stay illegal
  until `masking.min_marines_to_move` marines exist -- newly trained
  marines spawn near home, so while blocked they default to passive
  defense there rather than being committed piecemeal.
- Moving to any sector *other than home* needs a separate, higher
  `masking.min_marines_to_advance` (default `20`, matching
  `scripted_policy.py`'s own `attack_threshold`): without this as a hard
  mask, nothing stopped the RL policy from sending the whole army into
  unexplored territory with only `min_marines_to_move` marines -- confirmed
  live as marines exploring too early with too few marines and dying
  immediately, which then taught the policy to avoid moving at all rather
  than to wait for mass.
- Moving *to* home (e.g. recalling a scattered force, or defending) still
  only needs the lower `min_marines_to_move` bar.

## Garrison

- Every move order -- whichever sector it targets -- is only ever given to
  the army *minus* a small standing garrison (`env.garrison_size`, default
  4): those marines are held at the command center and simply excluded
  from the sweep, so an offensive can never strip the base of every
  defender.
- This isn't a policy choice (the action space didn't change); it's
  enforced by `SC2FightEnv._update_garrison()` underneath whatever action
  gets chosen, so it applies identically to the scripted teacher and the
  RL-trained policy without touching the BC dataset or the observation.
- Membership is a stable set of marine tags -- nearest to the command
  center, topped up from the next-nearest survivor as garrisoned marines
  die -- not reassigned from scratch each step, so the same few marines
  hold the position instead of churning.
- **Why it exists**: every RL episode observed before it was added showed
  the home-defense penalty maxed out, win or loss -- the single "move the
  whole army" action structurally could not hedge between offense and
  defense, so a long game's inevitable opportunistic raid on an empty base
  was often what actually decided a loss the search-and-destroy should
  have won outright.

## Where a move actually goes

- `action_translation.py` turns `move_army_to_sector_i` into one
  attack-move per non-garrisoned marine, with a little jitter so they don't
  all path to the identical point.
- If the sector holds a *known enemy structure* (visible, or a fog snapshot
  of one seen earlier), the target is the structure itself -- the one
  nearest the army -- so "go to the sector with the building" resolves to
  "go to the building."
- A move to the command center's sector targets the command center itself,
  so a recall gathers the army at the base rather than at the cell's
  midpoint. "Home" is never a hard-coded sector index: the recall sector
  follows the command center (`sector_grid.home_sector`), and "the base"
  for defense purposes is every sector with one of our buildings in it --
  with ~7-unit cells the base straddles several, and a hard-coded sector 0
  silently missed attacks on the barracks next door (confirmed live as the
  scripted teacher never returning to defend, and losing most games, right
  after the grid was laid over the playable area).
- Otherwise it's the *reachable* point nearest the sector's center: the
  minimap `pathable` feature layer (`env/pathing.py`), which at minimap
  size == raw resolution is in the exact frame unit positions use,
  flood-filled from the command center so only ground actually connected
  to the base counts. Both halves matter:
  - The playable rectangle contains unpathable terrain -- cliffs, and on
    Simple64 the two corners that aren't bases -- and a sector center on a
    cliff sent the army to park at the cliff's edge.
  - *Pathable* isn't *reachable*: the layer marks cliff-top plateaus and
    isolated pockets as pathable even with no ramp to them, so "nearest
    pathable cell" still picked ground the army could never stand on (both
    confirmed live as marines "trying to reach areas they can't reach").
  - *Reachable* still isn't enough: Simple64 is two plateaus with cliff
    lines between them, and the Euclidean-nearest reachable cell to a point
    on a plateau is often at the *foot* of the cliff directly below it --
    legal, reachable, wrong level. Marines sent there stand under the cliff
    with no vision of the building above, and since that building never
    dies the teacher re-targets the sector forever (confirmed live as the
    army piling up under a different map edge each game).
  - **Fix**: a known enemy structure's target is snapped to the nearest
    reachable ground *on the same terrain level* as the structure, using
    the minimap `height_map` layer (its own footprint is unpathable), and a
    sector's sweep target prefers interior cells over cliff-edge cells,
    which are ambiguous about which level they belong to.
- A sector with no reachable ground at all is removed from the action space
  for the episode (`SC2FightEnv.unreachable_sectors`) and pre-marked
  explored so no exploration incentive points at it; the env prints the
  reachable cell count and any such sectors once at reset (`[env] pathing:
  ...`), or a warning if the observation has no pathable layer.
- Every target is also clamped strictly inside the game's own
  `playable_area` (read from `SC2Env.game_info` at reset), because the
  playable area is inset from the nominal 64x64 square: an earlier version
  biased edge-sector targets all the way to the literal map corner to reach
  corner buildings on the old 4x4 grid, and since that point is unpathable
  the attack-move never completed -- the whole army would park at the
  corner cliff for the rest of the game.
- **Coordinate-frame subtlety**: `raw_units` positions and the `world`
  argument of raw actions are *not* world coordinates. PySC2 scales them by
  `raw_resolution / max(map width, height)` and flips the y axis
  (`features.Features.init_camera`), and applies the inverse to raw
  actions, so observations and actions agree with each other -- but
  `game_info` (map size, playable area) is in real world coordinates and
  must be pushed through the same transform before it can be compared to
  anything else.

## The grid is laid over the playable area, not the map square

- Simple64 is actually 88x96 world units with a playable area of
  (12,12)-(76,76); in the raw frame that is `x 8.0..50.7, y 13.3..56.0` --
  a ~43-unit square sitting off-center in the 64x64 square.
- A grid over the full square wasted the entire last column and first row
  on sectors nothing could ever reach, so their `explored` flag stayed 0
  forever and the exploration/stale-search incentives kept pulling the army
  toward those edges (confirmed live as the army bunching at a map edge).
- At reset the env reads the playable area from `game_info`, converts it to
  the raw frame, and re-lays the sector grid over it (`SectorGrid.bounds`)
  -- the number of sectors, and so the action and observation space sizes,
  never change, only the geometry.
- Home-relative mirroring reflects about the playable area's center for the
  same reason.
- The env prints the bounds it used once (`[env] sector grid laid over
  ...`).

## Neural network

- `MaskablePPO`'s default `MlpPolicy`
  (`sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy`) is two
  small separate PyTorch MLPs reading the same 236-dim input:
  - **policy head**: 2 hidden layers x 64 units, `Tanh` activation,
    outputting 40 logits -> the per-action probabilities after masking.
  - **value head**: same shape, outputting a single scalar -- the
    estimated value of the current state.
- No shared trunk and no CNN/spatial convolution -- the input is already
  the flat engineered feature vector above, not raw pixels, so a small MLP
  is all that's needed.

## Training algorithm (PPO)

Each training iteration:

1. Roll out `n_steps` (2048 by default) actions in the live environment
   using the current policy, recording
   observations/actions/rewards/masks/value estimates.
2. Compute advantages via Generalized Advantage Estimation (GAE, `gamma` /
   `gae_lambda`).
3. Run `n_epochs` passes of minibatch (`batch_size`) gradient descent over
   that rollout with the Adam optimizer, on the PPO clipped surrogate loss
   (`clip_range` limits how far a single update can move the policy) plus a
   value-function loss and an entropy bonus (`ent_coef`, encourages
   exploration).

This is what actually updates the PyTorch weights -- `model.learn()` in
`src/sc2rl/training/train.py` runs this loop, and it's the same
net_arch/algorithm regardless of CPU or GPU (`sb3-contrib` picks the device
automatically via `device="auto"`).

The PPO settings in `configs/default.yaml` deliberately differ from SB3's
defaults because of two things about this problem: episodes have no step
limit and run thousands of steps, and the policy starts from a
behavior-cloned initialization worth preserving.

- `n_steps` is 2048 (a 256-step rollout almost never contained an episode
  end, so every update was bootstrapping from a mid-game value estimate).
- `gamma` is 0.995 (at 0.99 the ~100-step discount horizon shrank the
  +-10 terminal reward to ~0.05 by the time credit reached the decisions
  that decide a game -- smaller than the immediate dense shaping terms,
  which the policy then optimized instead).
- `learning_rate` is 1e-4 (at 3e-4 fine-tuning was observed to win a few
  early episodes and then drift steadily worse).
- `ent_coef` is 0.02 -- the built-in "randomization": it penalizes a
  peaked action distribution so the policy keeps sampling alternatives. If
  a run collapses into a do-nothing local minimum, check
  `train/entropy_loss` in the console output -- trending toward 0 means the
  policy has become near-deterministic and stopped exploring, and raising
  `ent_coef` is the first lever.

## Reward

PySC2's own terminal win/loss reward (+1/-1/0), passed straight through,
plus optional dense per-step shaping (`env.reward.shaping_enabled`, on by
default) built from real PySC2 signals -- `obs.observation.score_cumulative`
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
  present and no marines there to respond, capped per episode at
  `home_defense_penalty_cap` (default `1.0`). Episodes have no step limit
  (only PySC2's own game-end conditions), so an uncapped version of this
  could accumulate for hundreds or thousands of steps in an unusually long
  episode -- confirmed live: an early, untrained episode's `ep_rew_mean`
  reached **-46.4** after only 18,000 timesteps, far beyond anything the
  terminal-reward-scale analysis below accounted for, because that analysis
  only ever reasoned about a single-step transition, never an
  episode-length accumulation of an uncapped per-step penalty.
- **Scouting bonus** (OODA loop -- Observe): a one-time reward the first
  time an enemy unit is seen in a given sector during an episode, rewarding
  exploration itself rather than only its downstream combat consequences.
- **Exploration bonus** (default `0.1`): a one-time reward the first time
  a friendly marine is present in a given sector during an episode (the
  base's sectors count as already visited at spawn, and so do sectors with
  no pathable ground), independent of whether an enemy is there. Raised
  from 0.02 after the policy settled into staying home with 20+ marines: at
  0.02 a new sector was worth less than half of one marine (+0.05), so
  exploring could never compete with sitting still. This exists
  specifically as a counterweight to the home-defense penalty: that penalty
  fires every step the home sector is undefended, for as long as that
  holds, with no cap, while `scouting_bonus` only pays once per sector and
  economic reward stops once `economic_value_cap` is hit. Without a
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

**Terminal reward scale keeps outcome dominant.**

- Even with every component capped, their *sum* can still reach a few
  points of reward regardless of outcome -- capping bounds each channel,
  but only the terminal term actually guarantees a win's total stays
  positive and a loss's stays negative.
- `reward.terminal_reward_scale` (default `20.0`) is deliberately set well
  above 1: with the current caps, worst-case POSITIVE shaping per episode
  is roughly `shaping_coefficient * (economic_value_cap + kill_value_scale
  * kill_value_cap) + (scouting_bonus + exploration_bonus) * num_sectors +
  approach_reward_cap` ~= 11.5, and worst-case NEGATIVE shaping is roughly
  `-(shaping_coefficient * economic_value_cap + home_defense_penalty_cap +
  stale_search_penalty_cap + time_penalty_cap + approach_reward_cap)` ~=
  -12 (economic value has no further downside once it's dropped to zero;
  killed value never decreases, so it has no negative side at all).
- 20x here (+-20) comfortably dominates both directions with margin,
  regardless of how long an episode runs or how much shaping it racks up
  either way.
- Tested directly against the actual shipped defaults
  (`test_default_config_guarantees_any_win_outscores_a_heavily_shaped_loss`,
  plus `test_win_reward_stays_positive_despite_a_long_troubled_episode` for
  the specific failure mode of an unbounded-length episode -- the earlier,
  single-step version of this test could not have caught `ep_rew_mean`
  reaching -46.4 in one long early episode, since `home_defense_penalty`
  and `stale_search_penalty` are both per-step terms that only overrun
  their intended bound across many steps, not within a single one),
  specifically to catch this class of imbalance if the caps/scale are ever
  retuned again.
- Shaping applies on every step, **including the terminal one** -- a loss
  typically means the base/army gets wiped out right at the end, so
  computing the economic-value delta there too is what actually charges the
  agent for that collapse; skipping it (an earlier bug) meant reward
  already banked from building an economy earlier in the game was never
  offset by the final defeat.
