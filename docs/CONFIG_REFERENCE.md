# Config reference

## Key defaults (see `configs/default.yaml`)

- Map: `Simple64` vs. `very_easy` Zerg bot (matches the old repo's baseline,
  both overridable).
- Action space: `no_op`, `build_supply_depot`, `build_barracks`,
  `train_marine`, plus one `move_army_to_sector_i` per grid cell (default
  6x6 = 36 cells) -- 40 actions total.
- Reward: PySC2's own terminal win/loss reward plus dense shaping, on by
  default (see [HOW_IT_WORKS.md](HOW_IT_WORKS.md#reward) for the individual
  terms).
- Single environment (`DummyVecEnv` with one `SC2FightEnv`). StarCraft II's
  per-step client overhead is the real bottleneck, not neural-net compute
  or environment parallelism -- revisit `SubprocVecEnv` (multiple
  concurrent `SC2Env` instances) only if training throughput turns out to
  matter in practice.

## Tuning the reward/masking config knobs

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
| `reward.terminal_reward_scale` | `20.0` | Multiplies PySC2's own terminal win/loss reward -- deliberately dominant, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md#reward) |

## Reading the reward breakdown

- `SC2FightEnv.step()` also returns each component separately in its
  `info` dict (`reward_terminal`, `reward_economic`, `reward_kill`,
  `reward_home_defense`, `reward_scouting`, `reward_exploration`,
  `reward_stale_search`, summing to the total reward).
- `training/callbacks.py`'s `RewardBreakdownCallback` (wired into every
  training run by default) accumulates these per episode and prints a line
  to the console the moment each episode ends, e.g.:

  ```
  [episode end] total=-20.104  terminal=-20.000 economic=+0.320 kill=+0.004 home_defense=-0.150 scouting=+0.020 exploration=+0.200 stale_search=-0.040 time=-0.850 approach=+0.392
  ```

- It also logs each component to TensorBoard under `reward_breakdown/*`.
  This is how to actually see which term is driving an imbalance instead
  of reasoning about it from formulas.

## Notes on specific knobs

- `masking.min_marines_to_advance` and `reward.concentration_threshold`
  are separate knobs on purpose -- one is a hard action-legality gate, the
  other a soft reward scaling -- but they default to the same value (20)
  since they're both expressing "this is what counts as a real, committed
  fighting force" for this scenario; tune them independently if that stops
  making sense (e.g. a larger map where you'd want a bigger minimum force
  before committing).
- `masking.min_marines_to_move` is a separate, lower bar -- it only gates
  whether the army can move to the HOME sector at all (e.g. recalling a
  scattered force), not whether it's ready to advance elsewhere.
