# Training

See [SETUP.md](SETUP.md) first if you haven't installed dependencies yet,
and [HOW_IT_WORKS.md](HOW_IT_WORKS.md) for why the PPO settings below
differ from Stable-Baselines3's own defaults.

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

## Checkpoints and resuming

- Checkpoints land in `training.checkpoint_dir` (`checkpoints/` by
  default, `.gitignore`d) as `.zip` files -- each one is a full snapshot:
  policy + value network weights, the optimizer's state, and the
  hyperparameters used to train it.
- One is saved every `training.checkpoint_freq` timesteps, plus a
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

## Imitation-learning warm start (behavior cloning)

Same spirit as AlphaStar's imitation-learning bootstrap from human replays,
adapted to what our action space actually supports: AlphaStar imitated raw
replay actions (mouse/camera/hotkeys) because its action space *was* that
full interface. Ours is a simplified custom abstraction with no direct
mapping from human replay data, so the practical equivalent is a
**scripted teacher** (`env/scripted_policy.py`) operating in our own action
space.

**Scripted teacher behavior:**

- Build a small base, keep training marines, recall the whole offensive
  force home only if a sector with one of our buildings is under a threat
  bigger than the standing garrison can handle on its own (`garrison_size`,
  passed in by the caller -- a single stray unit near home no longer
  interrupts the search, which previously made the army destroy a few
  enemies elsewhere and then abandon a real, more distant target every
  time it happened).
- Once mobilized to a real attack force (`ScriptedPolicyConfig.attack_threshold`,
  default 20 marines, matching `masking.min_marines_to_advance`), commit
  to a **search-and-destroy** pattern. Below the threshold it holds at the
  base.
- **Hysteresis once launched**: the env remembers the army reached
  `masking.min_marines_to_advance` ("mobilized"), the movement mask then
  keeps advancing legal down to `masking.min_marines_to_continue` (default
  8), and the teacher keeps pressing as long as the mask allows. Without
  that, an army that launched at 20 and lost a few marines was forbidden
  from going anywhere but home and walked away from the enemy's last two
  buildings (observed live -- that game was only won because a couple of
  stragglers never got the recall).
- Only regroups at the command center once the mask closes, rather than
  holding mid-map -- holding mid-map meant getting picked off while
  reinforcements piled up at home, observed live as the army parked at the
  map's center for a very long time.
- **Search pattern** (over legal targets only -- sectors with no pathable
  ground are never picked): sweep sectors farthest-from-home first,
  redirecting immediately to any sector where the enemy is actually
  spotted.
- A new order only fires once the majority of marines are idle
  (`UnitInfo.is_idle` -- no active order, so neither mid-fight nor still
  traveling); while busy, the teacher issues `no_op`, which doesn't
  interrupt existing orders. Confirmed live, this matters for two reasons:
  reissuing move orders while marines were still mid-approach or mid-fight
  was part of why the group would scatter instead of staying clustered, and
  an exact-grid-cell "arrival" check meant the search could sit idle for a
  long time on an already-cleared area before timing out -- the idle
  signal reacts immediately instead.
- `search_timeout_steps` (default 60) is now just a safety net in case
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

**Reading the BC loss:**

- It's the mean negative log-likelihood of the teacher's action, so
  `exp(-loss)` is the average probability the policy gives to what the
  teacher did: 0.2 means ~82%.
- It does not go to zero and shouldn't be expected to -- the teacher's
  choice depends on its own hidden state (current search target,
  cleared-sector memory), so the same observation can legitimately map to
  different actions.
- Each epoch also prints `held_out_loss` on a ~10% held-out slice of the
  dataset that was never trained on (never a random shuffle: consecutive
  steps of one game are near-duplicates, so a shuffled split leaks and
  flatters). That number answers "should I train more epochs?": if it's
  still falling alongside the training loss, yes (`--bc-epochs 30`); if
  it's flat or rising while the training loss keeps dropping, the floor
  has been reached and more epochs would only memorize the dataset.
- The loss is a proxy anyway -- what matters is whether the pretrained
  policy plays like the teacher, which `sc2rl.inference.play` on the saved
  model shows directly.

**Held-out split is whole trailing episodes, not a raw slice of the
concatenated array**, and spans at least two of them whenever there are
enough to spare (`training/behavior_cloning.py`'s
`_episode_validation_split`, using the `episode_ids` array
`collect_demonstrations.py` saves alongside the observations). This matters
more than it sounds like it should: confirmed live, one unusually long
episode by itself exceeded an entire 10%-of-samples target, so a plain tail
slice landed inside that one game -- the reported held-out loss was really
just "how well does the policy predict this specific game," and it picked
epoch 1 as best while later epochs' held-out loss climbed from 0.50 to over
1.3, nothing like the clean bottom-then-rise curve a genuine multi-game
split gives. A dataset saved before `episode_ids` existed (or with fewer
than 3 distinct episodes) falls back to the old plain tail slice
automatically.

**Dataset/checkpoint compatibility:** a dataset is tied to the observation
layout it was collected under -- any change to `observation.py`'s features
or to `env.grid` changes the vector width, and `train.py` will refuse the
stale dataset with a message saying to re-collect. Old checkpoints are
likewise incompatible after such a change (the network's input layer is
sized to the observation), so start a fresh run rather than `--resume-from`.

## After training

```powershell
python -m sc2rl.inference.play --checkpoint checkpoints/final_model --episodes 5 --visualize
```

For natural-language control of the trained agent while it plays, see
[COMMAND_CONSOLE.md](COMMAND_CONSOLE.md).
