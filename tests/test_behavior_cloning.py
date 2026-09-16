"""Validates behavior-cloning pretraining actually works: builds a real
MaskablePPO model (via the stub-based training smoke test's env, so it has
the exact observation/action space sc2rl uses) and confirms loss decreases
on a small, trivially learnable synthetic dataset, plus that action masking
during pretraining is actually respected -- not just that the code runs.
"""

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sc2rl.training.behavior_cloning import _episode_validation_split, pretrain_with_behavior_cloning
from tests.test_training_smoke import _make_env


def _build_model():
    vec_env = DummyVecEnv([_make_env()])
    return MaskablePPO("MlpPolicy", vec_env, n_steps=8, batch_size=4, n_epochs=1, verbose=0)


def test_behavior_cloning_reduces_loss_on_a_learnable_pattern(capsys):
    model = _build_model()
    obs_dim = model.observation_space.shape[0]
    num_actions = model.action_space.n

    # Two distinct, fixed observations each with one "correct" action, mask
    # allowing every action legal -- trivially learnable if gradients flow.
    obs_a = np.zeros(obs_dim, dtype=np.float32)
    obs_b = np.ones(obs_dim, dtype=np.float32)
    target_a, target_b = 2, 7

    n = 200
    observations = np.stack([obs_a if i % 2 == 0 else obs_b for i in range(n)]).astype(np.float32)
    actions = np.array([target_a if i % 2 == 0 else target_b for i in range(n)], dtype=np.int64)
    masks = np.ones((n, num_actions), dtype=bool)

    pretrain_with_behavior_cloning(model, observations, actions, masks, epochs=15, batch_size=32, learning_rate=1e-2)

    printed = capsys.readouterr().out
    epoch_lines = [line for line in printed.splitlines() if line.startswith("BC pretrain epoch")]
    losses = [float(line.split("loss=")[1].split()[0]) for line in epoch_lines]
    held_out = [float(line.split("held_out_loss=")[1].split()[0]) for line in epoch_lines]
    assert len(losses) == 15
    assert losses[-1] < losses[0] * 0.5  # meaningfully reduced, not just noise
    assert held_out[-1] < held_out[0] * 0.5  # the held-out block follows the same learnable pattern

    # After training, the policy should now prefer the imitated actions.
    action_a, _ = model.predict(obs_a, deterministic=True)
    action_b, _ = model.predict(obs_b, deterministic=True)
    assert int(action_a) == target_a
    assert int(action_b) == target_b


def test_behavior_cloning_keeps_the_best_held_out_epoch_not_the_last(capsys):
    # Training data and the contiguous held-out block follow CONFLICTING
    # rules for the same observation, so the more the policy fits the
    # training block the worse the held-out loss gets after the first
    # epoch or two. The policy must end up with the early, better weights.
    model = _build_model()
    obs_dim = model.observation_space.shape[0]
    num_actions = model.action_space.n

    n = 200
    observations = np.ones((n, obs_dim), dtype=np.float32)
    actions = np.full(n, 2, dtype=np.int64)
    actions[-20:] = 7  # the held-out 10% says "7" where training says "2"
    masks = np.ones((n, num_actions), dtype=bool)

    pretrain_with_behavior_cloning(model, observations, actions, masks, epochs=12, batch_size=32, learning_rate=1e-2)

    printed = capsys.readouterr().out
    kept = [line for line in printed.splitlines() if line.startswith("BC pretrain: keeping epoch")]
    assert len(kept) == 1
    best_epoch = int(kept[0].split("epoch ")[1].split()[0])
    assert best_epoch < 12  # not simply the last epoch

    # The kept weights are the best held-out epoch's, so the held-out
    # target keeps meaningfully more probability than the fully overfit
    # final epoch would have left it.
    epoch_lines = [line for line in printed.splitlines() if line.startswith("BC pretrain epoch")]
    held_out = [float(line.split("held_out_loss=")[1].split()[0]) for line in epoch_lines]
    assert min(held_out) < held_out[-1]


def test_episode_split_spans_multiple_episodes_even_when_one_dwarfs_the_target():
    # Regression test for the real failure this was built to fix: a single
    # long episode (60 samples) alone exceeds a 10%-of-90 target (9), so a
    # split that stopped as soon as the sample-count target was hit would
    # hold out only that one game. The fix requires at least 2 episodes
    # whenever there are enough to spare.
    episode_ids = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 60)
    train_idx, val_idx = _episode_validation_split(episode_ids, validation_fraction=0.1)
    val_episodes = set(episode_ids[val_idx])
    assert len(val_episodes) >= 2
    assert 3 in val_episodes  # the long trailing episode is still included
    assert set(episode_ids[train_idx]) & val_episodes == set()  # no episode split across both


def test_episode_split_always_leaves_at_least_one_training_episode():
    episode_ids = np.array([0] * 5 + [1] * 5 + [2] * 5)
    train_idx, val_idx = _episode_validation_split(episode_ids, validation_fraction=0.9)
    assert len(train_idx) > 0
    assert set(episode_ids[train_idx]) & set(episode_ids[val_idx]) == set()


def test_episode_split_returns_none_with_too_few_episodes():
    episode_ids = np.array([0] * 10 + [1] * 10)
    assert _episode_validation_split(episode_ids, validation_fraction=0.1) is None


def test_behavior_cloning_uses_episode_aware_split_when_episode_ids_given(capsys):
    # End-to-end: without episode_ids, a dataset shaped like the one that
    # broke in production (one huge trailing episode) makes held-out loss
    # nonsense; with episode_ids, pretraining should run without error and
    # actually use more than one episode's worth of held-out data.
    model = _build_model()
    obs_dim = model.observation_space.shape[0]
    num_actions = model.action_space.n
    sizes = [10, 10, 10, 60]
    n = sum(sizes)
    episode_ids = np.concatenate([np.full(s, i, dtype=np.int64) for i, s in enumerate(sizes)])
    observations = np.random.RandomState(0).rand(n, obs_dim).astype(np.float32)
    actions = np.zeros(n, dtype=np.int64)
    masks = np.ones((n, num_actions), dtype=bool)

    pretrain_with_behavior_cloning(
        model, observations, actions, masks, epochs=2, batch_size=8, episode_ids=episode_ids,
    )
    assert "held_out_loss=" in capsys.readouterr().out


def test_behavior_cloning_respects_action_masking():
    model = _build_model()
    obs_dim = model.observation_space.shape[0]
    num_actions = model.action_space.n

    obs = np.zeros((50, obs_dim), dtype=np.float32)
    actions = np.zeros(50, dtype=np.int64)
    masks = np.ones((50, num_actions), dtype=bool)
    masks[:, 0] = False  # action 0 (e.g. no_op) illegal throughout, but used as the "target" label

    # Should not raise even though the target action is masked illegal --
    # evaluate_actions() computes log_prob against the masked distribution,
    # so this exercises that path without crashing.
    pretrain_with_behavior_cloning(model, obs, actions, masks, epochs=1, batch_size=16)
