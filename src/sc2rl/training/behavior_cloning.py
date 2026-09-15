"""Supervised behavior-cloning pretraining: initializes a MaskablePPO
policy's weights by imitating a dataset of (observation, action,
action_mask) triples before RL fine-tuning starts, instead of leaving it at
its random initialization. Same spirit as AlphaStar's imitation-learning
warm start -- imitating our own scripted teacher (env/scripted_policy.py)
rather than human replays, since our action space is a simplified custom
abstraction that doesn't correspond to raw replay actions the way
AlphaStar's full-interface action space did.
"""

from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO


def pretrain_with_behavior_cloning(
    model: MaskablePPO,
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
) -> None:
    """The loss is the mean negative log-likelihood of the teacher's action:
    exp(-loss) is the average probability the policy assigns to what the
    teacher did (0.2 -> ~82%). It has a floor well above zero, because the
    teacher's choice depends on state the observation doesn't carry -- its
    own search target and cleared-sector memory -- so the same observation
    can legitimately map to different actions. The held-out loss says
    whether more epochs would help: still falling with the training loss
    means underfit (train longer); flat or rising while the training loss
    keeps falling means the floor has been reached (more epochs only
    memorize the dataset).

    The split is by contiguous block, not a random shuffle: consecutive
    steps of one game are near-duplicates, so a shuffled split leaks the
    training set into the held-out set and reports a flattering number.

    Early stopping: the weights from the epoch with the best held-out loss
    are what the policy ends up with, not the last epoch's. Confirmed
    necessary on a real 52k-sample dataset: held-out loss bottomed at
    ~0.195 around epoch 7 and then climbed to ~0.38 by epoch 30 while the
    training loss kept falling -- RL was warm-starting from an overfit
    policy that was worse than the one from twenty epochs earlier. So
    `epochs` is a budget, not a target."""
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    dataset_size = len(observations)
    validation_size = int(dataset_size * validation_fraction) if dataset_size >= 20 else 0
    train_size = dataset_size - validation_size

    actions_tensor = th.as_tensor(actions, dtype=th.long, device=policy.device)
    masks_tensor = th.as_tensor(masks, dtype=th.bool, device=policy.device)

    def batch_loss(batch_idx) -> th.Tensor:
        obs_batch, _ = policy.obs_to_tensor(observations[batch_idx])
        # evaluate_actions() applies action masking to the distribution
        # before computing log_prob, so an imitated action that would be
        # illegal under the current mask never gets rewarded credit.
        _, log_prob, _ = policy.evaluate_actions(
            obs_batch, actions_tensor[batch_idx], action_masks=masks_tensor[batch_idx],
        )
        return -log_prob.mean()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        permutation = np.random.permutation(train_size)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, train_size, batch_size):
            loss = batch_loss(permutation[start:start + batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        line = f"BC pretrain epoch {epoch + 1}/{epochs}: loss={epoch_loss / max(num_batches, 1):.4f}"
        if validation_size:
            with th.no_grad():
                held_out = np.arange(train_size, dataset_size)
                val_loss = sum(
                    batch_loss(held_out[s:s + batch_size]).item() * len(held_out[s:s + batch_size])
                    for s in range(0, validation_size, batch_size)
                ) / validation_size
            line += f"  held_out_loss={val_loss:.4f}"
            if val_loss < best_val_loss:
                best_val_loss, best_epoch = val_loss, epoch + 1
                best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                line += "  (best so far)"
        print(line)

    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"BC pretrain: keeping epoch {best_epoch} weights (held_out_loss={best_val_loss:.4f})")
