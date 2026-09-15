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
) -> None:
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    dataset_size = len(observations)

    actions_tensor = th.as_tensor(actions, dtype=th.long, device=policy.device)
    masks_tensor = th.as_tensor(masks, dtype=th.bool, device=policy.device)

    for epoch in range(epochs):
        permutation = np.random.permutation(dataset_size)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, dataset_size, batch_size):
            batch_idx = permutation[start:start + batch_size]
            obs_batch, _ = policy.obs_to_tensor(observations[batch_idx])
            actions_batch = actions_tensor[batch_idx]
            masks_batch = masks_tensor[batch_idx]

            # evaluate_actions() applies action masking to the distribution
            # before computing log_prob, so an imitated action that would be
            # illegal under the current mask never gets rewarded credit.
            _, log_prob, _ = policy.evaluate_actions(obs_batch, actions_batch, action_masks=masks_batch)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        print(f"BC pretrain epoch {epoch + 1}/{epochs}: loss={epoch_loss / max(num_batches, 1):.4f}")
