"""Label-preserving, uncertified compression baselines at fixed state counts."""

import numpy as np
from .core import Chain, Partition, label_partition, reachability


def _allocate(chain: Chain, k: int) -> tuple[Partition, list[int]]:
    groups = label_partition(chain)
    if not len(groups) <= k <= chain.size:
        raise ValueError("State count must cover labels and not exceed concrete size")
    quotas = [1] * len(groups)
    while sum(quotas) < k:
        eligible = [i for i, g in enumerate(groups) if quotas[i] < len(g)]
        chosen = max(eligible, key=lambda i: len(groups[i]) / quotas[i])
        quotas[chosen] += 1
    return groups, quotas


def random_partition(chain: Chain, k: int, seed: int = 0) -> Partition:
    groups, quotas = _allocate(chain, k)
    rng = np.random.default_rng(seed)
    return tuple(
        tuple(int(s) for s in block)
        for group, n in zip(groups, quotas)
        for block in np.array_split(rng.permutation(group), n)
    )


def feature_partition(
    chain: Chain, features: np.ndarray, k: int, seed: int = 0, iterations: int = 100
) -> Partition:
    """Lloyd clustering within labels; no equivalence certificate is implied."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] != chain.size or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("One finite nonempty feature vector per concrete state required")
    if iterations < 1:
        raise ValueError("Positive iteration count required")
    groups, quotas = _allocate(chain, k)
    rng = np.random.default_rng(seed)
    blocks = []
    for group, n in zip(groups, quotas):
        observations = x[list(group)]
        centers = observations[rng.choice(len(group), n, replace=False)].copy()
        previous = None
        for _ in range(iterations):
            distance = ((observations[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assignment = distance.argmin(axis=1)
            counts = np.bincount(assignment, minlength=n)
            for empty in np.flatnonzero(counts == 0):
                eligible = counts[assignment] > 1
                scores = np.where(eligible, distance[np.arange(len(group)), assignment], -np.inf)
                moved = int(scores.argmax())
                counts[assignment[moved]] -= 1
                assignment[moved] = empty
                counts[empty] += 1
            centers = np.vstack([observations[assignment == i].mean(axis=0) for i in range(n)])
            if previous is not None and np.array_equal(assignment, previous):
                break
            previous = assignment.copy()
        blocks.extend(
            tuple(group[int(s)] for s in np.flatnonzero(assignment == i)) for i in range(n)
        )
    return tuple(blocks)


def predictive_features(chain: Chain, horizons: tuple[int, ...] = (1, 5, 10, 25)) -> np.ndarray:
    if not horizons:
        raise ValueError("At least one predictive horizon required")
    return np.column_stack(
        [
            reachability(chain, target, horizon)
            for horizon in horizons
            for target in ("complete", "disengage")
        ]
    )
