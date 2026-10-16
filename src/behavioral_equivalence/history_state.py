"""Training-only sampling and initial partitions for pre-exposure history states."""

import numpy as np


def standard_training_sample(case_ids, random_case_ids, sample_size: int, seed: int):
    cases, random = map(np.asarray, (case_ids, random_case_ids))
    if (
        cases.ndim != 1
        or random.ndim != 1
        or cases.dtype.kind not in "iu"
        or random.dtype.kind not in "iu"
        or np.any(cases < 0)
        or np.any(random < 0)
        or len(np.unique(cases)) != len(cases)
        or len(np.unique(random)) != len(random)
        or not isinstance(sample_size, int)
        or sample_size < 1
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ValueError("Unique physical case identifiers and positive sample size required")
    eligible = np.flatnonzero(~np.isin(cases, random))
    if not len(eligible):
        raise ValueError("Standard-policy training exposures required")
    return np.sort(
        np.random.default_rng(seed).choice(eligible, min(sample_size, len(eligible)), replace=False)
    )


def initial_predictive_blocks(probabilities, block_count: int):
    """Equal-parent-count bins sorted by standard-policy teacher means, with stable ties."""
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 1
        or not len(p)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
        or not isinstance(block_count, int)
        or not 1 <= block_count <= len(p)
    ):
        raise ValueError("Finite Bernoulli means and feasible positive block count required")
    ordered = np.argsort(p, kind="stable")
    return tuple(
        tuple(sorted(int(s) for s in block)) for block in np.array_split(ordered, block_count)
    )
