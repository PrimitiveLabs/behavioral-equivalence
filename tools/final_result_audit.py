"""Independent sorted-user proper-loss arithmetic for saved final forecasts."""

import numpy as np


def loss_block(users, targets, probabilities):
    u, y, p = map(np.asarray, (users, targets, probabilities))
    if (
        u.ndim != 1
        or u.dtype.kind not in "iu"
        or np.any(u < 0)
        or y.shape != u.shape
        or p.ndim not in [1, 2]
        or p.shape[0] != len(u)
        or (p.ndim == 2 and p.shape[1] < 1)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError(
            "Aligned original users, binary targets and finite forecast replicas required"
        )
    matrix = p[:, None] if p.ndim == 1 else p
    clipped = np.clip(matrix.astype(np.float64), 1e-6, 1 - 1e-6)
    losses = -(y[:, None] * np.log(clipped) + (1 - y[:, None]) * np.log1p(-clipped)).mean(axis=1)
    squared = ((matrix - y[:, None]) ** 2).mean(axis=1)
    order = np.argsort(u, kind="stable")
    sorted_users = u[order]
    starts = (
        np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
        if len(u)
        else np.empty(0, dtype=int)
    )
    counts = np.diff(np.r_[starts, len(u)]) if len(u) else np.empty(0, dtype=int)
    return dict(
        users=sorted_users[starts],
        counts=counts,
        nll=np.add.reduceat(losses[order], starts) / counts if len(u) else np.empty(0),
        brier=np.add.reduceat(squared.astype(float)[order], starts) / counts
        if len(u)
        else np.empty(0),
        nll_sum=float(losses.sum()),
        brier_sum=float(squared.sum()),
    )


def combine(blocks):
    users = np.concatenate([b["users"] for b in blocks])
    counts = np.concatenate([b["counts"] for b in blocks])
    if len(np.unique(users)) != len(users):
        raise ValueError("Original users cross audit buckets")
    order = np.argsort(users)
    nll = np.concatenate([b["nll"] for b in blocks])[order]
    brier = np.concatenate([b["brier"] for b in blocks])[order]
    n = int(counts.sum())
    return dict(
        users=users[order],
        counts=counts[order],
        nll=nll,
        brier=brier,
        metrics=dict(
            cases=n,
            users=len(users),
            nll_case_mean=sum(b["nll_sum"] for b in blocks) / n if n else None,
            brier_case_mean=sum(b["brier_sum"] for b in blocks) / n if n else None,
            nll_equal_user_mean=float(nll.mean()) if len(users) else None,
            brier_equal_user_mean=float(brier.mean()) if len(users) else None,
        ),
    )


def check_metrics(actual, expected):
    if actual["cases"] != expected["cases"] or actual["users"] != expected["users"]:
        raise ValueError("Replayed original case/user support differs")
    differences = {}
    for key in ["nll_case_mean", "nll_equal_user_mean", "brier_case_mean", "brier_equal_user_mean"]:
        a, e = actual[key], expected[key]
        if a is None or e is None:
            if a != e:
                raise ValueError("Empty-cohort metric differs")
            differences[key] = 0.0
            continue
        if not np.isfinite(a) or not np.isfinite(e):
            raise ValueError("Finite reported and replayed proper losses required")
        difference = abs(a - e)
        tolerance = 1e-10 if key.startswith("nll") else 1e-6
        if difference > tolerance:
            raise ValueError("Replayed proper loss differs: " + key)
        differences[key] = difference
    return differences


def bootstrap_primary(differences, *, replications, seed):
    x = np.asarray(differences, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite original-user loss differences required")
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, replications, 32):
        indices = rng.integers(0, len(x), (min(32, replications - start), len(x)))
        values.extend(x[indices].mean(axis=1))
    return dict(mean=float(x.mean()), ci95=np.quantile(values, [0.025, 0.975]).tolist())
