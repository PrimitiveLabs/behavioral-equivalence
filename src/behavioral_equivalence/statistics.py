"""Paired user-clustered predictive comparisons with declared aggregation."""

import numpy as np


def prediction_scores(targets: np.ndarray, probabilities: np.ndarray, users: np.ndarray) -> dict:
    """Point metrics for a fixed sample, with equal-user primary aggregation."""
    y, p, u = map(np.asarray, (targets, probabilities, users))
    if (
        y.ndim != 1
        or p.shape != y.shape
        or u.shape != y.shape
        or not len(y)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError("Aligned nonempty binary targets, probabilities and user IDs required")
    unique, inverse = np.unique(u, return_inverse=True)
    counts = np.bincount(inverse)
    weights = 1 / (len(unique) * counts[inverse])
    clipped = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    nll = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    brier = (p - y) ** 2
    bins = np.minimum((p * 20).astype(int), 19)
    calibration = []
    for b in range(20):
        chosen = bins == b
        mass = float(weights[chosen].sum())
        calibration.append(
            {
                "lower": b / 20,
                "upper": (b + 1) / 20,
                "cases": int(chosen.sum()),
                "users": int(len(np.unique(u[chosen]))),
                "equal_user_weight": mass,
                "predicted": float(np.sum(weights[chosen] * p[chosen]) / mass) if mass else None,
                "observed": float(np.sum(weights[chosen] * y[chosen]) / mass) if mass else None,
            }
        )
    return {
        "cases": len(y),
        "users": len(unique),
        "nll_case_mean": float(nll.mean()),
        "brier_case_mean": float(brier.mean()),
        "nll_equal_user_mean": float(weights @ nll),
        "brier_equal_user_mean": float(weights @ brier),
        "probability_clip": 1e-6,
        "calibration_bins": calibration,
        "scope": "point metrics of this fixed case sample; uncertainty and retraining comparisons are separate",
    }


def paired_user_nll(
    targets: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    users: np.ndarray,
    replications: int = 2000,
    seed: int = 0,
    margin: float = 0.01,
) -> dict:
    """Percentile bootstrap over equally weighted user-mean loss differences.

    The paired prediction cases must already be aligned by immutable case ID.
    This interval conditions on the fitted models and does not include variation
    from retraining. The confirmatory analysis must account for training seeds.
    """
    y, ref, alt, u = map(np.asarray, (targets, reference, candidate, users))
    if any(x.ndim != 1 or len(x) != len(y) for x in (y, ref, alt, u)) or not len(y):
        raise ValueError("Aligned nonempty prediction vectors required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(ref).all() or not np.isfinite(alt).all():
        raise ValueError("Invalid target or nonfinite probability")
    if (ref < 0).any() or (ref > 1).any() or (alt < 0).any() or (alt > 1).any():
        raise ValueError("Probabilities must lie in [0,1]")
    if replications < 100 or not np.isfinite(margin) or margin <= 0:
        raise ValueError("At least 100 replications and positive finite margin required")
    ref, alt = np.clip(ref, 1e-6, 1 - 1e-6), np.clip(alt, 1e-6, 1 - 1e-6)
    loss_ref = -(y * np.log(ref) + (1 - y) * np.log1p(-ref))
    loss_alt = -(y * np.log(alt) + (1 - y) * np.log1p(-alt))
    unique, inverse = np.unique(u, return_inverse=True)
    n = len(unique)
    if n < 2:
        raise ValueError("At least two independent user clusters required")
    counts = np.bincount(inverse)
    differences = np.bincount(inverse, weights=loss_alt - loss_ref) / counts
    rng = np.random.default_rng(seed)
    bootstrap = []
    for left in range(0, replications, 64):
        indices = rng.integers(0, n, (min(64, replications - left), n))
        bootstrap.extend(differences[indices].mean(axis=1).tolist())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "aggregation": "equal user-weighted mean NLL difference",
        "candidate_minus_reference_nats": float(differences.mean()),
        "ci95": [float(lower), float(upper)],
        "users": n,
        "cases": len(y),
        "margin_nats": margin,
        "noninferior_at_margin": bool(upper < margin),
        "replications": replications,
        "seed": seed,
        "probability_clip": 1e-6,
        "scope": "conditional on fixed fitted models; not a compression result",
    }
