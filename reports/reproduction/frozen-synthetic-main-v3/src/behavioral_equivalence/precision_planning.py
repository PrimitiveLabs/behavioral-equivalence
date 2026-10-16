"""Validation-only conditional normal planning for paired user-level NLL contrasts."""

from math import ceil, sqrt
from statistics import NormalDist
import numpy as np


def user_nll_differences(targets, reference, candidate, users):
    y, r, c, u = map(np.asarray, (targets, reference, candidate, users))
    if (
        y.ndim != 1
        or not len(y)
        or any(x.shape != y.shape for x in (r, c, u))
        or not np.isin(y, [0, 1]).all()
        or any(not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1) for x in (r, c))
        or u.dtype.kind not in "iu"
        or np.any(u < 0)
    ):
        raise ValueError(
            "Aligned binary responses, valid probabilities and integer user identifiers required"
        )
    r, c = np.clip(r, 1e-6, 1 - 1e-6), np.clip(c, 1e-6, 1 - 1e-6)
    loss = -(y * np.log(c) + (1 - y) * np.log1p(-c)) + (y * np.log(r) + (1 - y) * np.log1p(-r))
    ids, inverse = np.unique(u, return_inverse=True)
    if len(ids) < 2:
        raise ValueError("At least two user clusters required for variance planning")
    return np.bincount(inverse, weights=loss) / np.bincount(inverse)


def normal_precision_plan(
    differences, gaps=(0.01, 0.0075, 0.005, 0.0025), *, alpha=0.025, power=0.8
):
    """Approximate planning only, not a finite-sample or human-validity guarantee.

    A gap is the assumed distance of the true mean from the decision boundary.
    For NI it is margin minus true degradation; for improvement it is the
    assumed reduction in loss. Validation variation is only a planning proxy.
    """
    d = np.asarray(differences, dtype=float)
    if (
        d.ndim != 1
        or len(d) < 2
        or not np.isfinite(d).all()
        or not gaps
        or any(not np.isfinite(g) or g <= 0 for g in gaps)
        or not 0 < alpha < 0.5
        or not 0.5 < power < 1
    ):
        raise ValueError(
            "Finite user differences, positive gaps and valid planning probabilities required"
        )
    variance = float(d.var(ddof=1))
    sd = sqrt(variance)
    z = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(power)
    return dict(
        validation_users=len(d),
        validation_user_variance=variance,
        validation_user_sd=sd,
        one_sided_alpha=alpha,
        target_power=power,
        required_users=[
            dict(gap_nats=float(g), normal_approximate_users=max(2, ceil((z * sd / g) ** 2)))
            for g in gaps
        ],
        scope="Conditional normal approximation using validation user-loss variation; independent representative user clusters and stable future variance assumed. No test scores, finite-sample power guarantee or across-seed independence claim.",
    )


def forecast_identity(candidate, reference, environments):
    """Exact identity of stored coarse-state forecasts over a declared context family."""
    c, r = np.asarray(candidate), np.asarray(reference)
    e = np.asarray(environments)
    if (
        c.ndim != 2
        or c.shape != r.shape
        or not c.size
        or e.ndim != 1
        or not len(e)
        or e.dtype.kind not in "iu"
        or np.any(e < 0)
        or np.any(e >= c.shape[1])
        or any(not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1) for a in (c, r))
    ):
        raise ValueError("Aligned probability tables and valid context indices required")
    return bool(np.array_equal(c[:, e], r[:, e]))


def improvement_precision_plan(differences, gaps, *, structurally_identical):
    """Do not report achievable positive improvement for structurally identical models."""
    plan = normal_precision_plan(differences, gaps)
    if not isinstance(structurally_identical, bool):
        raise ValueError("Structural identity must be established over the declared context family")
    if structurally_identical:
        if np.any(np.asarray(differences) != 0):
            raise ValueError("Identical forecasts cannot have unequal paired losses")
        plan["required_users"] = [
            dict(gap_nats=float(g), normal_approximate_users=None) for g in gaps
        ]
        plan["improvement_status"] = "impossible_for_identical_forecasts"
    else:
        plan["improvement_status"] = "conditional_variance_proxy_only"
    plan["structurally_identical"] = structurally_identical
    return plan
