"""Empirical compatibility diagnostics for cohort-level probability intervals.

This does not estimate the coverage of a confidence set for an unknown human
transition kernel, nor count individual binary outcomes inside probability bounds.
"""

import numpy as np


def cohort_interval_diagnostic(
    targets,
    lower,
    upper,
    users,
    *,
    replications=2000,
    seed=20261015,
    minimum_users=50,
    minimum_cases=2000,
):
    """Compare paired equal-user observed rates with mean forecast bounds.

    Each user contributes one mean observed response and one mean for each
    supplied forecast endpoint. Percentile user resamples assess both paired
    endpoint gaps. The two 97.5% intervals form a nominal Bonferroni 95% family
    for this cohort only, under the bootstrap's sampling assumptions. There is
    no guarantee across a selected family of cohorts or under calendar shocks.
    """
    y, lo, hi, u = map(np.asarray, (targets, lower, upper, users))
    if (
        y.ndim != 1
        or any(a.shape != y.shape for a in (lo, hi, u))
        or not np.isin(y, [0, 1]).all()
        or any(not np.isfinite(a).all() for a in (lo, hi))
        or np.any(lo < 0)
        or np.any(hi > 1)
        or np.any(lo > hi)
        or u.dtype.kind not in "iu"
        or np.any(u < 0)
        or not isinstance(replications, int)
        or replications < 100
        or not isinstance(minimum_users, int)
        or minimum_users < 2
        or not isinstance(minimum_cases, int)
        or minimum_cases < 2
    ):
        raise ValueError(
            "Aligned binary responses, ordered probability bounds and user clusters required"
        )
    unique, inverse = np.unique(u, return_inverse=True)
    result = dict(
        cases=len(y),
        users=len(unique),
        minimum_users=minimum_users,
        minimum_cases=minimum_cases,
        estimand="Equal-user average recorded response probability in the declared observed cohort",
        confidence_interval_coverage=None,
        scope="Paired cohort mean compatibility diagnostic, not individual outcome containment or unknown human-kernel confidence coverage. Original users are the resampling clusters; common calendar dependence, model misspecification and selection remain limitations.",
    )
    if len(y) < minimum_cases or len(unique) < minimum_users:
        return dict(**result, status="unsupported")
    counts = np.bincount(inverse)
    observed = np.bincount(inverse, weights=y) / counts
    predicted_lower = np.bincount(inverse, weights=lo) / counts
    predicted_upper = np.bincount(inverse, weights=hi) / counts
    gaps = np.column_stack([observed - predicted_lower, predicted_upper - observed])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((replications, 2))
    for left in range(0, replications, 32):
        size = min(32, replications - left)
        indices = rng.integers(0, len(unique), (size, len(unique)))
        bootstrap[left : left + size] = gaps[indices].mean(axis=1)
    endpoints = np.quantile(bootstrap, [0.0125, 0.9875], axis=0)
    if np.any(endpoints[1] < 0):
        finding = "mean_incompatibility_supported_under_bootstrap_assumptions"
    elif np.all(endpoints[0] > 0):
        finding = "mean_compatibility_supported_under_bootstrap_assumptions"
    else:
        finding = "unresolved"
    return dict(
        **result,
        status="evaluated",
        finding=finding,
        observed_equal_user_rate=float(observed.mean()),
        mean_lower_probability=float(predicted_lower.mean()),
        mean_upper_probability=float(predicted_upper.mean()),
        mean_interval_width=float((predicted_upper - predicted_lower).mean()),
        observed_minus_lower=dict(
            mean=float(gaps[:, 0].mean()), nominal_ci975=endpoints[:, 0].tolist()
        ),
        upper_minus_observed=dict(
            mean=float(gaps[:, 1].mean()), nominal_ci975=endpoints[:, 1].tolist()
        ),
        nominal_endpoint_family_confidence=0.95,
        multiplicity="Bonferroni across the two endpoint gaps in this declared cohort only; bootstrap approximation, no finite-sample guarantee",
        replications=replications,
        seed=seed,
    )
