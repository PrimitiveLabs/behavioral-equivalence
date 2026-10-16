"""Simultaneous count-ratio bounds with arbitrary dependence within users."""

from fractions import Fraction
import math

import numpy as np

from .confidence import _endpoint, _exp_lower


def cluster_ratio_bounds(counts, *, maximum_per_user, alpha=Fraction(1, 20), bits=32):
    """Bound E[user successor counts] / E[user row counts].

    counts has shape (independent users, rows, successors). Include users with
    zero visits. A fixed, outcome-blind per-user cap bounds each entire table.
    Independence across users and a frozen representation are assumptions,
    not conclusions of this function. These are not human Markov certificates.
    """
    a = np.asarray(counts)
    if (
        a.ndim != 3
        or min(a.shape) == 0
        or a.dtype.kind not in "iu"
        or np.any(a < 0)
        or type(maximum_per_user) is not int
        or maximum_per_user < 1
        or any(sum(map(int, user.flat)) > maximum_per_user for user in a)
    ):
        raise ValueError("Nonnegative integer user tables within a fixed positive cap required")
    totals = np.asarray(
        [
            [sum(int(v) for v in a[:, row, col]) for col in range(a.shape[2])]
            for row in range(a.shape[1])
        ],
        dtype=object,
    )
    return cluster_ratio_from_totals(
        totals, users=a.shape[0], maximum_per_user=maximum_per_user, alpha=alpha, bits=bits
    )


def cluster_ratio_from_totals(totals, *, users, maximum_per_user, alpha=Fraction(1, 20), bits=32):
    """Memory-bounded sufficient-statistic interface.

    Upstream processing must verify the per-user cap, include zero-visit users,
    and prevent duplicate users. Aggregate totals alone cannot verify those facts.
    """
    a = np.asarray(totals, dtype=object)
    if (
        a.ndim != 2
        or min(a.shape) == 0
        or any(not isinstance(v, (int, np.integer)) or v < 0 for v in a.flat)
        or type(users) is not int
        or users < 1
        or type(maximum_per_user) is not int
        or maximum_per_user < 1
        or sum(int(v) for v in a.flat) > users * maximum_per_user
    ):
        raise ValueError("Integer aggregate counts within the declared user capacity required")
    if not isinstance(alpha, Fraction) or not 0 < alpha < 1 or not 8 <= bits <= 48:
        raise ValueError("Rational alpha and precision from 8 to 48 required")
    rows, successors = a.shape
    coordinates = rows * (successors + 1)
    threshold = Fraction(2 * coordinates, 1) / alpha
    denominator = 1 << bits
    proposed = math.sqrt(math.log(float(threshold)) / (2 * users))
    numerator = min(denominator, math.ceil(proposed * denominator) + 1)
    radius = Fraction(numerator, denominator)
    while numerator < denominator and _exp_lower(2 * users * radius**2) < threshold:
        numerator += 1
        radius = Fraction(numerator, denominator)
    lo = np.zeros((rows, successors))
    hi = np.ones((rows, successors))
    normalizer = users * maximum_per_user
    for row in range(rows):
        totals = [int(v) for v in a[row]]
        mean_denominator = Fraction(sum(totals), normalizer)
        denominator_lower = max(Fraction(), mean_denominator - radius)
        denominator_upper = min(Fraction(1), mean_denominator + radius)
        for col, total in enumerate(totals):
            mean = Fraction(total, normalizer)
            lower = max(Fraction(), mean - radius) / denominator_upper
            upper = (
                min(Fraction(1), (mean + radius) / denominator_lower)
                if denominator_lower
                else Fraction(1)
            )
            lo[row, col] = _endpoint(lower, False)
            hi[row, col] = _endpoint(upper, True)
    return (
        lo,
        hi,
        {
            "schema_version": "bounded-user-count-ratio-v1",
            "users": users,
            "coordinates": coordinates,
            "maximum_per_user": maximum_per_user,
            "alpha": [alpha.numerator, alpha.denominator],
            "radius": [numerator, denominator],
            "tail_inequality_verified": numerator == denominator
            or _exp_lower(2 * users * radius**2) >= threshold,
            "estimand": "Ratio of expected per-user successor count to expected per-user row count, where the latter is positive",
            "assumptions": "Independent user tables, common target population, fixed representation and outcome-blind cap; arbitrary within-user dependence allowed",
            "human_multistep_coverage_established": False,
        },
    )


def pooled_decision_intervals(counts, *, users, maximum_per_user, alpha=Fraction(1, 20)):
    """Compile observed pooled emissions and conditional transitions to 3K stages.

    This pooled count model differs from a training-teacher emission model.
    Half the simultaneous error allowance covers emissions, half transitions.
    The certified cap applies to counts before both deterministic aggregations.
    """
    from .uncertainty import IntervalChain

    a = np.asarray(counts)
    if a.ndim != 3 or a.shape[1] != 2 or a.shape[0] != a.shape[2] or a.dtype.kind not in "iu":
        raise ValueError("Integer counts[K,2,K] required")
    if not isinstance(alpha, Fraction) or not 0 < alpha < 1:
        raise ValueError("Rational alpha in (0,1) required")
    k = a.shape[0]
    # Object reduction prevents machine-integer overflow before validation.
    emission_counts = a.astype(object).sum(axis=2)
    e_lo, e_hi, e_cert = cluster_ratio_from_totals(
        emission_counts, users=users, maximum_per_user=maximum_per_user, alpha=alpha / 2
    )
    t_lo, t_hi, t_cert = cluster_ratio_from_totals(
        a.reshape(2 * k, k), users=users, maximum_per_user=maximum_per_user, alpha=alpha / 2
    )
    lo, hi = np.zeros((3 * k, 3 * k)), np.zeros((3 * k, 3 * k))
    for state in range(k):
        lo[3 * state, 3 * state + 1 : 3 * state + 3] = e_lo[state]
        hi[3 * state, 3 * state + 1 : 3 * state + 3] = e_hi[state]
        for outcome in range(2):
            lo[3 * state + 1 + outcome, ::3] = t_lo[2 * state + outcome]
            hi[3 * state + 1 + outcome, ::3] = t_hi[2 * state + outcome]
    model = IntervalChain(
        lo, hi, tuple(label for _ in range(k) for label in ("pending", "negative", "positive"))
    )
    return model, {
        "schema_version": "pooled-user-decision-intervals-v1",
        "alpha": [alpha.numerator, alpha.denominator],
        "emissions": e_cert,
        "transitions": t_cert,
        "teacher_emissions_used": False,
        "human_multistep_coverage_established": False,
    }
