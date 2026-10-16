"""Uniform H1 uncertainty/abstraction envelope for categorical endpoint quotients."""

from fractions import Fraction
import math
import numpy as np


def outward(value, upper):
    result = float(value)
    exact = Fraction.from_float(result)
    if (upper and exact < value) or (not upper and exact > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    return result


def uniform_endpoint_envelope(lower, upper, blocks):
    lo, hi = map(np.asarray, (lower, upper))
    if (
        lo.ndim != 2
        or not len(lo)
        or lo.shape[1] not in [2, 4]
        or hi.shape != lo.shape
        or any(not np.isfinite(x).all() for x in [lo, hi])
        or np.any(lo < 0)
        or np.any(hi > 1)
        or np.any(lo > hi)
    ):
        raise ValueError("Ordered finite categorical probability intervals required")
    flat = [s for b in blocks for s in b]
    if (
        any(not len(b) for b in blocks)
        or len(flat) != len(lo)
        or sorted(flat) != list(range(len(lo)))
    ):
        raise ValueError("Complete disjoint nonempty parent-state partition required")
    low = [[Fraction.from_float(float(v)) for v in row] for row in lo]
    high = [[Fraction.from_float(float(v)) for v in row] for row in hi]
    if any(sum(row_lower) > 1 or sum(row_upper) < 1 for row_lower, row_upper in zip(low, high)):
        raise ValueError("Every interval row must admit a probability simplex")
    averages = []
    delta = Fraction()
    for block in blocks:
        a = [sum(low[s][j] for s in block) / len(block) for j in range(lo.shape[1])]
        b = [sum(high[s][j] for s in block) / len(block) for j in range(lo.shape[1])]
        averages.append((a, b))
        for s in block:
            bound = sum(max(high[s][j] - a[j], b[j] - low[s][j]) for j in range(lo.shape[1])) / 2
            delta = max(delta, min(Fraction(1), bound))
    abstract_lower = np.array([[outward(x, False) for x in a] for a, _ in averages])
    abstract_upper = np.array([[outward(x, True) for x in b] for _, b in averages])
    original_lower = np.empty_like(lo, dtype=float)
    original_upper = np.empty_like(hi, dtype=float)
    for block, (a, b) in zip(blocks, averages):
        original_lower[list(block)] = [outward(max(Fraction(), x - delta), False) for x in a]
        original_upper[list(block)] = [outward(min(Fraction(1), x + delta), True) for x in b]
    return dict(
        abstract_lower=abstract_lower,
        abstract_upper=abstract_upper,
        original_lower=original_lower,
        original_upper=original_upper,
        uniform_delta=outward(delta, True),
        delta_exact=[str(delta.numerator), str(delta.denominator)],
        scope="Uniform over feasible categorical rows in the stated interval box, quotient is the uniform block mean of each admissible parent. H1 only. Statistical membership and human transfer assumptions are separate. Coordinate dependence is relaxed conservatively.",
    )
