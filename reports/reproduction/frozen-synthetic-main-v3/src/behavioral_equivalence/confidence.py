"""Outward-rounded Hoeffding sets with an exact rational tail certificate.

Coverage requires the fixed-design iid row experiment in spec/proofs.md.
"""

from fractions import Fraction
import math
import numpy as np
from .uncertainty import IntervalChain


def _exp_lower(x: Fraction, terms: int = 100) -> Fraction:
    total = term = Fraction(1)
    for k in range(1, terms + 1):
        term *= x / k
        total += term
    return total


def _endpoint(value: Fraction, upper: bool) -> float:
    rounded = float(value)
    exact = Fraction.from_float(rounded)
    if (upper and exact < value) or (not upper and exact > value):
        rounded = math.nextafter(rounded, math.inf if upper else -math.inf)
    return rounded


def certified_hoeffding(
    counts: np.ndarray, labels: tuple[str, ...], alpha: Fraction = Fraction(1, 20), bits: int = 32
) -> tuple[IntervalChain, dict]:
    a = np.asarray(counts)
    n = len(labels)
    if a.shape != (n, n) or n == 0 or a.dtype.kind not in "iu" or (a < 0).any():
        raise ValueError("Square nonnegative integer row counts required")
    if not isinstance(alpha, Fraction) or not 0 < alpha < 1 or not 8 <= bits <= 48:
        raise ValueError("Rational alpha in (0,1) and precision from 8 to 48 required")
    denominator = 1 << bits
    threshold = 2 * n * n / alpha
    lower, upper = np.zeros((n, n)), np.ones((n, n))
    certificates = []
    by_sample_count = {}
    for s in range(n):
        m = sum(int(v) for v in a[s])
        if m == 0:
            certificates.append(
                {"samples": 0, "radius_numerator": denominator, "tail": "whole simplex"}
            )
            continue
        if m not in by_sample_count:
            # Floating operations only propose a radius. The exact inequality
            # below independently decides whether that proposal is sufficient.
            proposed = math.sqrt(math.log(float(threshold)) / (2 * m))
            numerator = min(denominator, math.ceil(proposed * denominator) + 1)
            radius = Fraction(numerator, denominator)
            while numerator < denominator and _exp_lower(2 * m * radius * radius) < threshold:
                numerator += 1
                radius = Fraction(numerator, denominator)
            if numerator < denominator:
                proof = _exp_lower(2 * m * radius * radius)
                evidence = {
                    "terms": 100,
                    "exp_lower_numerator_hex": format(proof.numerator, "x"),
                    "exp_lower_denominator_hex": format(proof.denominator, "x"),
                }
            else:
                evidence = {"tail": "whole simplex"}
            by_sample_count[m] = (radius, {"samples": m, "radius_numerator": numerator, **evidence})
        radius, row_proof = by_sample_count[m]
        certificates.append(row_proof)
        for j in range(n):
            empirical = Fraction(int(a[s, j]), m)
            lower[s, j] = _endpoint(max(Fraction(), empirical - radius), False)
            upper[s, j] = _endpoint(min(Fraction(1), empirical + radius), True)
    result = IntervalChain(lower, upper, labels)
    cert = {
        "schema_version": "hoeffding-rational-tail-v1",
        "alpha": [alpha.numerator, alpha.denominator],
        "radius_denominator": denominator,
        "rows": certificates,
        "coordinate_count": n * n,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "sampling_assumptions": "frozen representation, fixed sample counts, iid categorical samples within each fixed row; not automatically applicable to human trajectories",
    }
    return result, cert


def verify_hoeffding(counts: np.ndarray, labels: tuple[str, ...], certificate: dict) -> None:
    alpha = Fraction(*certificate["alpha"])
    denominator = certificate["radius_denominator"]
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("Radius denominator must be a power of two")
    _, expected = certified_hoeffding(counts, labels, alpha, denominator.bit_length() - 1)
    if expected != certificate:
        raise ValueError("Confidence certificate differs from recomputation")
