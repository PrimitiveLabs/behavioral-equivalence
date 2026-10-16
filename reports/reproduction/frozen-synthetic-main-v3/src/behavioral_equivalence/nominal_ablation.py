"""Empirical-kernel control removing statistical transition uncertainty.

Tiny coordinate intervals enclose rational point estimates only. They are not
confidence intervals. The same hidden-state abstraction envelope remains in use.
"""

from fractions import Fraction
import hashlib
import json
import math
import numpy as np
from .uncertainty import IntervalChain


def _endpoint(value: Fraction, upper: bool) -> float:
    rounded = float(value)
    represented = Fraction.from_float(rounded)
    if (upper and represented < value) or (not upper and represented > value):
        return math.nextafter(rounded, math.inf if upper else -math.inf)
    return rounded


def empirical_point_intervals(
    counts: np.ndarray,
    labels: tuple[str, ...],
    known_terminal_closure: bool,
) -> tuple[IntervalChain, dict]:
    """Enclose exact normalized counts, with explicit uniform zero-row imputation."""
    a = np.asarray(counts)
    n = len(labels)
    if n == 0 or a.shape != (n, n) or a.dtype.kind not in "iu" or (a < 0).any():
        raise ValueError("Square nonnegative integer row counts required")
    lower, upper = np.zeros((n, n)), np.zeros((n, n))
    zero_rows = []
    for s, label in enumerate(labels):
        allowed = [
            j
            for j in range(n)
            if not known_terminal_closure
            or label not in {"complete", "disengage"}
            or labels[j] == label
        ]
        if any(int(a[s, j]) for j in range(n) if j not in allowed):
            raise ValueError("Counts contradict known terminal closure")
        total = sum(int(value) for value in a[s])
        if total == 0:
            zero_rows.append(s)
        for j in range(n):
            value = (
                Fraction(int(a[s, j]), total)
                if total
                else Fraction(1, len(allowed))
                if j in allowed
                else Fraction()
            )
            lower[s, j] = _endpoint(value, False)
            upper[s, j] = _endpoint(value, True)
    payload = {"counts": a.tolist(), "labels": labels}
    result = IntervalChain(lower, upper, labels)
    certificate = {
        "schema_version": "empirical-point-numeric-envelope-v1",
        "counts_and_labels_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "known_terminal_closure": known_terminal_closure,
        "zero_sample_rows": zero_rows,
        "zero_row_policy": "uniform over the specified allowed successors",
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "statistical_coverage_claim": "none",
        "numeric_scope": "outward binary-float enclosures of exact rational empirical point rows",
        "ablation_scope": "removes statistical transition uncertainty, retains hidden-state envelope",
    }
    return result, certificate


def verify_empirical_point(counts: np.ndarray, labels: tuple[str, ...], certificate: dict) -> None:
    _, expected = empirical_point_intervals(counts, labels, certificate["known_terminal_closure"])
    if expected != certificate:
        raise ValueError("Empirical point certificate differs from recomputation")
