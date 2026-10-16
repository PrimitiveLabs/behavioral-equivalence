"""Uniform uncertainty abstraction by a label-preserving rectangular envelope.

This enclosure already includes hidden concrete-state choices. It does not add
an independently estimated nominal abstraction radius to a confidence set.
"""

from fractions import Fraction
import hashlib
import json
import math
import numpy as np
from .core import Chain, Partition, check_partition
from .integer_bounds import interval_kernel
from .uncertainty import IntervalChain


def _outward(value: Fraction, upper: bool) -> float:
    rounded = float(value)
    represented = Fraction.from_float(rounded)
    if (upper and represented < value) or (not upper and represented > value):
        return math.nextafter(rounded, math.inf if upper else -math.inf)
    return rounded


def abstract_envelope(model: IntervalChain, blocks: Partition) -> tuple[IntervalChain, dict]:
    """Enclose every projected concrete row, including all simplex constraints.

    For a block C, min P(C) = max(sum lower[C], 1-sum upper[outside C]).
    The corresponding maximum uses upper[C] and lower[outside C]. Across
    origins in one abstract block, take the hull. Convex combinations of
    those rows also belong to the hull, covering history-dependent hidden
    state mixtures. Robust checking uses per-step rectangular semantics.
    """
    rows = interval_kernel(model)
    n = len(model.labels)
    # Partition validation uses a trivial valid chain, not nominal interval midpoints.
    check_partition(Chain(np.eye(n), model.labels, np.full(n, 1 / n)), blocks)
    mapping = {s: i for i, block in enumerate(blocks) for s in block}
    k = len(blocks)
    projected = []
    for row in rows:
        lo = [0] * k
        hi = [0] * k
        for s in range(n):
            lo[mapping[s]] += row.lower[s]
            hi[mapping[s]] += row.upper[s]
        lower_sum, upper_sum = sum(lo), sum(hi)
        projected.append(
            (
                tuple(
                    Fraction(max(lo[b], row.total - upper_sum + hi[b]), row.total) for b in range(k)
                ),
                tuple(
                    Fraction(min(hi[b], row.total - lower_sum + lo[b]), row.total) for b in range(k)
                ),
            )
        )
    lower = np.empty((k, k))
    upper = np.empty((k, k))
    exact = []
    for b, block in enumerate(blocks):
        lo = [min(projected[s][0][j] for s in block) for j in range(k)]
        hi = [max(projected[s][1][j] for s in block) for j in range(k)]
        lower[b] = [_outward(v, False) for v in lo]
        upper[b] = [_outward(v, True) for v in hi]
        exact.append(
            {
                "lower": [[v.numerator, v.denominator] for v in lo],
                "upper": [[v.numerator, v.denominator] for v in hi],
            }
        )
    abstract = IntervalChain(lower, upper, tuple(model.labels[b[0]] for b in blocks))
    interval_kernel(abstract)
    payload = {"lower": model.lower.tolist(), "upper": model.upper.tolist(), "labels": model.labels}
    certificate = {
        "schema_version": "uncertainty-envelope-v1",
        "source_digest": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "partition": [list(b) for b in blocks],
        "exact_projected_hulls": exact,
        "abstract_lower": lower.tolist(),
        "abstract_upper": upper.tolist(),
        "labels": list(abstract.labels),
        "semantics": "label-path enclosure, row-rectangular per-step adversary",
        "abstraction_allowance": "included in envelope; do not add a nominal epsilon again",
        "statistical_scope": "conditional on concrete kernel membership; no human sampling guarantee",
    }
    return abstract, certificate


def verify_envelope(model: IntervalChain, certificate: dict) -> None:
    _, expected = abstract_envelope(model, tuple(tuple(b) for b in certificate["partition"]))
    if expected != certificate:
        raise ValueError("Uncertainty abstraction certificate differs from recomputation")
