"""Exact certificates for rational kernels defined by normalized serialized weights.

This path is intended for small independent correctness fixtures. The numerical
pilot path uses floating arithmetic and has a different certification status.
"""

from fractions import Fraction
from .core import Chain, Partition, check_partition

RationalMatrix = tuple[tuple[Fraction, ...], ...]


def normalized_kernel(chain: Chain) -> RationalMatrix:
    result = []
    for row in chain.transitions:
        weights = tuple(Fraction(float(x)) for x in row)
        total = sum(weights, Fraction())
        result.append(tuple(x / total for x in weights))
    return tuple(result)


def exact_values(
    p: RationalMatrix, labels: tuple[str, ...], target: str, horizon: int
) -> tuple[Fraction, ...]:
    values = tuple(Fraction(label == target) for label in labels)
    for _ in range(horizon):
        values = tuple(
            Fraction(1)
            if labels[s] == target
            else sum((x * v for x, v in zip(row, values)), Fraction())
            for s, row in enumerate(p)
        )
    return values


def certify(chain: Chain, blocks: Partition, target: str, horizon: int) -> dict:
    """Return exact rational event bounds and an observed-error witness.

    Every arithmetic operation contributing to the certificate is rational.
    The input kernel is explicitly defined by row-normalizing the input floats
    interpreted as exact binary rationals. This semantics is part of the output.
    """
    check_partition(chain, blocks)
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    p = normalized_kernel(chain)
    aggregated = tuple(
        tuple(sum((row[s] for s in block), Fraction()) for block in blocks) for row in p
    )
    q = tuple(
        tuple(
            sum((aggregated[s][j] for s in block), Fraction()) / len(block)
            for j in range(len(blocks))
        )
        for block in blocks
    )
    epsilon = max(
        sum((abs(x - y) for x, y in zip(aggregated[s], q[i])), Fraction()) / 2
        for i, block in enumerate(blocks)
        for s in block
    )
    delta = 1 - (1 - epsilon) ** horizon
    abstract = exact_values(q, tuple(chain.labels[b[0]] for b in blocks), target, horizon)
    concrete = exact_values(p, chain.labels, target, horizon)
    actual = max(abs(concrete[s] - abstract[i]) for i, b in enumerate(blocks) for s in b)
    if actual > delta:
        raise ArithmeticError("Exact witness contradicts the coupling certificate")
    return {
        "schema_version": "0.1",
        "source_digest": chain.digest(),
        "kernel_semantics": "exact binary-rational weights normalized within each row",
        "partition": [list(b) for b in blocks],
        "target": target,
        "horizon": horizon,
        "numeric_status": "exact_rational",
        "epsilon": str(epsilon),
        "delta": str(delta),
        "abstract_values": [str(v) for v in abstract],
        "intervals": [
            [str(max(Fraction(), v - delta)), str(min(Fraction(1), v + delta))] for v in abstract
        ],
        "observed_max_error": str(actual),
        "scope": "specified_normalized_model_only",
    }


def verify(chain: Chain, certificate: dict) -> None:
    blocks = tuple(tuple(b) for b in certificate["partition"])
    computed = certify(chain, blocks, certificate["target"], certificate["horizon"])
    if computed != certificate:
        raise ValueError("Certificate differs from exact recomputation")
