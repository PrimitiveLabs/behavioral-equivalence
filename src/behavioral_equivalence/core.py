"""Finite labeled chains, quotient construction, and finite-horizon properties."""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Chain:
    transitions: FloatArray
    labels: tuple[str, ...]
    initial: FloatArray

    def __post_init__(self) -> None:
        p = np.array(self.transitions, dtype=np.float64, copy=True)
        mu = np.array(self.initial, dtype=np.float64, copy=True)
        n = len(self.labels)
        if n == 0 or p.shape != (n, n) or mu.shape != (n,):
            raise ValueError("Chain dimensions do not match labels")
        if not np.isfinite(p).all() or not np.isfinite(mu).all():
            raise ValueError("Probabilities must be finite")
        if (p < 0).any() or (mu < 0).any():
            raise ValueError("Probabilities must be nonnegative")
        if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12):
            raise ValueError("Transition rows must sum to one")
        if not math.isclose(float(mu.sum()), 1, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Initial distribution must sum to one")
        p.setflags(write=False)
        mu.setflags(write=False)
        object.__setattr__(self, "transitions", p)
        object.__setattr__(self, "initial", mu)

    @property
    def size(self) -> int:
        return len(self.labels)

    def digest(self) -> str:
        payload = json.dumps(
            {"p": self.transitions.tolist(), "labels": self.labels, "mu": self.initial.tolist()},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def check_partition(chain: Chain, blocks: Partition) -> None:
    flat = [s for block in blocks for s in block]
    if any(not b for b in blocks) or sorted(flat) != list(range(chain.size)):
        raise ValueError("Partition must cover each state exactly once")
    if any(len({chain.labels[s] for s in b}) != 1 for b in blocks):
        raise ValueError("Partition mixes observable labels")


def label_partition(chain: Chain) -> Partition:
    return tuple(
        tuple(s for s, label in enumerate(chain.labels) if label == value)
        for value in sorted(set(chain.labels))
    )


def aggregated_rows(chain: Chain, blocks: Partition) -> FloatArray:
    check_partition(chain, blocks)
    return np.column_stack([chain.transitions[:, b].sum(axis=1) for b in blocks])


def exact_partition(
    chains: Sequence[Chain], initial_partition: Partition | None = None
) -> Partition:
    """Coarsest common exact partition of rationally normalized input weights.

    Float sums can hide distinctions smaller than an aggregation ulp. Compute
    signatures with exact rational arithmetic, using the same kernel semantics
    as the exact certificate checker. Floating quotient evaluation remains a
    separate numerical path with its own reported tolerance.
    """
    _check_family(chains)
    kernels = []
    zero = Fraction()
    for chain in chains:
        kernel = []
        for row in chain.transitions:
            weights = [(j, Fraction(float(value))) for j, value in enumerate(row) if value]
            total = sum((value for _, value in weights), zero)
            kernel.append([(j, value / total) for j, value in weights])
        kernels.append(kernel)
    blocks = initial_partition if initial_partition is not None else label_partition(chains[0])
    check_partition(chains[0], blocks)
    while True:
        class_index = [0] * chains[0].size
        for i, block in enumerate(blocks):
            for state in block:
                class_index[state] = i
        rows = []
        for kernel in kernels:
            aggregated = []
            for row in kernel:
                values = [zero] * len(blocks)
                for successor, probability in row:
                    values[class_index[successor]] += probability
                aggregated.append(tuple(values))
            rows.append(aggregated)
        refined = []
        for block in blocks:
            groups: dict[tuple[Fraction, ...], list[int]] = {}
            for state in block:
                signature = tuple(value for environment in rows for value in environment[state])
                groups.setdefault(signature, []).append(state)
            refined.extend(tuple(group) for group in groups.values())
        result = tuple(refined)
        if result == blocks:
            return result
        blocks = result


def _check_family(chains: Sequence[Chain]) -> None:
    if not chains or any(c.labels != chains[0].labels for c in chains):
        raise ValueError("Environment family must share nonempty labeled state space")


def quotient(chain: Chain, blocks: Partition) -> tuple[Chain, float]:
    rows = aggregated_rows(chain, blocks)
    p = np.vstack([rows[list(block)].mean(axis=0) for block in blocks])
    discrepancy = max(
        float(np.abs(rows[s] - p[i]).sum() / 2) for i, block in enumerate(blocks) for s in block
    )
    mu = np.array([chain.initial[list(b)].sum() for b in blocks])
    return Chain(p, tuple(chain.labels[b[0]] for b in blocks), mu), discrepancy


def approximate_partition(
    chains: Sequence[Chain], epsilon: float, initial_partition: Partition | None = None
) -> Partition:
    """Split label blocks until every state matches its actual quotient row.

    This deterministic greedy refinement is certified, not necessarily optimal.
    Every split changes successor classes, so all classes are rechecked.
    """
    _check_family(chains)
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    if epsilon == 0:
        return exact_partition(chains, initial_partition)
    blocks = initial_partition if initial_partition is not None else label_partition(chains[0])
    check_partition(chains[0], blocks)
    while True:
        split_index = None
        witness = None
        for c in chains:
            rows = aggregated_rows(c, blocks)
            for i, block in enumerate(blocks):
                center = rows[list(block)].mean(axis=0)
                distances = np.abs(rows[list(block)] - center).sum(axis=1) / 2
                if float(distances.max()) > epsilon:
                    split_index = i
                    witness = block[int(distances.argmax())]
                    break
            if split_index is not None:
                break
        if split_index is None:
            return blocks
        block = blocks[split_index]
        blocks = (
            blocks[:split_index]
            + ((witness,), tuple(s for s in block if s != witness))
            + blocks[split_index + 1 :]
        )


def reachability(chain: Chain, target: str, horizon: int) -> FloatArray:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    goal = np.array([label == target for label in chain.labels])
    values = goal.astype(float)
    for _ in range(horizon):
        values = np.where(goal, 1, np.einsum("ij,j->i", chain.transitions, values))
    return values


def rational_reachability(chain: Chain, target: str, horizon: int) -> list[Fraction]:
    """Independent exact arithmetic on the input binary floats, for small fixtures."""
    if horizon < 0:
        raise ValueError("Horizon must be nonnegative")
    p = [[Fraction(float(x)) for x in row] for row in chain.transitions]
    values = [Fraction(label == target) for label in chain.labels]
    for _ in range(horizon):
        previous = values
        values = [
            Fraction(1)
            if chain.labels[s] == target
            else sum((prob * v for prob, v in zip(row, previous)), Fraction())
            for s, row in enumerate(p)
        ]
    return values


def coupling_bound(epsilon: float, horizon: int, eta: float = 0) -> float:
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (epsilon, eta)):
        raise ValueError("Coupling errors must lie in [0,1]")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    if epsilon == 1:
        return 1 if horizon else eta
    # Stable for very small epsilon and large horizons.
    return eta + (1 - eta) * -math.expm1(horizon * math.log1p(-epsilon))
