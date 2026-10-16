"""Observable witnesses and environment-family rechecking for abstractions."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .core import Chain, Partition, approximate_partition, quotient, reachability
from .synthetic import World, intervene


def outcome_distribution(chain: Chain, horizon: int) -> np.ndarray:
    complete = reachability(chain, "complete", horizon)
    disengage = reachability(chain, "disengage", horizon)
    # This catalog requires completion and disengagement to remain terminal.
    for s, label in enumerate(chain.labels):
        if label in ("complete", "disengage"):
            if any(p > 0 and chain.labels[j] != label for j, p in enumerate(chain.transitions[s])):
                raise ValueError("Outcome catalog requires absorbing terminal labels")
    return np.column_stack([complete, disengage, np.maximum(0, 1 - complete - disengage)])


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape or (p < 0).any() or (q < 0).any():
        raise ValueError("Outcome distributions must be aligned and nonnegative")
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("Outcome distributions must sum to one")
    midpoint = (p + q) / 2
    value = 0.0
    for distribution in (p, q):
        positive = distribution > 0
        value += (
            float(
                np.sum(distribution[positive] * np.log(distribution[positive] / midpoint[positive]))
            )
            / 2
        )
    return value


@dataclass(frozen=True)
class Witness:
    strength: float
    states: tuple[int, int]
    js_nats: float
    completion_gap: float
    environment_evaluations: int


def search_witness(
    world: World, partition: Partition, strengths: Sequence[float], horizon: int
) -> Witness:
    if not strengths:
        raise ValueError("A nonempty feasible environment set is required")
    quotient(world.chain, partition)
    pairs = [(a, b) for block in partition for i, a in enumerate(block) for b in block[i + 1 :]]
    if not pairs:
        raise ValueError("No merged states to distinguish")
    best = None
    for strength in strengths:
        model = intervene(world, float(strength))
        predictions = outcome_distribution(model, horizon)
        for a, b in pairs:
            score = js_divergence(predictions[a], predictions[b])
            if best is None or score > best.js_nats:
                best = Witness(
                    float(strength),
                    (a, b),
                    score,
                    float(abs(predictions[a, 0] - predictions[b, 0])),
                    len(strengths),
                )
    return best


def refine(world: World, witness: Witness, epsilon: float) -> Partition:
    # The new partition is certified across the training and discovered environment.
    # Other held-out environments need their own evaluation and certificate.
    return approximate_partition([world.chain, intervene(world, witness.strength)], epsilon)


def budgeted_witness(
    world: World,
    partition: Partition,
    method: str,
    budget: int,
    horizon: int,
    seed: int = 0,
    maximum_strength: float = 0.3,
) -> Witness:
    """Evaluate exactly budget distinct environments, caching every observation."""
    from dataclasses import replace

    if method not in ("random", "grid", "adaptive") or budget < 2 or not 0 < maximum_strength <= 1:
        raise ValueError("Supported search, budget>=2 and feasible positive bound required")
    rng = np.random.default_rng(seed)
    scheduled = (
        rng.uniform(0, maximum_strength, budget).tolist()
        if method == "random"
        else np.linspace(0, maximum_strength, budget).tolist()
        if method == "grid"
        else [0.0, maximum_strength]
    )
    observations: dict[float, Witness] = {}
    while len(observations) < budget:
        if scheduled:
            strength = float(scheduled.pop(0))
        else:
            best = max(observations, key=lambda x: observations[x].js_nats)
            ordered = sorted(observations)
            index = ordered.index(best)
            neighbor = ordered[index - 1] if index else ordered[1]
            strength = (best + neighbor) / 2
        if strength in observations:
            raise ArithmeticError("Search exhausted distinct floating-point environments")
        observations[strength] = search_witness(world, partition, [strength], horizon)
    winner = max(observations.values(), key=lambda w: w.js_nats)
    return replace(winner, environment_evaluations=len(observations))
