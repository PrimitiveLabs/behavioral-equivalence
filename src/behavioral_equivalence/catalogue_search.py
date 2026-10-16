"""Equal-budget environment search with no access to unevaluated outcomes."""

from collections.abc import Callable
import numpy as np
from .core import Partition

SEARCH_METHODS = ("random", "maximin_grid", "rbf_ucb", "local_search")


def maximum_within_block_js(
    probabilities: np.ndarray, blocks: Partition
) -> tuple[float, tuple[int, int]]:
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 2
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1)
    ):
        raise ValueError("Finite probability distributions required")
    pairs = [(a, b) for block in blocks for i, a in enumerate(block) for b in block[i + 1 :]]
    if not pairs or sorted(s for block in blocks for s in block) != list(range(len(p))):
        raise ValueError("Partition with at least one merged pair required")
    a, b = np.array(pairs).T
    left, right = p[a], p[b]
    midpoint = (left + right) / 2
    divergence = np.zeros(len(pairs))
    for rows in (left, right):
        ratio = np.divide(rows, midpoint, out=np.ones_like(rows), where=rows > 0)
        divergence += (rows * np.log(ratio)).sum(axis=1) / 2
    winner = int(np.argmax(divergence))
    return max(0.0, float(divergence[winner])), pairs[winner]


def search_catalogue(
    points: np.ndarray, evaluate: Callable[[int], float], method: str, budget: int, seed: int
) -> dict:
    points = np.asarray(points, dtype=float)
    if (
        method not in SEARCH_METHODS
        or points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or len(np.unique(points, axis=0)) != len(points)
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or not 1 <= budget <= len(points)
    ):
        raise ValueError(
            "Distinct finite catalogue, supported method and feasible integer budget required"
        )
    rng = np.random.default_rng(seed)
    selected, scores = [], []
    for step in range(budget):
        available = np.array([i for i in range(len(points)) if i not in selected])
        if method == "random":
            choice = int(rng.choice(available))
        elif not selected:
            choice = int(available[np.argmin((points[available] ** 2).sum(axis=1))])
        elif method == "local_search" and step % 3 != 0:
            best = selected[int(np.argmax(scores))]
            distance = ((points[available] - points[best]) ** 2).sum(axis=1)
            choice = int(available[np.argmin(distance)])
        elif method == "rbf_ucb":
            known = points[selected]
            covariance = np.exp(-((known[:, None] - known[None, :]) ** 2).sum(axis=2) / 0.5)
            covariance += np.eye(len(selected)) * 0.001
            cross = np.exp(-((points[available, None] - known[None, :]) ** 2).sum(axis=2) / 0.5)
            # Only already-observed scalar disagreements enter this surrogate.
            prediction = np.einsum("ij,j->i", cross, np.linalg.solve(covariance, np.array(scores)))
            solved = np.linalg.solve(covariance, cross.T)
            variance = np.maximum(0, 1 - np.einsum("ij,ji->i", cross, solved))
            choice = int(available[np.argmax(prediction + 0.1 * np.sqrt(variance))])
        else:
            distance = (
                ((points[available, None] - points[np.array(selected)][None, :]) ** 2)
                .sum(axis=2)
                .min(axis=1)
            )
            choice = int(available[np.argmax(distance)])
        score = float(evaluate(choice))
        if not np.isfinite(score) or not 0 <= score <= np.log(2) + 1e-12:
            raise ValueError("Evaluator must return a finite Jensen-Shannon divergence")
        selected.append(choice)
        scores.append(score)
    best = int(np.argmax(scores))
    return dict(
        indices=selected,
        scores=scores,
        winner=selected[best],
        maximum_js_nats=scores[best],
        environment_evaluations=len(selected),
    )


def random_refinement_to_count(blocks: Partition, target: int, seed: int) -> Partition:
    """A label-preserving nested random control at exactly the target state count."""
    n = sum(map(len, blocks))
    if not len(blocks) <= target <= n:
        raise ValueError("Target must be between original and singleton state counts")
    rng = np.random.default_rng(seed)
    result = list(blocks)
    while len(result) < target:
        candidates = [i for i, block in enumerate(result) if len(block) > 1]
        selected = int(rng.choice(candidates))
        state = int(rng.choice(result[selected]))
        remaining = tuple(s for s in result[selected] if s != state)
        result[selected : selected + 1] = [remaining, (state,)]
    return tuple(result)
