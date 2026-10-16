"""Two-dimensional synthetic response surfaces with a structural negative control."""

import numpy as np
from .core import Chain
from .synthetic import World

SURFACES = ("linear", "localized", "oscillatory", "negative_control")


def environment_chain(world: World, point: tuple[float, float], surface: str) -> Chain:
    """Apply a fixed feasible environment, with neutral origin equal to training.

    Surfaces are synthetic assumptions, not psychological mechanisms. The
    negative control redistributes probability within ground-truth blocks while
    preserving all block transition probabilities and observable path laws.
    """
    x = np.asarray(point, dtype=float)
    if (
        surface not in SURFACES
        or x.shape != (2,)
        or not np.isfinite(x).all()
        or (np.abs(x) > 1).any()
    ):
        raise ValueError("A documented surface and finite point in [-1,1]^2 required")
    p = world.chain.transitions.copy()
    if surface == "negative_control":
        if world.perturbation != 0:
            raise ValueError("Structural negative control requires an exactly lumpable source")
        shifted = np.arange(world.chain.size).reshape(-1, 4)[:, [1, 2, 3, 0]].ravel()
        amount = float(np.abs(x).max())
        p = (1 - amount) * p + amount * p[:, shifted]
        return Chain(p, world.chain.labels, world.chain.initial)
    rng = np.random.default_rng([world.seed, 81027])
    directions = rng.normal(size=(world.chain.size, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    centers = rng.uniform(-0.8, 0.8, size=(world.chain.size, 2))
    phases = rng.uniform(-np.pi, np.pi, size=world.chain.size)
    if surface == "linear":
        response = np.tanh(2 * np.einsum("ij,j->i", directions, x))
    elif surface == "localized":
        response = np.exp(-((centers - x) ** 2).sum(axis=1) / 0.18) - np.exp(
            -(centers**2).sum(axis=1) / 0.18
        )
    else:
        response = (
            np.sin(3 * np.pi * np.einsum("ij,j->i", directions, x) + phases) - np.sin(phases)
        ) / 2
    complete = next(i for i, label in enumerate(world.chain.labels) if label == "complete")
    disengage = next(i for i, label in enumerate(world.chain.labels) if label == "disengage")
    for s, label in enumerate(world.chain.labels):
        if label == "active":
            amount = 0.3 * abs(float(response[s]))
            p[s] *= 1 - amount
            p[s, complete if response[s] >= 0 else disengage] += amount
    return Chain(p, world.chain.labels, world.chain.initial)


def environment_catalogues() -> tuple[np.ndarray, np.ndarray]:
    """Fixed 64-point search catalogue and separate 16-point evaluation set."""
    grid = np.linspace(-1, 1, 8)
    search = np.array([(x, y) for x in grid for y in grid])
    heldout = np.random.default_rng(20261016).uniform(-1, 1, (16, 2))
    return search, heldout


def interpolate_environment(
    observed_points: np.ndarray,
    observed: list[Chain],
    point: tuple[float, float],
    bandwidth: float = 0.5,
) -> Chain:
    """Convex radial interpolation from queried kernels, never held-out kernels.

    The neutral training kernel is an ordinary supplied observation. This is a
    deliberately simple response-surface learner shared by all search methods.
    Its error is distinct from the certified quotient approximation error.
    """
    points = np.asarray(observed_points, dtype=float)
    x = np.asarray(point, dtype=float)
    if (
        points.shape != (len(observed), 2)
        or not observed
        or x.shape != (2,)
        or not np.isfinite(points).all()
        or not np.isfinite(x).all()
        or not np.isfinite(bandwidth)
        or bandwidth <= 0
        or any(
            c.labels != observed[0].labels or not np.array_equal(c.initial, observed[0].initial)
            for c in observed
        )
    ):
        raise ValueError(
            "Aligned observed kernels, finite coordinates and positive bandwidth required"
        )
    distance = ((points - x) ** 2).sum(axis=1)
    exact = np.flatnonzero(distance == 0)
    if len(exact):
        return observed[int(exact[0])]
    log_weights = -distance / (2 * bandwidth**2)
    weights = np.exp(log_weights - log_weights.max())
    weights /= weights.sum()
    p = sum(float(weight) * chain.transitions for weight, chain in zip(weights, observed))
    return Chain(p, observed[0].labels, observed[0].initial)
