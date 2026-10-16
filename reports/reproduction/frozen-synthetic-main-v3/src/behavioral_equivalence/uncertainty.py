"""Row-rectangular interval chains and finite-horizon robust reachability."""

from dataclasses import dataclass
import math
import numpy as np
from .core import FloatArray


@dataclass(frozen=True)
class IntervalChain:
    lower: FloatArray
    upper: FloatArray
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        lo = np.array(self.lower, dtype=float, copy=True)
        hi = np.array(self.upper, dtype=float, copy=True)
        n = len(self.labels)
        if lo.shape != (n, n) or hi.shape != (n, n) or n == 0:
            raise ValueError("Interval dimensions must match labels")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise ValueError("Intervals must be finite")
        if (lo < 0).any() or (hi > 1).any() or (lo > hi).any():
            raise ValueError("Invalid probability interval")
        if (lo.sum(axis=1) > 1 + 1e-12).any() or (hi.sum(axis=1) < 1 - 1e-12).any():
            raise ValueError("Interval row has no stochastic member")
        lo.setflags(write=False)
        hi.setflags(write=False)
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)


def multinomial_intervals(
    counts: FloatArray, labels: tuple[str, ...], alpha: float = 0.05
) -> IntervalChain:
    """Simultaneous coordinate Hoeffding intervals for independently sampled rows.

    Each row consists of fixed-count independent categorical trials. Union bound
    covers all n*n coordinates. No independence between coordinates is needed.
    No claim is made for adaptive human histories or learned-state selection.
    """
    x = np.asarray(counts, dtype=float)
    n = len(labels)
    if x.shape != (n, n) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Counts must be a nonnegative finite square matrix")
    if (x != np.floor(x)).any() or not 0 < alpha < 1:
        raise ValueError("Integer counts and alpha in (0,1) required")
    totals = x.sum(axis=1)
    empirical = np.divide(x, totals[:, None], out=np.zeros_like(x), where=totals[:, None] > 0)
    radius = np.sqrt(math.log(2 * n * n / alpha) / (2 * np.maximum(totals, 1)))
    lo = np.maximum(0, empirical - radius[:, None])
    hi = np.minimum(1, empirical + radius[:, None])
    lo[totals == 0] = 0
    hi[totals == 0] = 1
    return IntervalChain(lo, hi, labels)


def row_extreme(lower: FloatArray, upper: FloatArray, values: FloatArray, maximize: bool) -> float:
    """Solve a bounded-simplex linear objective by allocating remaining mass."""
    probabilities = lower.copy()
    residual = 1 - float(probabilities.sum())
    order = np.argsort(values, kind="stable")
    if maximize:
        order = order[::-1]
    for destination in order:
        allocation = min(max(0, residual), float(upper[destination] - lower[destination]))
        probabilities[destination] += allocation
        residual -= allocation
    if residual > 1e-10:
        raise ValueError("Infeasible interval row")
    return float(probabilities @ values)


def robust_reachability(
    model: IntervalChain, target: str, horizon: int
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
        raise ValueError("Horizon must be a nonnegative integer")
    goal = np.array([label == target for label in model.labels])
    lower = goal.astype(float)
    upper = lower.copy()
    for _ in range(horizon):
        lower = np.array(
            [
                1 if goal[s] else row_extreme(lo, hi, lower, False)
                for s, (lo, hi) in enumerate(zip(model.lower, model.upper))
            ]
        )
        upper = np.array(
            [
                1 if goal[s] else row_extreme(lo, hi, upper, True)
                for s, (lo, hi) in enumerate(zip(model.lower, model.upper))
            ]
        )
    return lower, upper


def clopper_pearson_intervals(
    counts: FloatArray, labels: tuple[str, ...], alpha: float = 0.05
) -> IntervalChain:
    """Simultaneous exact-binomial coordinate intervals for fixed iid row samples.

    Bonferroni covers all n*n multinomial coordinates. Beta quantiles are
    computed numerically; the mathematical construction has finite-sample
    coverage, while directed-rounding certification remains separate.
    """
    from scipy.stats import beta

    x = np.asarray(counts, dtype=float)
    n = len(labels)
    if n < 1 or x.shape != (n, n) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Nonnegative finite square counts required")
    if (x != np.floor(x)).any() or not 0 < alpha < 1:
        raise ValueError("Integer counts and alpha in (0,1) required")
    totals = x.sum(axis=1)[:, None]
    total_grid = np.broadcast_to(totals, x.shape)
    tail = alpha / (2 * n * n)
    lower = np.zeros_like(x)
    upper = np.ones_like(x)
    positive = x > 0
    below_total = x < total_grid
    lower[positive] = beta.ppf(tail, x[positive], total_grid[positive] - x[positive] + 1)
    upper[below_total] = beta.ppf(
        1 - tail, x[below_total] + 1, total_grid[below_total] - x[below_total]
    )
    return IntervalChain(lower, upper, labels)
