"""One-step forecasts for the pooled calibration-count model."""

from fractions import Fraction
import numpy as np


def pooled_emission_forecasts(counts, lower, upper, states):
    """Return nominal emissions and bounds at immutable assigned origin states.

    Unsupported empirical rows use the declared neutral nominal value 0.5.
    Their interval remains the supplied uncertainty interval, typically [0,1].
    The forecast estimand differs from training-teacher probabilities.
    """
    c, lo, hi, s = map(np.asarray, (counts, lower, upper, states))
    k = c.shape[0] if c.ndim == 3 else 0
    if (
        k < 1
        or c.shape != (k, 2, k)
        or c.dtype.kind not in "iu"
        or np.any(c < 0)
        or lo.shape != (3 * k, 3 * k)
        or hi.shape != lo.shape
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(lo < 0)
        or np.any(hi > 1)
        or np.any(lo > hi)
        or s.ndim != 1
        or s.dtype.kind not in "iu"
        or np.any(s < 0)
        or np.any(s >= k)
    ):
        raise ValueError("Aligned pooled counts, lifted intervals and valid origin states required")
    totals = c.astype(object).sum(axis=(1, 2))
    positive = c.astype(object)[:, 1].sum(axis=1)
    nominal = np.array(
        [float(Fraction(int(y), int(n))) if n else 0.5 for y, n in zip(positive, totals)]
    )
    return nominal[s], lo[3 * s, 3 * s + 2], hi[3 * s, 3 * s + 2]
