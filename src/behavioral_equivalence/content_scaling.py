"""Training-only normalization of intrinsic duration, preserving missingness."""

import numpy as np


def _context(values):
    x = np.asarray(values, dtype=np.float32)
    if (
        x.ndim != 2
        or x.shape[1] != 2
        or not np.isfinite(x).all()
        or not np.isin(x[:, 1], [0, 1]).all()
    ):
        raise ValueError("Finite two-column duration/missing context required")
    return x


def fit_duration_scaling(training):
    x = _context(training)
    known = x[x[:, 1] == 0, 0].astype(float)
    return {
        "mean": float(known.mean()) if len(known) else 0.0,
        "scale": max(float(known.std()), 1e-6) if len(known) else 1.0,
    }


def scale_duration(values, scaling):
    x = _context(values).copy()
    if not np.isfinite([scaling["mean"], scaling["scale"]]).all() or scaling["scale"] <= 0:
        raise ValueError("Finite normalization and positive scale required")
    known = x[:, 1] == 0
    x[known, 0] = (x[known, 0] - scaling["mean"]) / scaling["scale"]
    x[~known, 0] = 0
    return x
