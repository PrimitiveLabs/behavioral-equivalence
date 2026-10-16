"""Training-only normalization, deterministic assignment and teacher emissions."""

import numpy as np


def fit_representation_scaling(training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(training, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Finite nonempty training representation matrix required")
    return x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)


def assign_centers(
    representations: np.ndarray, mean: np.ndarray, scale: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    x, mean, scale, centers = [
        np.asarray(v, dtype=np.float64) for v in (representations, mean, scale, centers)
    ]
    if (
        x.ndim != 2
        or centers.ndim != 2
        or not len(centers)
        or x.shape[1] != centers.shape[1]
        or mean.shape != (x.shape[1],)
        or scale.shape != mean.shape
        or any(not np.isfinite(v).all() for v in (x, mean, scale, centers))
        or (scale <= 0).any()
    ):
        raise ValueError("Aligned finite representations, centers and positive scales required")
    normalized = (x - mean) / scale
    # Fixed float64 recurrence; ties select the smallest centroid index.
    distance = (
        (normalized**2).sum(axis=1)[:, None]
        + (centers**2).sum(axis=1)[None, :]
        - 2 * np.einsum("nd,kd->nk", normalized, centers)
    )
    return np.argmin(distance, axis=1)


def emission_probabilities(
    counts: np.ndarray, teacher_sum: np.ndarray, global_probability: float
) -> np.ndarray:
    counts, teacher_sum = np.asarray(counts), np.asarray(teacher_sum, dtype=np.float64)
    if (
        counts.ndim != 1
        or teacher_sum.shape != counts.shape
        or counts.dtype.kind not in "iu"
        or (counts < 0).any()
        or not np.isfinite(teacher_sum).all()
        or (teacher_sum < 0).any()
        or (teacher_sum > counts).any()
        or not np.isfinite(global_probability)
        or not 0 <= global_probability <= 1
    ):
        raise ValueError("Valid training counts and teacher probability sums required")
    return np.divide(
        teacher_sum, counts, out=np.full(len(counts), global_probability), where=counts > 0
    )
