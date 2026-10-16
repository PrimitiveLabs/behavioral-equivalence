"""Prespecified intrinsic video-duration groups, without response information."""

import numpy as np


def duration_groups(duration_ms: np.ndarray) -> np.ndarray:
    duration = np.asarray(duration_ms, dtype=float)
    if duration.ndim != 1:
        raise ValueError("One-dimensional duration vector required")
    result = np.searchsorted(np.array([7000, 18000, 60000]), duration, side="left").astype(np.int8)
    result[(~np.isfinite(duration)) | (duration <= 0)] = -1
    return result
