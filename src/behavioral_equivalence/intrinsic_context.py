"""Intrinsic-duration context with an explicit prior-upload-date restriction."""

import numpy as np


def duration_context(duration_ms: np.ndarray, upload_day: np.ndarray, exposure_ms: np.ndarray):
    duration, upload, exposure = map(np.asarray, (duration_ms, upload_day, exposure_ms))
    if (
        duration.ndim != 1
        or upload.shape != duration.shape
        or exposure.shape != duration.shape
        or upload.dtype.kind not in "iu"
        or exposure.dtype.kind not in "iu"
    ):
        raise ValueError("Aligned duration and integer upload/exposure times required")
    known = (
        np.isfinite(duration)
        & (duration > 0)
        & (upload != np.iinfo(np.int32).min)
        & (upload < exposure // 86400000)
    )
    result = np.zeros((len(duration), 2), dtype=np.float32)
    result[known, 0] = np.log1p(duration[known] / 1000)
    result[:, 1] = ~known
    return result, known
