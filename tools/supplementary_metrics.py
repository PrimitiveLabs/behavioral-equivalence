"""User-disjoint categorical scoring blocks, including per-category calibration."""

import numpy as np
from behavioral_equivalence.streaming_scores import merge_metric_blocks


def calibration_cells(probabilities, indicators, users, inverse, counts):
    bins = np.minimum((probabilities * 20).astype(int), 19)
    weights = 1 / counts[inverse]
    cells = []
    for index in range(20):
        chosen = bins == index
        cells.append(
            dict(
                cases=int(chosen.sum()),
                users=len(np.unique(users[chosen])),
                user_mass=float(weights[chosen].sum()),
                predicted_sum=float(weights[chosen] @ probabilities[chosen]),
                observed_sum=float(weights[chosen] @ indicators[chosen]),
            )
        )
    return cells


def categorical_block(users, target, probabilities, floor=1e-6):
    u, y, p = map(np.asarray, (users, target, probabilities))
    if (
        p.ndim != 2
        or p.shape[1] not in [2, 4]
        or u.shape != (len(p),)
        or y.shape != u.shape
        or u.dtype.kind not in "iu"
        or y.dtype.kind not in "iu"
        or np.any(u < 0)
        or not np.isin(y, range(-1, p.shape[1])).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)
        or not np.isfinite(floor)
        or not 0 < floor < 1 / p.shape[1]
    ):
        raise ValueError(
            "Aligned categorical probabilities, endpoint labels and original users required"
        )
    keep = y >= 0
    u, y, p = u[keep], y[keep], p[keep]
    unique, inverse = np.unique(u, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique))
    q = np.maximum(p, floor)
    q /= q.sum(axis=1, keepdims=True)
    nll = -np.log(q[np.arange(len(q)), y])
    brier = np.sum((q - np.eye(p.shape[1])[y]) ** 2, axis=1)
    top = calibration_cells(p.max(axis=1), p.argmax(axis=1) == y, u, inverse, counts)
    return dict(
        user_ids=unique,
        case_counts=counts,
        user_nll=np.bincount(inverse, weights=nll, minlength=len(unique)) / counts,
        user_brier=np.bincount(inverse, weights=brier, minlength=len(unique)) / counts,
        case_nll_sum=float(nll.sum()),
        case_brier_sum=float(brier.sum()),
        calibration=top,
        class_calibration=[
            calibration_cells(p[:, j], y == j, u, inverse, counts) for j in range(p.shape[1])
        ],
        missing_cases=int((~keep).sum()),
        classes=p.shape[1],
    )


def merge_categorical_blocks(blocks):
    if not blocks:
        raise ValueError("At least one declared categorical source bucket required")
    classes = blocks[0]["classes"]
    if any(b["classes"] != classes for b in blocks):
        raise ValueError("Endpoint class counts differ across buckets")
    result = merge_metric_blocks(blocks)
    metrics = result["metrics"]
    users = metrics["users"]
    categories = {}
    for category in range(classes):
        cells = []
        for index in range(20):
            rows = [b["class_calibration"][category][index] for b in blocks]
            mass = sum(r["user_mass"] for r in rows)
            cells.append(
                dict(
                    lower=index / 20,
                    upper=(index + 1) / 20,
                    cases=sum(r["cases"] for r in rows),
                    users=sum(r["users"] for r in rows),
                    equal_user_weight=mass / users if users else 0,
                    predicted=sum(r["predicted_sum"] for r in rows) / mass if mass else None,
                    observed=sum(r["observed_sum"] for r in rows) / mass if mass else None,
                )
            )
        categories[str(category)] = cells
    metrics.update(
        class_calibration_bins=categories,
        top_label_calibration_bins=metrics.pop("calibration_bins"),
        missing_cases=sum(b["missing_cases"] for b in blocks),
        classes=classes,
        brier_convention="Sum over all category squared errors; binary value is twice positive-class Brier",
        probability_floor_rule="Clip each category below at 1e-6 then renormalize for proper losses; calibration uses original probabilities",
    )
    return result
