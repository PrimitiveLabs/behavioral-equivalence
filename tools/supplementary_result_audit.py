"""Independent sorted-user categorical forecast, calibration and bootstrap arithmetic."""

import numpy as np


def _cells(p, y, user_index, counts):
    index = np.minimum(np.floor(p * 20).astype(np.int64), 19)
    weights = 1 / np.repeat(counts, counts)
    user_bins = np.unique(user_index * 20 + index) % 20
    return dict(
        cases=np.bincount(index, minlength=20),
        users=np.bincount(user_bins, minlength=20),
        mass=np.bincount(index, weights=weights, minlength=20),
        predicted=np.bincount(index, weights=weights * p, minlength=20),
        observed=np.bincount(index, weights=weights * y, minlength=20),
    )


def categorical_audit_block(users, targets, probabilities):
    u, y, p = map(np.asarray, (users, targets, probabilities))
    if (
        u.ndim != 1
        or u.dtype.kind not in "iu"
        or np.any(u < 0)
        or y.shape != u.shape
        or y.dtype.kind not in "iu"
        or p.ndim != 2
        or len(p) != len(u)
        or p.shape[1] not in [2, 4]
        or not np.isin(y, range(-1, p.shape[1])).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError(
            "Aligned original users, categorical outcomes and simplex forecasts required"
        )
    original = len(y)
    valid = y != -1
    u, y, p = u[valid], y[valid], p[valid]
    order = np.argsort(u, kind="stable")
    u, y, p = u[order], y[order], p[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]]) if len(u) else np.empty(0, dtype=int)
    counts = np.diff(np.r_[starts, len(u)]) if len(u) else np.empty(0, dtype=int)
    user_index = np.repeat(np.arange(len(starts)), counts)
    q = p.copy().astype(float)
    q[q < 1e-6] = 1e-6
    q = q / q.sum(axis=1)[:, None]
    truth = np.take_along_axis(q, y[:, None], axis=1).ravel()
    loss = -np.log(truth)
    squared = np.square(q).sum(axis=1) - 2 * truth + 1
    return dict(
        users=u[starts],
        counts=counts,
        nll=np.add.reduceat(loss, starts) / counts if len(u) else np.empty(0),
        brier=np.add.reduceat(squared, starts) / counts if len(u) else np.empty(0),
        nll_sum=float(loss.sum()),
        brier_sum=float(squared.sum()),
        missing_cases=original - len(y),
        class_cells=[_cells(p[:, j], y == j, user_index, counts) for j in range(p.shape[1])],
        top_cells=_cells(p.max(axis=1), np.argmax(p, axis=1) == y, user_index, counts),
    )


def verify_calibration(expected, blocks, users, tolerance=1e-10):
    maximum = 0.0
    groups = [(expected["top_label_calibration_bins"], [b["top_cells"] for b in blocks])]
    groups += [
        (rows, [b["class_cells"][int(category)] for b in blocks])
        for category, rows in expected["class_calibration_bins"].items()
    ]
    for rows, cells in groups:
        total = {
            key: sum((c[key] for c in cells), np.zeros(20))
            for key in ["cases", "users", "mass", "predicted", "observed"]
        }
        if len(rows) != 20:
            raise ValueError("All20 calibration bins required")
        for i, row in enumerate(rows):
            if row["cases"] != int(total["cases"][i]) or row["users"] != int(total["users"][i]):
                raise ValueError("Calibration original case/user support differs")
            mass = total["mass"][i]
            values = dict(
                equal_user_weight=mass / users if users else 0.0,
                predicted=total["predicted"][i] / mass if mass else None,
                observed=total["observed"][i] / mass if mass else None,
            )
            for key, value in values.items():
                if value is None or row[key] is None:
                    if value != row[key]:
                        raise ValueError("Empty calibration cell differs")
                else:
                    if not np.isfinite(value) or not np.isfinite(row[key]):
                        raise ValueError("Finite reported and replayed calibration values required")
                    difference = abs(value - row[key])
                    maximum = max(maximum, difference)
                    if difference > tolerance:
                        raise ValueError("Replayed calibration cell differs: " + key)
    if expected["missing_cases"] != sum(b["missing_cases"] for b in blocks):
        raise ValueError("Missing endpoint support differs")
    return maximum


def bootstrap_many(differences, *, replications, seed, confidence=0.95):
    x = np.asarray(differences, dtype=float)
    if (
        x.ndim != 2
        or len(x) < 2
        or not x.shape[1]
        or not np.isfinite(x).all()
        or type(replications) is not int
        or replications < 100
        or not 0 < confidence < 1
    ):
        raise ValueError(
            "Finite paired original-user difference matrix and bootstrap settings required"
        )
    rng = np.random.default_rng(seed)
    values = np.empty((replications, x.shape[1]))
    for start in range(0, replications, 8):
        n = min(8, replications - start)
        indices = rng.integers(0, len(x), size=(n, len(x)))
        values[start : start + n] = x[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    return np.quantile(values, [tail, 1 - tail], axis=0)
