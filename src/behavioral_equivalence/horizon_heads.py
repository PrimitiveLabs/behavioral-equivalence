"""Training-only logistic event heads on frozen origin features."""

import warnings
import numpy as np
from scipy.special import expit
from .nested_design import human_event_suite
from .quantization import fit_representation_scaling


def fit_horizon_heads(
    features, targets, users, horizons, *, C=1.0, max_iter=500, tol=1e-6, solver="newton-cholesky"
):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    events = human_event_suite(tuple(horizons))
    x, y, u = map(np.asarray, (features, targets, users))
    if (
        x.ndim != 2
        or not len(x)
        or not x.shape[1]
        or not np.isfinite(x).all()
        or y.shape != (len(x), len(events))
        or not np.isin(y, [-1, 0, 1]).all()
        or u.shape != (len(x),)
        or u.dtype.kind not in "iu"
        or np.any(u < 0)
        or not np.isfinite(C)
        or C <= 0
        or not isinstance(max_iter, int)
        or max_iter < 1
        or not np.isfinite(tol)
        or tol <= 0
    ):
        raise ValueError(
            "Aligned finite origin features, event targets and user identifiers required"
        )
    mean, scale = fit_representation_scaling(x)
    standardized = (x - mean) / scale
    k = len(events)
    coefficients = np.zeros((k, x.shape[1]))
    intercepts = np.zeros(k)
    is_constant = np.zeros(k, dtype=bool)
    constants = np.zeros(k)
    records = []
    first = next(
        (i for i, e in enumerate(events) if e.mode == "occupancy" and e.horizon == 1), None
    )
    for i, event in enumerate(events):
        observed = y[:, i] >= 0
        response = y[observed, i]
        ids, inverse = np.unique(u[observed], return_inverse=True)
        h = (event.horizon + 1) // 2
        row = dict(
            event_index=i,
            observed_cases=int(observed.sum()),
            observed_users=len(ids),
            positives=int(response.sum()),
        )
        if event.mode == "count_at_least" and event.threshold > h:
            if np.any(response != 0):
                raise ValueError("An impossible event has a positive training label")
            is_constant[i] = True
            row.update(kind="structural_zero", iterations=0)
        elif event.horizon == 1 and event.mode == "reachability":
            if first is None or not np.array_equal(observed, y[:, first] >= 0):
                raise ValueError("H1 head reuse requires identical observed origin support")
            complement = event.targets == ("negative",)
            expected = 1 - y[observed, first] if complement else y[observed, first]
            if not np.array_equal(response, expected):
                raise ValueError("H1 event labels contradict the occupancy identity")
            sign = -1 if complement else 1
            coefficients[i] = sign * coefficients[first]
            intercepts[i] = sign * intercepts[first]
            is_constant[i] = is_constant[first]
            constants[i] = 1 - constants[first] if complement else constants[first]
            row.update(
                kind="complement" if complement else "reuse", source_head=first, iterations=0
            )
        elif not len(response):
            raise ValueError("No observed training target for a nonstructural event")
        elif len(np.unique(response)) == 1:
            is_constant[i] = True
            constants[i] = (float(response.sum()) + 0.5) / (len(response) + 1)
            row.update(kind="jeffreys_constant", probability=float(constants[i]), iterations=0)
        else:
            weight = 1 / np.bincount(inverse)[inverse]
            weight *= len(response) / weight.sum()
            estimator = LogisticRegression(
                C=C, solver=solver, max_iter=max_iter, tol=tol, random_state=20261015
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(standardized[observed], response, sample_weight=weight)
            coefficients[i] = estimator.coef_[0]
            intercepts[i] = estimator.intercept_[0]
            row.update(kind="logistic", iterations=int(estimator.n_iter_[0]))
        records.append(row)
    return dict(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        intercepts=intercepts,
        is_constant=is_constant,
        constants=constants,
    ), records


def predict_horizon_heads(features, parameters):
    x = np.asarray(features)
    mean, scale, weights, intercepts, flags, constants = [
        np.asarray(parameters[k])
        for k in ["mean", "scale", "coefficients", "intercepts", "is_constant", "constants"]
    ]
    if (
        x.ndim != 2
        or mean.ndim != 1
        or x.shape[1] != len(mean)
        or scale.shape != mean.shape
        or weights.ndim != 2
        or weights.shape[1] != len(mean)
        or not len(weights)
        or any(a.shape != (len(weights),) for a in [intercepts, flags, constants])
        or flags.dtype.kind != "b"
        or any(not np.isfinite(a).all() for a in [x, mean, scale, weights, intercepts, constants])
        or np.any(scale <= 0)
        or np.any(constants < 0)
        or np.any(constants > 1)
    ):
        raise ValueError(
            "Finite aligned frozen horizon-head parameters and origin features required"
        )
    probabilities = expit(((x - mean) / scale) @ weights.T + intercepts)
    probabilities[:, flags] = constants[flags]
    return probabilities
