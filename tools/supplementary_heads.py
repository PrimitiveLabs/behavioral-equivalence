"""Coherent categorical probes for frozen representations, no encoder fitting."""

import numpy as np
from scipy.special import softmax


def predict_head(features, parameters):
    x = np.asarray(features, dtype=float)
    mean, scale, coefficients, intercept = [
        np.asarray(parameters[k]) for k in ["mean", "scale", "coefficients", "intercept"]
    ]
    if (
        x.ndim != 2
        or mean.ndim != 1
        or x.shape[1] != len(mean)
        or scale.shape != mean.shape
        or coefficients.ndim != 2
        or coefficients.shape[1] != len(mean)
        or len(coefficients) < 2
        or intercept.shape != (len(coefficients),)
        or any(not np.isfinite(v).all() for v in [x, mean, scale, coefficients, intercept])
        or np.any(scale <= 0)
    ):
        raise ValueError("Finite compatible categorical-head features and parameters required")
    return softmax(((x - mean) / scale) @ coefficients.T + intercept, axis=1)


def fit_head(features, target, users, classes, *, C=1.0):
    if not isinstance(classes, int) or isinstance(classes, bool) or classes not in [2, 4]:
        raise ValueError("Two or four outcome classes required")
    x, y, u = map(np.asarray, (features, target, users))
    if (
        x.ndim != 2
        or not len(x)
        or not x.shape[1]
        or not np.isfinite(x).all()
        or y.shape != (len(x),)
        or y.dtype.kind not in "iu"
        or not np.isin(y, range(-1, classes)).all()
        or u.shape != (len(x),)
        or u.dtype.kind not in "iu"
        or np.any(u < 0)
        or not isinstance(classes, int)
        or isinstance(classes, bool)
        or classes not in [2, 4]
        or not np.isfinite(C)
        or C <= 0
    ):
        raise ValueError(
            "Aligned finite train features, integer labels/users and 2/4 classes required"
        )
    observed = y >= 0
    if not observed.any():
        raise ValueError("No observed training targets")
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning

    mean = x[observed].mean(axis=0)
    scale = x[observed].std(axis=0)
    scale[scale < 1e-12] = 1
    design = (x[observed] - mean) / scale
    _, inverse = np.unique(u[observed], return_inverse=True)
    weight = 1 / np.bincount(inverse)[inverse]
    weight *= observed.sum() / weight.sum()
    # Fixed half-count pseudo-observations at the training feature mean keep
    # absent categories explicit and permit coherent all-class predictions.
    design = np.vstack([design, np.zeros((classes, x.shape[1]))])
    labels = np.r_[y[observed], np.arange(classes)]
    weight = np.r_[weight, np.full(classes, 0.5)]
    estimator = LogisticRegression(
        C=C, solver="newton-cholesky", max_iter=500, tol=1e-6, random_state=20261015
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(design, labels, sample_weight=weight)
    coefficients, intercept = estimator.coef_, estimator.intercept_
    if classes == 2:
        coefficients = np.vstack([np.zeros_like(coefficients), coefficients])
        intercept = np.r_[0.0, intercept]
    parameters = dict(mean=mean, scale=scale, coefficients=coefficients, intercept=intercept)
    return parameters, dict(
        observed_cases=int(observed.sum()),
        observed_users=len(np.unique(u[observed])),
        iterations=int(max(estimator.n_iter_)),
        classes=classes,
        smoothing="0.5 training-mean pseudo-observation per category",
        weighting="equal users within observed endpoint training cases",
    )


def categorical_scores(probabilities, target, users, floor=1e-6):
    """Paired-report building block: equal original users, endpoint-observed only."""
    p, y, u = map(np.asarray, (probabilities, target, users))
    if (
        p.ndim != 2
        or p.shape[1] not in [2, 4]
        or y.shape != (len(p),)
        or u.shape != (len(p),)
        or y.dtype.kind not in "iu"
        or u.dtype.kind not in "iu"
        or (u < 0).any()
        or not np.isin(y, range(-1, p.shape[1])).all()
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)
        or not np.isfinite(floor)
        or not 0 < floor < 1 / p.shape[1]
    ):
        raise ValueError("Valid categorical probabilities, endpoint labels and user IDs required")
    keep = y >= 0
    if not keep.any():
        return dict(cases=0, users=0, nll=None, brier=None)
    p = np.maximum(p[keep], floor)
    p /= p.sum(axis=1, keepdims=True)
    labels = y[keep]
    ids, inverse = np.unique(u[keep], return_inverse=True)
    nll = -np.log(p[np.arange(len(p)), labels])
    brier = np.sum((p - np.eye(p.shape[1])[labels]) ** 2, axis=1)
    counts = np.bincount(inverse)
    return dict(
        cases=int(keep.sum()),
        users=len(ids),
        nll=float(np.mean(np.bincount(inverse, weights=nll) / counts)),
        brier=float(np.mean(np.bincount(inverse, weights=brier) / counts)),
    )
