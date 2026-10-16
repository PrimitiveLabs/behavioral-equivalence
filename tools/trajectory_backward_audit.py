"""Independent scaled backward likelihood for numerical trajectory replay."""

import numpy as np


def backward_nll(transitions, emissions, states, outcomes):
    t, e = np.asarray(transitions), np.asarray(emissions)
    y = np.asarray(outcomes)
    if not y.shape[1]:
        return np.zeros(len(y))
    value = np.ones((len(y), len(e)))
    log_scale = np.zeros(len(y))
    for step in range(y.shape[1] - 1, -1, -1):
        observed = y[:, step]
        following = np.empty_like(value)
        for outcome in [0, 1]:
            keep = observed == outcome
            following[keep] = (
                value[keep] @ t[:, outcome, :].T if step + 1 < y.shape[1] else value[keep]
            ) * e[:, outcome]
        scale = following.max(axis=1)
        positive = scale > 0
        following[positive] /= scale[positive, None]
        with np.errstate(divide="ignore"):
            log_scale += np.log(scale)
        value = following
    with np.errstate(divide="ignore"):
        return -np.log(value[np.arange(len(y)), states]) - log_scale
