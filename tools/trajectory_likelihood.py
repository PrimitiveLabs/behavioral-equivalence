"""Scaled joint binary-sequence likelihood for the specified decision model."""

import numpy as np


def sequence_nll(transitions, emissions, initial_states, outcomes, *, return_prefix=False):
    """Return per-sequence joint NLL, including the origin outcome.

    Initial predictive states are fixed from origin-only histories. Each outcome
    is emitted before its outcome-conditioned latent transition. No observed
    future state assignments or contexts enter the forward calculation. Exact
    zero model probability returns infinity; no probability floor is introduced.
    Caller batches sequences and selects fully observed horizons independently.
    """
    t, e = np.asarray(transitions, dtype=float), np.asarray(emissions, dtype=float)
    states, y = np.asarray(initial_states), np.asarray(outcomes)
    k = len(e)
    if (
        e.shape != (k, 2)
        or k == 0
        or t.shape != (k, 2, k)
        or states.ndim != 1
        or states.dtype.kind not in "iu"
        or y.ndim != 2
        or y.shape[0] != len(states)
        or not np.isin(y, [0, 1]).all()
        or np.any(states < 0)
        or np.any(states >= k)
        or not np.isfinite(t).all()
        or not np.isfinite(e).all()
        or np.any(t < 0)
        or np.any(e < 0)
        or not np.allclose(t.sum(axis=2), 1, atol=1e-12, rtol=0)
        or not np.allclose(e.sum(axis=1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Stochastic binary decision model and aligned states/outcomes required")
    if type(return_prefix) is not bool:
        raise ValueError("Prefix output flag must be boolean")
    mass = np.eye(k)[states]
    loss = np.zeros(len(states))
    prefixes = np.empty(y.shape) if return_prefix else None
    for step in range(y.shape[1]):
        observed = y[:, step].astype(np.int64)
        mass *= e[:, observed].T
        scale = mass.sum(axis=1)
        positive = scale > 0
        loss[~positive] = np.inf
        loss[positive] -= np.log(scale[positive])
        mass[positive] /= scale[positive, None]
        if return_prefix:
            prefixes[:, step] = loss
        if step + 1 < y.shape[1]:
            following = np.zeros_like(mass)
            for outcome in [0, 1]:
                selected = observed == outcome
                following[selected] = mass[selected] @ t[:, outcome, :]
            mass = following
    return prefixes if return_prefix else loss
