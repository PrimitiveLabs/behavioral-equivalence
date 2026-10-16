"""Outcome-free case alignment and exact-normalized point forecast conventions."""

from fractions import Fraction
import numpy as np


def forecast_indices(case_ids, users, roles, user_roles, context_ids, protocol, split):
    ids, u, r, ur, context = map(np.asarray, (case_ids, users, roles, user_roles, context_ids))
    if (
        ids.ndim != 1
        or any(x.shape != ids.shape for x in (u, r, ur, context))
        or any(x.dtype.kind not in "iu" for x in (ids, u, r, ur, context))
        or np.any(ids < 0)
        or np.any(u < 0)
        or len(np.unique(ids)) != len(ids)
        or not np.array_equal(ids, context)
        or not np.isin(r, [-1, 0, 1, 2, 3]).all()
        or not np.isin(ur, [0, 1, 2, 3]).all()
        or protocol not in {"known_users", "unseen_users"}
        or split not in {"validation", "test"}
    ):
        raise ValueError(
            "Aligned immutable case/user/role/context identifiers and declared forecast split required"
        )
    code = {"validation": 1, "test": 3}[split]
    selected = r == code
    if protocol == "unseen_users":
        selected &= ur == code
    return np.flatnonzero(selected)


def available_history_lengths(tokens):
    """Count feedback-rate tokens1..16; start marker17 is an empty history."""
    x = np.asarray(tokens)
    if (
        x.ndim != 2
        or x.shape[1] < 1
        or x.dtype.kind not in "iu"
        or np.any(x < 0)
        or np.any(x > 17)
        or np.any((x[:, :-1] > 0) & (x[:, 1:] == 0))
        or np.any(np.sum(x > 0, axis=1) == 0)
    ):
        raise ValueError("Canonical right-aligned available-history token rows required")
    markers = (x == 17).any(axis=1)
    if np.any((x[markers, -1] != 17) | (np.sum(x[markers] > 0, axis=1) != 1)):
        raise ValueError("Start marker may only represent a wholly empty history")
    return ((x >= 1) & (x <= 16)).sum(axis=1)


def normalized_positive_emissions(emissions):
    """Match integer-kernel semantics for pending binary outcome rows.

    Returning the nearest float to the exact rational ratio matches saved
    next-positive forecasts; it does not issue a statistical confidence bound.
    """
    x = np.asarray(emissions, dtype=float)
    if (
        x.ndim != 2
        or x.shape[1] != 2
        or not len(x)
        or not np.isfinite(x).all()
        or np.any(x < 0)
        or not np.allclose(x.sum(axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Finite nonnegative near-stochastic binary emissions required")
    return np.array(
        [
            float(
                Fraction.from_float(float(row[1]))
                / (Fraction.from_float(float(row[0])) + Fraction.from_float(float(row[1])))
            )
            for row in x
        ]
    )


def conditional_response_forecasts(
    fine_states, environments, fine_to_coarse, latent_mapping, emissions
):
    """Apply exactly one fine/coarse/class map before fixed-context emission lookup."""
    states, env, coarse, mapping, p = map(
        np.asarray, (fine_states, environments, fine_to_coarse, latent_mapping, emissions)
    )
    if (
        states.ndim != 1
        or env.shape != states.shape
        or coarse.ndim != 1
        or mapping.ndim != 1
        or not len(coarse)
        or not len(mapping)
        or p.ndim != 2
        or p.shape[1] != 75
        or not len(p)
        or any(x.dtype.kind not in "iu" for x in (states, env, coarse, mapping))
        or np.any(states < 0)
        or np.any(states >= len(coarse))
        or np.any(env < 0)
        or np.any(env >= 75)
        or np.any(coarse < 0)
        or np.any(coarse >= len(mapping))
        or np.any(mapping < 0)
        or np.any(mapping >= len(p))
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError(
            "Aligned valid fine/coarse/class maps and fixed-context probabilities required"
        )
    return p[mapping[coarse[states]], env]
