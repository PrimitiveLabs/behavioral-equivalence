"""Disjoint user calibration and supported witnesses for observed random responses."""

import hashlib
import numpy as np
from .refinement import js_divergence


def calibration_discovery_mask(users: np.ndarray, seed: int) -> np.ndarray:
    """Assign entire users before inspecting responses; never split concurrent cases."""
    users = np.asarray(users)
    if users.ndim != 1 or users.dtype.kind not in "iu" or np.any(users < 0):
        raise ValueError("Nonnegative integer user identifiers required")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Nonnegative integer split seed required")
    unique, inverse = np.unique(users, return_inverse=True)
    selected = np.array(
        [
            hashlib.sha256(f"random-response-calibration-v1:{seed}:{int(u)}".encode()).digest()[0]
            < 128
            for u in unique
        ],
        dtype=bool,
    )
    return selected[inverse]


def response_table(states, environments, users, outcomes, state_count: int) -> dict:
    """Count observed binary responses, retaining unsupported and unknown strata.

    Counts and empirical means describe the supplied calibration observations.
    They are not confidence bounds for an independent human generating process.
    """
    states, environments, users, outcomes = map(np.asarray, (states, environments, users, outcomes))
    if (
        not isinstance(state_count, int)
        or state_count < 1
        or states.ndim != 1
        or any(x.shape != states.shape for x in (environments, users, outcomes))
        or any(x.dtype.kind not in "iu" for x in (states, environments, users))
        or np.any(states < 0)
        or np.any(states >= state_count)
        or np.any(environments < 0)
        or np.any(environments >= 75)
        or np.any(users < 0)
        or not np.isin(outcomes, [0, 1]).all()
    ):
        raise ValueError("Aligned state, scenario/group, user and binary response arrays required")
    cells = states.astype(np.int64) * 75 + environments.astype(np.int64)
    counts = np.bincount(cells, minlength=state_count * 75).reshape(state_count, 75)
    positives = np.bincount(cells, weights=outcomes, minlength=state_count * 75).reshape(
        state_count, 75
    )
    pairs = np.unique(np.rec.fromarrays([cells, users], names=("cell", "user")))
    user_counts = np.bincount(pairs["cell"], minlength=state_count * 75).reshape(state_count, 75)
    probabilities = np.full((state_count, 75), np.nan)
    np.divide(positives, counts, out=probabilities, where=counts > 0)
    return dict(counts=counts, positives=positives, users=user_counts, probabilities=probabilities)


def supported_witnesses(
    table: dict, blocks: tuple[tuple[int, ...], ...], *, min_users=50, min_cases=2000
) -> list[dict]:
    """Rank supported state-pair differences within one fixed scenario/item group.

    All four known-duration groups are enumerated, with explicit query counts.
    Ranking a calibration table is model selection, not new response acquisition.
    """
    counts, users, probabilities = (
        np.asarray(table[k]) for k in ("counts", "users", "probabilities")
    )
    if (
        counts.ndim != 2
        or counts.shape[1] != 75
        or users.shape != counts.shape
        or probabilities.shape != counts.shape
        or min_users < 1
        or min_cases < 1
        or np.any(counts < 0)
        or np.any(users < 0)
        or np.any(users > counts)
        or not np.isfinite(probabilities[counts > 0]).all()
        or np.any(probabilities[counts > 0] < 0)
        or np.any(probabilities[counts > 0] > 1)
    ):
        raise ValueError("Valid aligned response table and positive support thresholds required")
    flat = [s for b in blocks for s in b]
    if any(not b for b in blocks) or sorted(flat) != list(range(len(counts))):
        raise ValueError("Complete disjoint nonempty state partition required")
    result = []
    for block in blocks:
        for i, a in enumerate(block):
            for b in block[i + 1 :]:
                for env in range(75):
                    if env % 5 == 4:
                        continue
                    if (
                        min(counts[a, env], counts[b, env]) < min_cases
                        or min(users[a, env], users[b, env]) < min_users
                    ):
                        continue
                    p, q = float(probabilities[a, env]), float(probabilities[b, env])
                    result.append(
                        dict(
                            states=(a, b),
                            scenario=env // 5,
                            duration_group=env % 5,
                            environment=env,
                            js_nats=js_divergence(np.array([p, 1 - p]), np.array([q, 1 - q])),
                            probability_gap=abs(p - q),
                        )
                    )
    return sorted(result, key=lambda w: (-w["js_nats"], w["environment"], w["states"]))
