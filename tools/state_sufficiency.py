"""Descriptive within-state response heterogeneity, preserving original users."""

import numpy as np


def cell_totals(users, states, categories, targets, predictions, category_count):
    u, s, c, y, p = map(np.asarray, (users, states, categories, targets, predictions))
    if (
        u.ndim != 1
        or any(a.shape != u.shape for a in [s, c, y, p])
        or any(a.dtype.kind not in "iu" for a in [u, s, c])
        or np.any(u < 0)
        or np.any(s < 0)
        or np.any(c < 0)
        or np.any(c >= category_count)
        or type(category_count) is not int
        or category_count < 1
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError(
            "Aligned original users, states, categories, binary outcomes and probabilities required"
        )
    if not len(u):
        return []
    cells = np.empty(len(u), dtype=[("state", s.dtype), ("category", c.dtype)])
    cells["state"], cells["category"] = s, c
    keys = np.empty(len(u), dtype=[("state", s.dtype), ("category", c.dtype), ("user", u.dtype)])
    keys["state"], keys["category"], keys["user"] = s, c, u
    pairs, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    observed = np.bincount(inverse, weights=y, minlength=len(pairs)) / counts
    predicted = np.bincount(inverse, weights=p, minlength=len(pairs)) / counts
    unique, case_counts = np.unique(cells, return_counts=True)
    pair_cells = np.empty(len(pairs), dtype=cells.dtype)
    pair_cells["state"], pair_cells["category"] = pairs["state"], pairs["category"]
    grouped = np.searchsorted(unique, pair_cells)
    user_counts = np.bincount(grouped, minlength=len(unique))
    observed_sums = np.bincount(grouped, weights=observed, minlength=len(unique))
    predicted_sums = np.bincount(grouped, weights=predicted, minlength=len(unique))
    result = [
        dict(
            state=int(cell["state"]),
            category=int(cell["category"]),
            cases=int(case_counts[i]),
            users=int(user_counts[i]),
            user_observed_sum=float(observed_sums[i]),
            user_predicted_sum=float(predicted_sums[i]),
        )
        for i, cell in enumerate(unique)
    ]
    return result


def combine_cells(blocks, minimum_users=50, minimum_cases=2000):
    totals = {}
    for block in blocks:
        for row in block:
            key = (row["state"], row["category"])
            if key not in totals:
                totals[key] = {
                    k: 0 for k in ["cases", "users", "user_observed_sum", "user_predicted_sum"]
                }
            for k in totals[key]:
                totals[key][k] += row[k]
    rows = []
    for (state, category), r in sorted(totals.items()):
        supported = r["users"] >= minimum_users and r["cases"] >= minimum_cases
        observed = r["user_observed_sum"] / r["users"]
        predicted = r["user_predicted_sum"] / r["users"]
        rows.append(
            dict(
                state=state,
                category=category,
                cases=r["cases"],
                users=r["users"],
                supported=supported,
                observed_equal_user_rate=observed if supported else None,
                predicted_equal_user_rate=predicted if supported else None,
                residual=observed - predicted if supported else None,
            )
        )
    spans = []
    for state in sorted({r["state"] for r in rows}):
        supported = [r for r in rows if r["state"] == state and r["supported"]]
        if len(supported) >= 2:
            rates = [r["observed_equal_user_rate"] for r in supported]
            spans.append(
                dict(
                    state=state,
                    supported_categories=len(supported),
                    observed_rate_span=max(rates) - min(rates),
                )
            )
    return dict(
        cells=rows,
        within_state_spans=spans,
        scope="Descriptive equal-user rates conditional on state and history/context stratum. No p-values, causal effects, Markov sufficiency proof or adjusted discovery claim. Input blocks must have disjoint users.",
    )
