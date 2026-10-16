"""Reconstruct ordered response sequences against audited horizon aggregates."""

import numpy as np


def reconstruct_sequences(users, roles, outcomes, origins, observed, counts, last, horizons):
    u, r, y, ix, mask, total, final = map(
        np.asarray, (users, roles, outcomes, origins, observed, counts, last)
    )
    hs = tuple(horizons)
    if (
        u.ndim != 1
        or r.shape != u.shape
        or y.shape != u.shape
        or ix.ndim != 1
        or ix.dtype.kind not in "iu"
        or np.any(ix < 0)
        or np.any(ix >= len(u))
        or np.any(ix[1:] <= ix[:-1])
        or not hs
        or any(type(h) is not int or h < 1 for h in hs)
        or len(set(hs)) != len(hs)
        or any(a.shape != (len(ix), len(hs)) for a in [mask, total, final])
        or mask.dtype.kind != "b"
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Aligned binary bundles and audited horizon arrays required")
    sequences = np.full((len(ix), max(hs)), -1, dtype=np.int8)
    # Explicit slice checks independently validate the stored aggregate targets.
    for row, origin in enumerate(ix):
        for column, h in enumerate(hs):
            end = int(origin) + h
            valid = end <= len(u) and bool(
                np.all(u[origin:end] == u[origin]) and np.all(r[origin:end] == 3)
            )
            if bool(mask[row, column]) != valid:
                raise ValueError("Stored horizon support differs from original bundle sequence")
            if valid:
                window = y[origin:end]
                if int(window.sum()) != total[row, column] or window[-1] != final[row, column]:
                    raise ValueError("Reconstructed sequence differs from stored horizon targets")
                sequences[row, :h] = window
            elif total[row, column] != -1 or final[row, column] != -1:
                raise ValueError("Censored horizon must retain missing target sentinels")
    return sequences
