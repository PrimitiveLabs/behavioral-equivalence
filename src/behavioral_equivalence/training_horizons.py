"""Outcome-blind within-role origins for fitting direct multihorizon predictors."""

import numpy as np
from .multistep_cohort import select_test_origins, observed_horizon_cohorts


def training_role_cohort(users, timestamps, roles, role, horizons, per_user_cap=64, seed=20261021):
    """Reuse the tested contiguous-window construction for training/validation.

    Role remapping is internal only. Excluded bundles remain excluded, and the
    caller must save actual source roles rather than the temporary sentinel.
    """
    r = np.asarray(roles)
    if (
        not isinstance(role, int)
        or isinstance(role, bool)
        or role not in {0, 1}
        or r.ndim != 1
        or r.dtype.kind not in "iu"
        or not np.isin(r, [-1, 0, 1, 2, 3]).all()
    ):
        raise ValueError("Only declared training or validation roles may supply fitting origins")
    mapped = np.where(r == role, 3, -1).astype(np.int8)
    origins = select_test_origins(users, timestamps, mapped, per_user_cap, seed)
    cohorts = observed_horizon_cohorts(users, timestamps, mapped, origins, horizons)
    for cohort in cohorts:
        cohort["scope"] = (
            "Conditional on H observed consecutive bundles within the declared fitting/validation role"
        )
    return origins, cohorts
