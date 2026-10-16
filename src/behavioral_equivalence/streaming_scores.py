"""Bucketwise predictive metrics and shared user-cluster bootstrap comparisons.

Only user-level sufficient statistics need remain in memory across source
buckets. Private user identifiers and losses must not enter public reports.
"""

import numpy as np


def metric_block(case_ids, users, targets, probabilities):
    """Summarize one complete user-disjoint source bucket for one model/cohort.

    A matrix supplies equally replicated forecasts for each original case.
    Proper losses are averaged across replicas before averaging within users;
    averaging probabilities first would estimate a different predictor.
    """
    ids, u, y, p = map(np.asarray, (case_ids, users, targets, probabilities))
    if (
        ids.ndim != 1
        or any(a.shape != ids.shape for a in (u, y))
        or p.ndim not in (1, 2)
        or p.shape[0] != len(ids)
        or (p.ndim == 2 and p.shape[1] < 1)
        or any(a.dtype.kind not in "iu" or np.any(a < 0) for a in (ids, u))
        or len(np.unique(ids)) != len(ids)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p > 1)
    ):
        raise ValueError(
            "Unique immutable cases, integer users and aligned binary forecasts required"
        )
    unique, inverse = np.unique(u, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique))
    replicates = p.shape[1] if p.ndim == 2 else 1
    matrix = p if p.ndim == 2 else p[:, None]
    q = np.clip(matrix.astype(float), 1e-6, 1 - 1e-6)
    nll = -(y[:, None] * np.log(q) + (1 - y[:, None]) * np.log1p(-q)).mean(axis=1)
    brier = ((matrix - y[:, None]) ** 2).mean(axis=1)
    nll_sum = np.bincount(inverse, weights=nll, minlength=len(unique))
    brier_sum = np.bincount(inverse, weights=brier, minlength=len(unique))
    p = matrix.reshape(-1)
    calibration_users = np.repeat(u, replicates)
    calibration_targets = np.repeat(y, replicates)
    bins = np.minimum((p * 20).astype(int), 19)
    weights = np.repeat(1.0 / counts[inverse], replicates) / replicates
    calibration = []
    for index in range(20):
        chosen = bins == index
        calibration.append(
            dict(
                cases=int(chosen.sum()),
                users=int(len(np.unique(calibration_users[chosen]))),
                user_mass=float(weights[chosen].sum()),
                predicted_sum=float(weights[chosen] @ p[chosen]),
                observed_sum=float(weights[chosen] @ calibration_targets[chosen]),
            )
        )
    return dict(
        user_ids=unique,
        forecast_replicates=replicates,
        case_counts=counts,
        user_nll=nll_sum / counts,
        user_brier=brier_sum / counts,
        case_nll_sum=float(nll.sum()),
        case_brier_sum=float(brier.sum()),
        calibration=calibration,
    )


def merge_metric_blocks(blocks):
    """Merge complete buckets, rejecting a user split across source blocks.

    This contract is essential: normalizing a user's separate partial batches
    would give that user multiple votes. Batch neural inference within a bucket
    first, then summarize its complete cohort with metric_block.
    """
    replicate_counts = {block.get("forecast_replicates", 1) for block in blocks}
    if len(replicate_counts) > 1:
        raise ValueError("All buckets must use the same forecast replication rule")
    replicates = next(iter(replicate_counts), 1)
    keys = ("user_ids", "case_counts", "user_nll", "user_brier")
    arrays = {
        key: np.concatenate([b[key] for b in blocks])
        if blocks
        else np.empty(0, dtype=np.int64 if key in keys[:2] else float)
        for key in keys
    }
    u = arrays["user_ids"]
    if len(np.unique(u)) != len(u):
        raise ValueError("A user occurs in multiple supposedly user-disjoint buckets")
    order = np.argsort(u)
    arrays = {key: value[order] for key, value in arrays.items()}
    users = len(u)
    cases = int(arrays["case_counts"].sum())
    bins = []
    for index in range(20):
        cells = [block["calibration"][index] for block in blocks]
        mass = sum(cell["user_mass"] for cell in cells)
        bins.append(
            dict(
                lower=index / 20,
                upper=(index + 1) / 20,
                cases=sum(cell["cases"] for cell in cells),
                users=sum(cell["users"] for cell in cells),
                equal_user_weight=mass / users if users else 0.0,
                predicted=sum(cell["predicted_sum"] for cell in cells) / mass if mass else None,
                observed=sum(cell["observed_sum"] for cell in cells) / mass if mass else None,
            )
        )
    metrics = dict(
        cases=cases,
        users=users,
        nll_case_mean=sum(b["case_nll_sum"] for b in blocks) / cases if cases else None,
        brier_case_mean=sum(b["case_brier_sum"] for b in blocks) / cases if cases else None,
        nll_equal_user_mean=float(arrays["user_nll"].mean()) if users else None,
        brier_equal_user_mean=float(arrays["user_brier"].mean()) if users else None,
        calibration_bins=bins,
        probability_clip=1e-6,
        scope="Fixed-case point metrics; calendar dependence and model uncertainty are separate",
    )
    if replicates > 1:
        metrics.update(
            forecast_replicates_per_case=replicates,
            prediction_copies=cases * replicates,
            calibration_unit="Forecast copies, equal mass within each original case and user; not independent samples",
        )
    return dict(private_user_statistics=arrays, metrics=metrics)


def shared_user_bootstrap(
    user_ids,
    case_counts,
    model_losses,
    comparisons,
    *,
    replications=2000,
    seed=20261015,
    batch_size=32,
    margin=0.01,
):
    """Pair all contrasts on the same resampled original user clusters.

    Multinomial counts implement the same resampling law as explicit indices.
    Draw indices, then multiply their counts by user-level contrasts, avoiding
    a replications-by-users-by-models tensor. Intervals are pointwise percentile
    intervals, not multiplicity-adjusted secondary discovery guarantees.
    """
    users, counts = np.asarray(user_ids), np.asarray(case_counts)
    names = list(model_losses)
    if (
        users.ndim != 1
        or counts.shape != users.shape
        or len(users) < 2
        or users.dtype.kind not in "iu"
        or np.any(users < 0)
        or len(np.unique(users)) != len(users)
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
        or not names
        or not comparisons
        or any(a not in names or b not in names for a, b in comparisons.values())
        or not isinstance(replications, int)
        or replications < 100
        or not isinstance(batch_size, int)
        or batch_size < 1
        or not np.isfinite(margin)
        or margin <= 0
    ):
        raise ValueError(
            "Aligned original user clusters, declared contrasts and bootstrap settings required"
        )
    losses = {name: np.asarray(value) for name, value in model_losses.items()}
    if any(value.shape != users.shape or not np.isfinite(value).all() for value in losses.values()):
        raise ValueError("One finite mean loss per original user and fitted model required")
    labels = list(comparisons)
    differences = np.column_stack([losses[c] - losses[r] for c, r in comparisons.values()])
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((replications, len(labels)))
    for left in range(0, replications, batch_size):
        size = min(batch_size, replications - left)
        sampled = rng.integers(0, len(users), (size, len(users)))
        multiplicities = np.vstack([np.bincount(row, minlength=len(users)) for row in sampled])
        bootstrap[left : left + size] = (multiplicities @ differences) / len(users)
    bounds = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {
        label: dict(
            candidate=comparisons[label][0],
            reference=comparisons[label][1],
            candidate_minus_reference_nats=float(differences[:, index].mean()),
            ci95=[float(bounds[0, index]), float(bounds[1, index])],
            users=len(users),
            cases=int(counts.sum()),
            margin_nats=margin,
            noninferior_at_margin=bool(bounds[1, index] < margin),
            improved=bool(bounds[1, index] < 0),
            replications=replications,
            seed=seed,
            scope="Shared user-cluster percentile resamples conditional on fitted models; pointwise intervals, no independent-fold, calendar or retraining guarantee",
        )
        for index, label in enumerate(labels)
    }
