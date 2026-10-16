"""Fixed per-user case sampling and availability-guarded history construction.

Polars is a pinned cloud dependency. This constructs private case artifacts;
it neither fits a model nor evaluates final predictive outcomes.
"""

import numpy as np
from .calendar_splits import EvaluationCalendar, ROLE_NAMES, calendar_roles
from .features import bundle_history_tokens


def prepare_sample(
    events, bundles, calendar: EvaluationCalendar, embargo_ms: int, caps: dict, seed: int
):
    import polars as pl

    if set(caps) != set(ROLE_NAMES) or any(
        not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in caps.values()
    ):
        raise ValueError("Positive per-user cap for every evaluation role required")
    bundles = bundles.sort(["user_id", "time_ms"]).with_row_index("feature_id")
    if bundles.select(pl.struct(["user_id", "time_ms"]).is_duplicated().any()).item():
        raise ValueError("Timestamp bundles must be unique")
    roles = calendar_roles(
        calendar,
        bundles["time_ms"].to_numpy(),
        bundles["available_sum_ms"].to_numpy(),
        bundles["user_split"].to_numpy(),
        "known_users",
        embargo_ms,
    )
    metadata = bundles.select(
        ["user_id", "time_ms", "available_sum_ms", "user_split", "feature_id"]
    ).with_columns(pl.Series("role_code", roles))
    eligible = events.select(["case_id", "user_id", "time_ms"]).join(
        metadata, on=["user_id", "time_ms"], how="left", validate="m:1"
    )
    if eligible["feature_id"].null_count() or events["case_id"].n_unique() != len(events):
        raise ValueError("Unique case IDs and complete bundle mapping required")
    eligible = (
        eligible.filter(pl.col("role_code") >= 0)
        .with_columns(pl.col("case_id").hash(seed=seed).alias("priority"))
        .sort(["user_id", "role_code", "priority", "case_id"])
        .with_columns(pl.col("case_id").cum_count().over(["user_id", "role_code"]).alias("rank"))
        .filter(
            pl.col("rank")
            <= pl.col("role_code").replace_strict({i: caps[r] for i, r in enumerate(ROLE_NAMES)})
        )
        .drop(["priority", "rank"])
        .join(
            events.select(["case_id", "tab", "video_id", "is_rand", "is_click"]),
            on="case_id",
            how="left",
            validate="1:1",
        )
        .sort("case_id")
    )
    # Selection uses fixed row IDs, timestamps, user roles and a seeded random
    # priority. Behavioral target values are joined only after selection.
    users = bundles["user_id"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    histories = np.empty((len(bundles), 32), dtype=np.uint8)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = bundles.slice(int(left), int(right - left))
        histories[left:right] = bundle_history_tokens(
            group["time_ms"].to_numpy(),
            group["available_sum_ms"].to_numpy(),
            group["clicks"].to_numpy(),
            group["exposures"].to_numpy(),
            extra_embargo_ms=embargo_ms,
        )
    selected_histories = histories[eligible["feature_id"].to_numpy()]
    return eligible, selected_histories
