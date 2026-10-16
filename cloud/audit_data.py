"""Independent audits of prepared bundles and both partition protocols."""

import modal

app = modal.App("behavioral-equivalence-data-audit-v2")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("polars==1.31.0")


@app.function(
    image=image, cpu=2, memory=8192, timeout=600, max_containers=1, volumes={"/artifacts": volume}
)
def audit() -> dict:
    from pathlib import Path
    import hashlib
    import json
    import time
    import polars as pl

    start = time.monotonic()
    root = Path("/artifacts/datasets/kuairand-1k/prepared-v4")
    events = pl.read_parquet(root / "interactions.parquet")
    bundles = pl.read_parquet(root / "timestamp-bundles.parquet")
    expected = events.group_by(["user_id", "time_ms"]).agg(
        pl.len().alias("exposures"),
        pl.col("is_click").sum().alias("clicks"),
        pl.col("outcome_available_ms").max().alias("available_ms"),
    )
    joined = expected.join(bundles, on=["user_id", "time_ms"], suffix="_stored", validate="1:1")
    for field in ["exposures", "clicks", "available_ms"]:
        if joined.filter(pl.col(field) != pl.col(field + "_stored")).height:
            raise AssertionError("Bundle aggregate mismatch: " + field)
    if joined.height != bundles.height or bundles["exposures"].sum() != events.height:
        raise AssertionError("Missing bundle observations")
    if events.filter(pl.col("outcome_available_ms") < pl.col("time_ms")).height:
        raise AssertionError("Invalid availability")
    if (
        events.group_by(["user_id", "time_ms"])
        .agg(pl.col("chronological_split").n_unique().alias("n"))
        .filter(pl.col("n") != 1)
        .height
    ):
        raise AssertionError("Same timestamp crosses chronological split")
    users = events["user_id"].unique().to_list()
    user_order = sorted(users, key=lambda u: hashlib.sha256(f"20261015:{u}".encode()).digest())
    ranks = {u: i for i, u in enumerate(user_order)}
    names = ["train", "validation", "calibration", "test"]
    inspected = 0
    for user in users:
        sequence = bundles.filter(pl.col("user_id") == user).sort("time_ms")
        n = sequence.height
        # Independent scalar boundaries, not the pipeline's Polars expression.
        expected_split = [
            names[0]
            if i < int(0.7 * n)
            else names[1]
            if i < int(0.8 * n)
            else names[2]
            if i < int(0.9 * n)
            else names[3]
            for i in range(n)
        ]
        if sequence["chronological_split"].to_list() != expected_split:
            raise AssertionError("Chronological boundary mismatch")
        group = events.filter(pl.col("user_id") == user)["user_split"].unique().to_list()
        i = ranks[user]
        n_users = len(users)
        expected_user = (
            "train"
            if i < int(0.7 * n_users)
            else "validation"
            if i < int(0.8 * n_users)
            else "calibration"
            if i < int(0.9 * n_users)
            else "test"
        )
        if group != [expected_user]:
            raise AssertionError("User split overlap or wrong assignment")
        inspected += sequence.height
    delay = events["outcome_available_ms"] - events["time_ms"]
    report = {
        "status": "complete",
        "source": "prepared-v4",
        "rows_audited": events.height,
        "bundles_audited": inspected,
        "users_audited": len(users),
        "checks": [
            "aggregate reconciliation",
            "no timestamp crosses chronological split",
            "independent per-user chronological boundaries",
            "user-disjoint assignment",
            "availability never precedes observation",
        ],
        "availability_delay_ms": {
            "median": delay.median(),
            "p99": delay.quantile(0.99),
            "max": delay.max(),
        },
        "not_established": [
            "true feedback availability semantics",
            "full-record deduplication",
            "randomized assignment identification",
            "27K pilot isolation",
        ],
        "wall_seconds": time.monotonic() - start,
    }
    (root / "independent-audit.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report
