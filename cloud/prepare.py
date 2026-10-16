"""Prepare real user trajectories on Modal; retain only predeclared fields."""

import modal

app = modal.App("behavioral-equivalence-prepare-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "polars==1.31.0", "pyarrow==20.0.0"
)


@app.function(
    image=image, cpu=2, memory=8192, timeout=1200, max_containers=1, volumes={"/artifacts": volume}
)
def prepare_kuairand() -> dict:
    import hashlib
    import json
    from pathlib import Path
    import tarfile
    import time
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.parquet as pq
    import polars as pl

    start = time.monotonic()
    root = Path("/artifacts/datasets/kuairand-1k")
    destination = root / "prepared-v4"
    destination.mkdir(exist_ok=False)
    archive = root / "source.tar.gz"
    source_columns = [
        "user_id",
        "video_id",
        "date",
        "hourmin",
        "time_ms",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time_ms",
        "duration_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
        "is_rand",
        "tab",
    ]
    binary_columns = {
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "is_rand",
    }
    schema = pa.schema(
        [(name, pa.int8() if name in binary_columns else pa.int64()) for name in source_columns]
    )
    fields = schema.names
    shards = []
    with tarfile.open(archive, "r:gz") as source:
        license_bytes = source.extractfile("KuaiRand-1K/LICENSE").read()
        (destination / "source-LICENSE.txt").write_bytes(license_bytes)
        embedded_license = license_bytes.decode("utf-8-sig")
        if "Attribution-ShareAlike 4.0 International" not in embedded_license:
            raise ValueError("Embedded archive license needs review before processing")
        for member in source.getmembers():
            if member.isfile() and "/log_" in member.name and member.name.endswith(".csv"):
                shard = destination / (Path(member.name).stem + ".parquet")
                with source.extractfile(member) as raw, pq.ParquetWriter(shard, schema) as writer:
                    reader = csv.open_csv(
                        raw,
                        convert_options=csv.ConvertOptions(
                            include_columns=fields, column_types={f.name: f.type for f in schema}
                        ),
                    )
                    for batch in reader:
                        table = pa.Table.from_batches([batch]).cast(schema)
                        writer.write_table(table)
                shards.append(str(shard))
    logs = pl.scan_parquet(shards)
    raw_count = logs.select(pl.len()).collect().item()
    if logs.filter(~pl.col("is_click").is_in([0, 1])).select(pl.len()).collect().item():
        raise ValueError("Nonbinary click target")
    # Detect disagreements before deduplicating repeated source records.
    keys = ["user_id", "video_id", "time_ms"]
    conflicts = (
        logs.group_by(keys)
        .agg(pl.col("is_click").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .select(pl.len())
        .collect()
        .item()
    )
    # These keys are candidate identifiers, not documented unique event IDs.
    # Deduplicate only complete source records. Preserve differing feedback and
    # retain shared timestamps as unordered observation bundles.
    unique = logs.unique(subset=fields).collect()
    duplicate_count = raw_count - unique.height
    # Preserve tied observations as one unordered timestamp bundle. Never let
    # an outcome from the current bundle enter its own prediction history.
    unique = unique.with_columns(
        pl.len().over(["user_id", "time_ms"]).alias("bundle_size"),
        (
            pl.col("time_ms")
            + pl.col("play_time_ms").clip(lower_bound=0)
            + pl.col("profile_stay_time").clip(lower_bound=0)
            + pl.col("comment_stay_time").clip(lower_bound=0)
        ).alias("outcome_available_ms"),
    )
    tie_rows = unique.filter(pl.col("bundle_size") > 1).height
    bundles = unique.group_by(["user_id", "time_ms"]).agg(
        pl.len().alias("exposures"),
        pl.col("is_click").sum().alias("clicks"),
        pl.col("outcome_available_ms").max().alias("available_ms"),
    )
    bundles = (
        bundles.sort(["user_id", "time_ms"])
        .with_columns(
            pl.int_range(0, pl.len()).over("user_id").alias("bundle_position"),
            pl.len().over("user_id").alias("user_bundle_n"),
        )
        .with_columns(
            pl.when(pl.col("bundle_position") < (pl.col("user_bundle_n") * 0.7).floor())
            .then(pl.lit("train"))
            .when(pl.col("bundle_position") < (pl.col("user_bundle_n") * 0.8).floor())
            .then(pl.lit("validation"))
            .when(pl.col("bundle_position") < (pl.col("user_bundle_n") * 0.9).floor())
            .then(pl.lit("calibration"))
            .otherwise(pl.lit("test"))
            .alias("chronological_split")
        )
    )
    ordered = unique.join(
        bundles.select(
            ["user_id", "time_ms", "bundle_position", "user_bundle_n", "chronological_split"]
        ),
        on=["user_id", "time_ms"],
        how="left",
        validate="m:1",
    ).sort(["user_id", "time_ms"] + [f for f in fields if f not in ("user_id", "time_ms")])
    bundles.write_parquet(destination / "timestamp-bundles.parquet")
    users = ordered["user_id"].unique().to_list()
    ranked = sorted(users, key=lambda u: hashlib.sha256(f"20261015:{u}".encode()).digest())
    n = len(ranked)
    assignment = {
        u: "train"
        if i < int(0.7 * n)
        else "validation"
        if i < int(0.8 * n)
        else "calibration"
        if i < int(0.9 * n)
        else "test"
        for i, u in enumerate(ranked)
    }
    ordered = ordered.with_columns(pl.col("user_id").replace_strict(assignment).alias("user_split"))
    output = destination / "interactions.parquet"
    ordered.write_parquet(output)
    # Never include full-period video popularity or user statistics as predictors.
    # Current-click labels and realized watch time are also unavailable as context.
    report = {
        "status": "complete",
        "scope": "real-data pilot preprocessing only",
        "rows_raw": raw_count,
        "rows_kept": ordered.height,
        "users": n,
        "duplicate_rows_removed": duplicate_count,
        "duplicate_policy": "identical complete source records across all 19 official columns",
        "retained_source_columns": fields,
        "ambiguous_candidate_key_label_groups": conflicts,
        "timestamp_tie_rows_excluded": 0,
        "rows_in_tied_timestamp_bundles": tie_rows,
        "timestamp_bundles": bundles.height,
        "history_policy": "strictly earlier timestamp bundles whose outcomes are available before prediction",
        "outcome_availability_assumption": "timestamp plus sum of nonnegative play/profile/comment durations is a conservative observation time; validate against log semantics",
        "random_exposures_kept": ordered.filter(pl.col("is_rand") == 1).height,
        "chronological_counts": ordered.group_by("chronological_split").len().to_dicts(),
        "user_counts": ordered.group_by("user_split").len().to_dicts(),
        "input_archive": "dfaafbb5fd16e9e6d2f9a6adaa4ea25df20a14bc26a90961c136e26c00a7bb2c",
        "embedded_license": "CC-BY-SA-4.0",
        "embedded_license_sha256": hashlib.sha256(license_bytes).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "feature_policy": "history of observed outcomes plus pre-exposure fields only",
        "prohibited_predictors": [
            "current is_click",
            "current play_time_ms",
            "full-period video_features_statistic",
            "full-period user aggregates",
        ],
        "min_time_ms": ordered["time_ms"].min(),
        "max_time_ms": ordered["time_ms"].max(),
        "wall_seconds": time.monotonic() - start,
        "remaining_audits": [
            "feature timestamp provenance",
            "pipeline versus independent split audit",
            "random assignment support and probabilities",
            "pilot/final user-ID mapping",
        ],
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report
