"""Full-record timestamp-bundle preparation; runs on cloud data dependencies."""

from .human_protocol import SOURCE_COLUMNS, BINARY_COLUMNS


def validate_source(logs) -> None:
    import polars as pl

    invalid = (
        logs.select(
            [
                ((~pl.col(field).is_in([0, 1])) | pl.col(field).is_null()).sum().alias(field)
                for field in sorted(BINARY_COLUMNS)
            ]
            + [
                pl.any_horizontal(pl.all().is_null()).any().alias("has_null"),
                (~pl.col("tab").is_between(0, 14)).any().alias("bad_tab"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )
    if any(invalid.values()):
        raise ValueError(f"Invalid source fields: {invalid}")


def prepare_tables(unique, assignment: dict[int, str]):
    """Use one split per timestamp bundle and preserve both timing proxies."""
    import polars as pl

    durations = (
        pl.col("play_time_ms").clip(lower_bound=0)
        + pl.col("profile_stay_time").clip(lower_bound=0)
        + pl.col("comment_stay_time").clip(lower_bound=0)
    )
    unique = unique.with_columns(durations.alias("feedback_duration_proxy"))
    bundles = unique.group_by(["user_id", "time_ms"]).agg(
        pl.len().alias("exposures"),
        pl.col("is_click").sum().alias("clicks"),
        (pl.col("time_ms").first() + pl.col("feedback_duration_proxy").max()).alias("available_ms"),
        (pl.col("time_ms").first() + pl.col("feedback_duration_proxy").sum()).alias(
            "available_sum_ms"
        ),
    )
    bundles = bundles.sort(["user_id", "time_ms"]).with_columns(
        pl.int_range(0, pl.len()).over("user_id").alias("bundle_position"),
        pl.len().over("user_id").alias("user_bundle_n"),
    )
    bundles = bundles.with_columns(
        pl.when(pl.col("bundle_position") < (pl.col("user_bundle_n") * 0.7).floor())
        .then(pl.lit("train"))
        .when(pl.col("bundle_position") < (pl.col("user_bundle_n") * 0.8).floor())
        .then(pl.lit("validation"))
        .when(pl.col("bundle_position") < (pl.col("user_bundle_n") * 0.9).floor())
        .then(pl.lit("calibration"))
        .otherwise(pl.lit("test"))
        .alias("chronological_split")
    )
    bundles = bundles.with_columns(pl.col("user_id").replace_strict(assignment).alias("user_split"))
    ordered = (
        unique.drop("feedback_duration_proxy")
        .join(
            bundles.select(
                [
                    "user_id",
                    "time_ms",
                    "bundle_position",
                    "user_bundle_n",
                    "chronological_split",
                    "user_split",
                ]
            ),
            on=["user_id", "time_ms"],
            how="left",
            validate="m:1",
        )
        .sort(
            ["user_id", "time_ms"] + [f for f in SOURCE_COLUMNS if f not in {"user_id", "time_ms"}]
        )
    )
    return ordered, bundles
