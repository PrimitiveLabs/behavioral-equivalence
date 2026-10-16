"""Cloud-only data integration tests, with an explicit local dependency skip."""

import pytest

from behavioral_equivalence.human_preparation import prepare_tables, validate_source
from behavioral_equivalence.human_protocol import SOURCE_COLUMNS

pl = pytest.importorskip("polars", reason="data integration runs in the pinned Modal data image")


def source_record(user, time, video, click, duration=5):
    row = {field: 0 for field in SOURCE_COLUMNS}
    row.update(user_id=user, time_ms=time, video_id=video, is_click=click, play_time_ms=duration)
    return row


def test_full_record_dedup_tied_bundles_and_split_boundaries():
    records = [source_record(0, t * 100, t, t % 2) for t in range(10)]
    records += [source_record(0, 600, 99, 1, 10), records[0], source_record(1, 100, 100, 0)]
    frame = pl.DataFrame(records)
    validate_source(frame.lazy())
    unique = frame.unique(subset=list(SOURCE_COLUMNS))
    ordered, bundles = prepare_tables(unique, {0: "train", 1: "test"})
    assert ordered.height == 12
    assert bundles.height == 11
    tie = bundles.filter((pl.col("user_id") == 0) & (pl.col("time_ms") == 600)).to_dicts()[0]
    assert tie["exposures"] == 2 and tie["clicks"] == 1
    assert tie["available_ms"] == 610 and tie["available_sum_ms"] == 615
    assert ordered.filter((pl.col("user_id") == 0) & (pl.col("time_ms") == 600))[
        "chronological_split"
    ].unique().to_list() == ["train"]
    user = bundles.filter(pl.col("user_id") == 0)
    assert user["chronological_split"].to_list() == ["train"] * 7 + [
        "validation",
        "calibration",
        "test",
    ]
    assert set(ordered.filter(pl.col("user_id") == 1)["user_split"].to_list()) == {"test"}
    assert set(SOURCE_COLUMNS).issubset(ordered.columns)


def test_source_boundary_rejects_missing_targets_invalid_binary_and_unknown_scenario():
    for field, value in [("is_click", 2), ("is_like", None), ("tab", 15)]:
        row = source_record(0, 100, 1, 0)
        row[field] = value
        with pytest.raises(ValueError, match="Invalid source"):
            validate_source(pl.DataFrame([row]).lazy())
