"""Run in the pinned cloud data-audit image."""

import hashlib
import struct
import pytest
from behavioral_equivalence.data_audit import audit_bucket
from behavioral_equivalence.human_protocol import SOURCE_COLUMNS
from behavioral_equivalence.human_preparation import prepare_tables

pytest.importorskip("duckdb", reason="independent data integration runs in Modal")
pl = pytest.importorskip("polars")


def test_duckdb_audits_ties_dedup_exclusion_split_and_identity(tmp_path):
    records = []
    for user, n in [(0, 10), (1, 2), (32, 3)]:
        for time in range(n):
            row = {f: 0 for f in SOURCE_COLUMNS}
            row.update(
                user_id=user, time_ms=100 * time, video_id=time, is_click=time % 2, play_time_ms=5
            )
            records.append(row)
    records += [records[0], dict(records[6], video_id=99, play_time_ms=10, is_click=1)]
    raw = pl.DataFrame(records)
    raw.write_parquet(tmp_path / "raw.parquet")
    eligible = raw.filter(pl.col("user_id") != 32).unique(subset=list(SOURCE_COLUMNS))
    events, bundles = prepare_tables(eligible, {0: "train", 1: "test"})
    events.write_parquet(tmp_path / "events.parquet")
    bundles.write_parquet(tmp_path / "bundles.parquet")
    identity = []
    for uid in [0, 1, 32]:
        pairs = sorted((r["time_ms"], r["tab"]) for r in records if r["user_id"] == uid)
        payload = b"".join(struct.pack("<qq", *p) for p in pairs)
        identity.append(
            dict(user_id=uid, rows=len(pairs), sha256=hashlib.sha256(payload).hexdigest())
        )
    result = audit_bucket(
        [tmp_path / "raw.parquet"],
        tmp_path / "events.parquet",
        tmp_path / "bundles.parquet",
        [32],
        {0: "train", 1: "test"},
        identity,
    )
    assert result["rows_audited"] == 13 and result["bundles_audited"] == 12
    assert result["excluded_pilot_users"] == 1
    assert all(v == 0 for v in result["checks"].values())
    events.head(events.height - 1).write_parquet(tmp_path / "events.parquet")
    with pytest.raises(AssertionError, match="audit failed"):
        audit_bucket(
            [tmp_path / "raw.parquet"],
            tmp_path / "events.parquet",
            tmp_path / "bundles.parquet",
            [32],
            {0: "train", 1: "test"},
            identity,
        )
