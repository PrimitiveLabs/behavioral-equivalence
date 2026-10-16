"""Independent DuckDB audit must hash integer timestamps without float conversion."""

import hashlib
import json
from pathlib import Path
import struct
import pytest
from behavioral_equivalence.data_audit import audit_bucket, COLUMNS

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")


@pytest.mark.parametrize("unsigned_time", [False, True])
def test_independent_audit_matches_exact_mixed_integer_timestamp_bytes(tmp_path, unsigned_time):
    rows, events, bundles = [], [], []
    for j in range(4):
        row = {key: 0 for key in COLUMNS}
        row.update(
            user_id=1, video_id=j, time_ms=2**53 + j, tab=j % 2, is_click=j % 2, date=20220428
        )
        rows.append(row)
        role = "train" if j < 2 else "validation" if j < 3 else "test"
        events.append(
            dict(
                row,
                bundle_position=j,
                user_bundle_n=4,
                chronological_split=role,
                user_split="train",
            )
        )
        bundles.append(
            dict(
                user_id=1,
                time_ms=row["time_ms"],
                exposures=1,
                clicks=row["is_click"],
                available_ms=row["time_ms"],
                available_sum_ms=row["time_ms"],
                bundle_position=j,
                user_bundle_n=4,
                chronological_split=role,
                user_split="train",
            )
        )
    raw = pa.Table.from_pylist(rows)
    index = raw.schema.get_field_index("time_ms")
    raw = raw.set_column(
        index,
        "time_ms",
        pa.array([r["time_ms"] for r in rows], type=pa.uint64() if unsigned_time else pa.int64()),
    )
    paths = [tmp_path / name for name in ["raw.parquet", "events.parquet", "bundles.parquet"]]
    for table, path in zip(
        [raw, pa.Table.from_pylist(events), pa.Table.from_pylist(bundles)], paths, strict=True
    ):
        pq.write_table(table, path)
    digest = hashlib.sha256(
        b"".join(struct.pack("<qq", row["time_ms"], row["tab"]) for row in rows)
    ).hexdigest()
    args = (
        [paths[0]],
        paths[1],
        paths[2],
        [],
        {1: "train"},
        [dict(user_id=1, rows=4, sha256=digest)],
    )
    result = audit_bucket(*args)
    assert result["status"] == "complete" and not any(result["checks"].values())
    fixture = json.loads(
        Path("data/synthetic/fixtures/release/audit-integer-identities-v1.json").read_text()
    )
    assert hashlib.sha256(fixture["source_module"].encode()).hexdigest() == fixture["source_sha256"]
    scope = {}
    exec(compile(fixture["source_module"], "frozen-independent-data-audit", "exec"), scope)
    if unsigned_time:
        with pytest.raises(AssertionError, match="identity_multiset_mismatch"):
            scope["audit_bucket"](*args)
    else:
        assert scope["audit_bucket"](*args) == result
