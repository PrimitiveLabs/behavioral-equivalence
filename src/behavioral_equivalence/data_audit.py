"""Independent DuckDB audit of a prepared human-data bucket.

No preparation transforms are imported here. All human records remain in cloud
storage; public reports contain aggregate counts and artifact hashes only.
"""

from pathlib import Path
import hashlib
import tempfile

COLUMNS = (
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
)


def audit_bucket(
    raw_paths: list[Path],
    event_path: Path,
    bundle_path: Path,
    excluded: list[int],
    assignment: dict[int, str],
    identities: list[dict],
) -> dict:
    import duckdb
    import numpy as np
    import pyarrow as pa

    with tempfile.TemporaryDirectory(prefix="behavioral-duckdb-") as spill:
        connection = duckdb.connect()
        connection.execute("SET threads=2")
        connection.execute("SET memory_limit='6GB'")
        connection.execute("SET temp_directory='" + spill + "'")
        connection.from_parquet([str(p) for p in raw_paths]).create_view("raw")
        connection.from_parquet(str(event_path)).create_view("events")
        connection.from_parquet(str(bundle_path)).create_view("bundles")
        connection.register(
            "cohort",
            pa.table(
                {
                    "user_id": pa.array(list(assignment), type=pa.int64()),
                    "assigned_split": list(assignment.values()),
                }
            ),
        )
        connection.register("excluded", pa.table({"user_id": pa.array(excluded, type=pa.int64())}))
        checks = {}

        def count(query):
            return int(connection.execute(query).fetchone()[0])

        checks["unknown_users"] = count(
            "SELECT count(*) FROM raw r LEFT JOIN cohort c USING(user_id) LEFT JOIN excluded x USING(user_id) WHERE c.user_id IS NULL AND x.user_id IS NULL"
        )
        invalid = " OR ".join(f'"{c}" IS NULL' for c in COLUMNS)
        binary = [
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "long_view",
            "is_profile_enter",
            "is_rand",
        ]
        invalid += " OR " + " OR ".join(f'"{c}" NOT IN (0,1)' for c in binary)
        invalid += " OR tab NOT BETWEEN 0 AND 14"
        checks["invalid_source_rows"] = count("SELECT count(*) FROM raw WHERE " + invalid)
        fields = ", ".join('"' + f + '"' for f in COLUMNS)
        connection.execute(
            "CREATE TEMP TABLE expected_events AS SELECT DISTINCT "
            + ", ".join('r."' + f + '"' for f in COLUMNS)
            + " FROM raw r JOIN cohort c USING(user_id)"
        )
        for left, right, name in [
            ("expected_events", "events", "source_rows_missing"),
            ("events", "expected_events", "extra_or_duplicate_prepared_rows"),
        ]:
            checks[name] = count(
                f"SELECT count(*) FROM (SELECT {fields} FROM {left} EXCEPT ALL SELECT {fields} FROM {right})"
            )
        connection.execute("""CREATE TEMP TABLE expected_bundles AS
            WITH grouped AS (
                SELECT user_id,time_ms,count(*) AS exposures,sum(is_click) AS clicks,
                    time_ms+max(greatest(play_time_ms,0)+greatest(profile_stay_time,0)+greatest(comment_stay_time,0)) AS available_ms,
                    time_ms+sum(greatest(play_time_ms,0)+greatest(profile_stay_time,0)+greatest(comment_stay_time,0)) AS available_sum_ms
                FROM expected_events GROUP BY user_id,time_ms
            ), ranked AS (
                SELECT *,row_number() OVER(PARTITION BY user_id ORDER BY time_ms)-1 AS bundle_position,
                    count(*) OVER(PARTITION BY user_id) AS user_bundle_n FROM grouped
            )
            SELECT r.*,CASE WHEN bundle_position<floor(user_bundle_n*CAST(.7 AS DOUBLE)) THEN 'train'
                WHEN bundle_position<floor(user_bundle_n*CAST(.8 AS DOUBLE)) THEN 'validation'
                WHEN bundle_position<floor(user_bundle_n*CAST(.9 AS DOUBLE)) THEN 'calibration'
                ELSE 'test' END AS chronological_split,c.assigned_split AS user_split
            FROM ranked r JOIN cohort c USING(user_id)""")
        bundle_fields = "user_id,time_ms,exposures,clicks,available_ms,available_sum_ms,bundle_position,user_bundle_n,chronological_split,user_split"
        for left, right, name in [
            ("expected_bundles", "bundles", "missing_or_incorrect_bundles"),
            ("bundles", "expected_bundles", "extra_or_duplicate_bundles"),
        ]:
            checks[name] = count(
                f"SELECT count(*) FROM (SELECT {bundle_fields} FROM {left} EXCEPT ALL SELECT {bundle_fields} FROM {right})"
            )
        checks["interaction_split_mismatch"] = (
            count("""SELECT count(*) FROM events e JOIN expected_bundles b USING(user_id,time_ms)
            WHERE e.bundle_position!=b.bundle_position OR e.user_bundle_n!=b.user_bundle_n
                OR e.chronological_split!=b.chronological_split OR e.user_split!=b.user_split""")
        )
        checks["timestamps_crossing_splits"] = count("""SELECT count(*) FROM
            (SELECT user_id,time_ms FROM events GROUP BY user_id,time_ms HAVING count(DISTINCT chronological_split)!=1)""")
        checks["users_crossing_user_splits"] = count("""SELECT count(*) FROM
            (SELECT user_id FROM events GROUP BY user_id HAVING count(DISTINCT user_split)!=1)""")
        checks["availability_proxy_order_violations"] = count(
            "SELECT count(*) FROM bundles WHERE available_sum_ms<available_ms OR available_ms<time_ms"
        )
        # Independently sort with DuckDB and stream hashes; no shared identity
        # implementation or complete per-user sequence is needed in memory.
        current_user = None
        hasher = None
        rows = 0
        actual = {}
        for batch in connection.execute(
            "SELECT user_id,CAST(time_ms AS BIGINT) AS time_ms,CAST(tab AS BIGINT) AS tab "
            "FROM raw ORDER BY user_id,time_ms,tab"
        ).fetch_record_batch(65536):
            u = batch.column(0).to_numpy()
            t = batch.column(1).to_numpy()
            a = batch.column(2).to_numpy()
            boundaries = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1, len(u)]
            for left, right in zip(boundaries, boundaries[1:]):
                uid = int(u[left])
                if uid != current_user:
                    if current_user is not None:
                        actual[current_user] = {"rows": rows, "sha256": hasher.hexdigest()}
                    current_user = uid
                    hasher = hashlib.sha256()
                    rows = 0
                hasher.update(
                    np.column_stack([t[left:right], a[left:right]]).astype("<i8").tobytes(order="C")
                )
                rows += int(right - left)
        if current_user is not None:
            actual[current_user] = {"rows": rows, "sha256": hasher.hexdigest()}
        expected = {r["user_id"]: {"rows": r["rows"], "sha256": r["sha256"]} for r in identities}
        checks["identity_multiset_mismatch"] = int(actual != expected)
        result = {
            "status": "complete" if not any(checks.values()) else "failed",
            "checks": checks,
            "raw_rows": count("SELECT count(*) FROM raw"),
            "raw_users": len(actual),
            "excluded_pilot_users": count(
                "SELECT count(DISTINCT r.user_id) FROM raw r JOIN excluded x USING(user_id)"
            ),
            "eligible_users": count("SELECT count(DISTINCT user_id) FROM events"),
            "rows_audited": count("SELECT count(*) FROM events"),
            "bundles_audited": count("SELECT count(*) FROM bundles"),
            "independent_backend": "DuckDB 1.3.2, independent SQL and streaming identity hashes",
            "timing_scope": "Checks the declared proxies and ordering; does not prove real-time feedback availability.",
        }
        connection.close()
    if result["status"] != "complete":
        raise AssertionError("Independent data audit failed: " + str(checks))
    return result
