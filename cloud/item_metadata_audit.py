"""Audit intrinsic item context and random-pool support, without response scores."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-item-metadata-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pyarrow==20.0.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_item_metadata.py", remote_path="/research/tests/test_item_metadata.py"
    )
)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(6144, 6144),
    timeout=1800,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run() -> dict:
    import hashlib
    import json
    import subprocess
    import tarfile
    import time
    import numpy as np
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from behavioral_equivalence.item_metadata import duration_groups

    start = time.monotonic()
    data = Path("/artifacts/datasets/kuairand-27k")
    prepared = data / "prepared-candidate-v2"
    source = {
        "item_metadata_audit.py": Path(__file__).read_text(),
        "item_metadata.py": Path(
            "/research/src/behavioral_equivalence/item_metadata.py"
        ).read_text(),
        "test_item_metadata.py": Path("/research/tests/test_item_metadata.py").read_text(),
    }
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    out = data / ("item-metadata-v1-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(source, indent=2))
    test = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_item_metadata.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (out / "smoke.json").write_text(
        json.dumps(
            {"returncode": test.returncode, "stdout": test.stdout, "stderr": test.stderr}, indent=2
        )
    )
    if test.returncode:
        raise RuntimeError("Duration-group fixture test failed")
    expected_archive = "b8a8a5b777e564202cff6b96676959ee0d6baccc7ad7376bcb4af6767362d41b"
    if sha256(data / "source.tar.gz") != expected_archive:
        raise ValueError("Official archive differs from verified download")
    audit = json.loads((prepared / "independent-audit.json").read_text())
    if audit["status"] != "complete":
        raise ValueError("Complete source-record audit required")
    # Public source reports 32,038,725 reindexed videos. Reject unexpected IDs
    # rather than truncating metadata or inventing mappings.
    n = 32038725
    seen = np.zeros(n, dtype=bool)
    duration = np.full(n, np.nan)
    upload_day = np.full(n, np.iinfo(np.int32).min, dtype=np.int32)
    member_name = "KuaiRand-27K/data/video_features_basic_27k.csv"
    metadata_rows = 0
    found = False
    with tarfile.open(data / "source.tar.gz", "r|gz") as archive:
        for member in archive:
            if member.name != member_name:
                continue
            found = True
            with archive.extractfile(member) as stream:
                reader = csv.open_csv(
                    stream,
                    read_options=csv.ReadOptions(block_size=16 * 1024**2),
                    convert_options=csv.ConvertOptions(
                        include_columns=["video_id", "video_duration", "upload_dt"],
                        column_types={
                            "video_id": pa.int64(),
                            "video_duration": pa.float64(),
                            "upload_dt": pa.string(),
                        },
                        null_values=csv.ConvertOptions().null_values
                        + ["UNKNOWN", "unknown", "None"],
                    ),
                )
                for batch in reader:
                    ids = batch.column("video_id").to_numpy()
                    if (
                        np.any(ids < 0)
                        or np.any(ids >= n)
                        or len(np.unique(ids)) != len(ids)
                        or seen[ids].any()
                    ):
                        raise ValueError(
                            "Metadata item IDs duplicate or exceed documented reindexed space"
                        )
                    seen[ids] = True
                    duration[ids] = batch.column("video_duration").to_numpy(zero_copy_only=False)
                    dates = pc.cast(
                        pc.strptime(
                            batch.column("upload_dt"),
                            format="%Y-%m-%d",
                            unit="s",
                            error_is_null=True,
                        ),
                        pa.int64(),
                    ).to_numpy(zero_copy_only=False)
                    valid = np.isfinite(dates)
                    upload_day[ids[valid]] = (dates[valid] // 86400).astype(np.int32)
                    metadata_rows += len(ids)
            break
    if not found:
        raise ValueError("Basic item metadata member is missing from official archive")
    np.save(out / "duration-ms.npy", duration, allow_pickle=False)
    np.save(out / "upload-day.npy", upload_day, allow_pickle=False)
    np.save(out / "metadata-present.npy", seen, allow_pickle=False)
    logs = []
    random_name = "log_random_4_22_to_5_08_27k.parquet"
    for bucket in audit["bucket_reports"]:
        path = prepared / "raw" / f"bucket-{bucket['bucket']:02d}" / random_name
        if sha256(path) != bucket["input_sha256"][random_name]:
            raise ValueError("Random-exposure source differs from independently audited input")
        table = pq.read_table(
            path, columns=["user_id", "video_id", "time_ms", "tab", "is_rand", "duration_ms"]
        )
        logs.append(
            {
                name: table[name].combine_chunks().to_numpy(zero_copy_only=False)
                for name in table.column_names
            }
        )
    random = {key: np.concatenate([part[key] for part in logs]) for key in logs[0]}
    if np.any(random["is_rand"] != 1):
        raise ValueError("Random-source flag is inconsistent")
    ids = random["video_id"]
    if np.any(ids < 0) or np.any(ids >= n):
        raise ValueError("Random item IDs outside metadata space")
    pool = np.unique(ids)
    if len(pool) != 7583:
        raise ValueError("Observed full random pool differs from documented 7583 items")
    np.savez_compressed(
        out / "random-pool.npz",
        video_id=pool,
        duration_ms=duration[pool],
        upload_day=upload_day[pool],
        duration_group=duration_groups(duration[pool]),
    )
    cohort = json.loads((prepared / "cohort-private.json").read_text())
    eligible = ~np.isin(random["user_id"], cohort["excluded_user_ids"])
    metadata_value = duration[ids]
    comparable = np.isfinite(metadata_value) & (metadata_value > 0) & (random["duration_ms"] > 0)
    difference = np.abs(metadata_value[comparable] - random["duration_ms"][comparable])
    groups = duration_groups(metadata_value)
    log_groups = duration_groups(random["duration_ms"])
    valid_upload = upload_day[ids] != np.iinfo(np.int32).min
    exposure_day = random["time_ms"] // 86400000
    report = {
        "status": "complete",
        "phase": "intrinsic item metadata and outcome-free random-source support audit",
        "source_digest": digest,
        "archive_sha256": expected_archive,
        "metadata_member": member_name,
        "metadata_rows": metadata_rows,
        "metadata_missing_ids": int((~seen).sum()),
        "metadata_invalid_duration_ids": int((duration_groups(duration) < 0).sum()),
        "metadata_missing_upload_ids": int((upload_day == np.iinfo(np.int32).min).sum()),
        "random_source_rows": len(ids),
        "observed_random_pool_items": len(pool),
        "random_source_eligible_nonpilot_rows": int(eligible.sum()),
        "pool_duration_group_item_counts": {
            str(g): int((duration_groups(duration[pool]) == g).sum()) for g in [-1, 0, 1, 2, 3]
        },
        "random_rows_missing_metadata": int((~seen[ids]).sum()),
        "random_rows_comparable_duration": int(comparable.sum()),
        "random_rows_duration_exact_mismatch": int((difference > 0).sum()),
        "random_rows_duration_over_1000ms_mismatch": int((difference > 1000).sum()),
        "random_rows_duration_group_mismatch": int(((groups != log_groups) & comparable).sum()),
        "random_rows_upload_after_utc_exposure_day": int(
            (valid_upload & (upload_day[ids] > exposure_day)).sum()
        ),
        "random_rows_upload_more_than_one_day_after_exposure": int(
            (valid_upload & (upload_day[ids] > exposure_day + 1)).sum()
        ),
        "output_sha256": {
            name: sha256(out / name)
            for name in [
                "duration-ms.npy",
                "upload-day.npy",
                "metadata-present.npy",
                "random-pool.npz",
            ]
        },
        "wall_seconds": time.monotonic() - start,
        "limits": [
            "Random-source support counts precede full-record deduplication and are not independent behavioral observations.",
            "Random-pool duration agreement does not establish agreement for every standard recommendation.",
            "Intrinsic item-feature timing remains a declared assumption; metadata snapshots do not provide complete provenance.",
            "No response scores, treatment-effect or arbitrary-interface transfer estimate is computed.",
            "Released users are selected by full-period random-exposure count; source insertion probability is withheld.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {**report, "report": str(out / "report.json")}


@app.local_entrypoint()
def main():
    import json

    call = run.spawn()
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": run.object_id}), flush=True
    )
    print(json.dumps(call.get(), indent=2), flush=True)
