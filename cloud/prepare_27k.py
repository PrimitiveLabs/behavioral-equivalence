"""Cloud-only full-record conversion, pilot exclusion and candidate data splits.

This pipeline creates data artifacts, not confirmatory outcome measurements.
"""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-kuairand27k-preparation-v2")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "polars==1.31.0", "pyarrow==20.0.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_human_preparation.py",
        remote_path="/research/tests/test_human_preparation.py",
    )
)
BUCKETS = 32


def paths():
    return Path("/artifacts/datasets/kuairand-27k"), Path(
        "/artifacts/datasets/kuairand-27k/prepared-candidate-v2"
    )


def source_snapshot() -> dict:
    import hashlib
    import json

    files = {
        "prepare_27k.py": Path(__file__).read_text(),
        "human_protocol.py": Path(
            "/research/src/behavioral_equivalence/human_protocol.py"
        ).read_text(),
    }
    files["human_preparation.py"] = Path(
        "/research/src/behavioral_equivalence/human_preparation.py"
    ).read_text()
    files["test_human_preparation.py"] = Path(
        "/research/tests/test_human_preparation.py"
    ).read_text()
    return {
        "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "files": files,
    }


def frozen_source() -> dict:
    import json

    _, out = paths()
    saved = json.loads((out / "source.json").read_text())
    if saved != source_snapshot():
        raise ValueError("Candidate preparation source changed after conversion started")
    return saved


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(8192, 8192),
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def convert() -> dict:
    import json
    import tarfile
    import time
    import numpy as np
    import pyarrow as pa
    import pyarrow.csv as csv
    import pyarrow.parquet as pq
    from behavioral_equivalence.human_protocol import (
        SOURCE_COLUMNS,
        BINARY_COLUMNS,
        validate_27k_logs,
    )

    root, out = paths()
    inspection = json.loads((root / "inspection.json").read_text())
    if inspection["status"] != "complete" or inspection.get(
        "applicable_license", inspection["embedded_license"]
    ) not in {
        "CC-BY-SA-4.0",
        "CC-BY-4.0",
    }:
        raise ValueError("Completed rights/schema inspection required")
    expected = set(validate_27k_logs(inspection["entries"]))
    volume.reload()
    if (out / "conversion.json").exists():
        frozen_source()
        return json.loads((out / "conversion.json").read_text())
    out.mkdir(exist_ok=False)
    (out / "source.json").write_text(json.dumps(source_snapshot(), indent=2))
    (out / "started.json").write_text(
        json.dumps(
            {
                "function_call_id": modal.current_function_call_id(),
                "phase": "raw conversion",
                "outcome_performance": "not inspected",
            }
        )
    )
    volume.commit()
    schema = pa.schema(
        [(name, pa.int8() if name in BINARY_COLUMNS else pa.int64()) for name in SOURCE_COLUMNS]
    )
    start = time.monotonic()
    records = []
    with tarfile.open(root / "source.tar.gz", "r|gz") as archive:
        for member in archive:
            if member.name not in expected:
                continue
            writers = []
            try:
                for b in range(BUCKETS):
                    folder = out / "raw" / f"bucket-{b:02d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    writers.append(
                        pq.ParquetWriter(
                            folder / f"{Path(member.name).stem}.parquet", schema, compression="zstd"
                        )
                    )
                counts = [0] * BUCKETS
                with archive.extractfile(member) as raw:
                    reader = csv.open_csv(
                        raw,
                        read_options=csv.ReadOptions(block_size=16 * 1024**2),
                        convert_options=csv.ConvertOptions(
                            include_columns=list(SOURCE_COLUMNS),
                            column_types={f.name: f.type for f in schema},
                        ),
                    )
                    for batch in reader:
                        table = pa.Table.from_batches([batch]).cast(schema)
                        bucket = table["user_id"].combine_chunks().to_numpy() % BUCKETS
                        for b in range(BUCKETS):
                            indices = np.flatnonzero(bucket == b)
                            if len(indices):
                                subset = table.take(pa.array(indices))
                                writers[b].write_table(subset)
                                counts[b] += len(indices)
            finally:
                for writer in writers:
                    writer.close()
            records.append({"source": member.name, "rows": sum(counts), "bucket_rows": counts})
            volume.commit()
            print(f"Converted {member.name}: {sum(counts)} rows", flush=True)
    if {r["source"] for r in records} != expected:
        raise ValueError("Missing source logs after conversion")
    result = {
        "status": "complete",
        "phase": "raw conversion only",
        "source_archive_sha256": inspection["archive_sha256"],
        "source_digest": source_snapshot()["sha256"],
        "buckets": BUCKETS,
        "logs": records,
        "raw_rows": sum(r["rows"] for r in records),
        "wall_seconds": time.monotonic() - start,
        "outcome_performance": "not inspected",
    }
    (out / "conversion.json").write_text(json.dumps(result, indent=2))
    (out / "source-LICENSE.txt").write_bytes((root / "source-LICENSE.txt").read_bytes())
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=900,
    retries=0,
    max_containers=4,
    volumes={"/artifacts": volume},
)
def fingerprint_bucket(bucket: int) -> dict:
    import json
    import polars as pl
    from behavioral_equivalence.human_protocol import identity_fingerprints

    if not 0 <= bucket < BUCKETS:
        raise ValueError("Unknown identity bucket")
    frozen_source()
    _, out = paths()
    conversion = json.loads((out / "conversion.json").read_text())
    if conversion["status"] != "complete":
        raise ValueError("Completed source conversion required")
    destination = out / "identity" / f"bucket-{bucket:02d}.json"
    if destination.exists():
        return {"bucket": bucket, "status": "complete", "cached": True}
    raw = (
        pl.scan_parquet(out / "raw" / f"bucket-{bucket:02d}" / "*.parquet")
        .select(["user_id", "time_ms", "tab"])
        .collect()
    )
    records = identity_fingerprints(
        raw["user_id"].to_numpy(), raw["time_ms"].to_numpy(), raw["tab"].to_numpy()
    )
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps({"records": records, "raw_rows": raw.height}, indent=2))
    volume.commit()
    return {"bucket": bucket, "status": "complete", "users": len(records), "raw_rows": raw.height}


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=7200,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def identify() -> dict:
    import json
    from behavioral_equivalence.human_protocol import confirmatory_cohort

    frozen_source()
    _, out = paths()
    results = list(fingerprint_bucket.map(range(BUCKETS), return_exceptions=True))
    if any(isinstance(r, Exception) or r["status"] != "complete" for r in results):
        raise RuntimeError(f"Identity bucket failure: {results}")
    volume.reload()
    final = []
    for b in range(BUCKETS):
        final.extend(json.loads((out / "identity" / f"bucket-{b:02d}.json").read_text())["records"])
    pilot = json.loads(
        Path("/artifacts/datasets/kuairand-1k/prepared-v4/pilot-user-fingerprints.json").read_text()
    )["records"]
    if len(final) != 27285 or len(pilot) != 1000:
        raise ValueError("User counts differ from the specified source releases")
    cohort = confirmatory_cohort(pilot, final, 20261015)
    (out / "cohort-private.json").write_text(json.dumps(cohort, indent=2))
    result = {k: v for k, v in cohort.items() if k not in {"excluded_user_ids", "user_assignment"}}
    result.update(
        status="complete",
        outcome_performance="not inspected",
        phase="pilot exclusion and user assignment",
    )
    (out / "cohort-report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(16384, 16384),
    timeout=1200,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def prepare_bucket(bucket: int) -> dict:
    import hashlib
    import json
    import time
    import polars as pl
    from behavioral_equivalence.human_protocol import SOURCE_COLUMNS
    from behavioral_equivalence.human_preparation import prepare_tables, validate_source

    if not 0 <= bucket < BUCKETS:
        raise ValueError("Unknown preparation bucket")
    frozen_source()
    _, out = paths()
    destination = out / "data" / f"bucket-{bucket:02d}"
    if (destination / "report.json").exists():
        return json.loads((destination / "report.json").read_text())
    cohort = json.loads((out / "cohort-private.json").read_text())
    start = time.monotonic()
    logs = pl.scan_parquet(out / "raw" / f"bucket-{bucket:02d}" / "*.parquet")
    raw_rows = logs.select(pl.len()).collect().item()
    validate_source(logs)
    eligible = logs.filter(~pl.col("user_id").is_in(cohort["excluded_user_ids"]))
    eligible_rows = eligible.select(pl.len()).collect().item()
    unique = eligible.unique(subset=list(SOURCE_COLUMNS)).collect()
    assignment = {int(k): v for k, v in cohort["user_assignment"].items()}
    ordered, bundles = prepare_tables(unique, assignment)
    destination.mkdir(parents=True, exist_ok=True)
    ordered.write_parquet(destination / "interactions.parquet")
    bundles.write_parquet(destination / "timestamp-bundles.parquet")

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            while block := f.read(8 * 1024**2):
                h.update(block)
        return h.hexdigest()

    result = {
        "status": "complete",
        "bucket": bucket,
        "raw_rows": raw_rows,
        "eligible_raw_rows": eligible_rows,
        "rows_kept": ordered.height,
        "full_record_duplicates_removed": eligible_rows - ordered.height,
        "users": ordered["user_id"].n_unique(),
        "timestamp_bundles": bundles.height,
        "source_digest": source_snapshot()["sha256"],
        "interactions_sha256": digest(destination / "interactions.parquet"),
        "bundles_sha256": digest(destination / "timestamp-bundles.parquet"),
        "wall_seconds": time.monotonic() - start,
        "timing_scope": "max and summed duration proxies retained; actual outcome availability is not established by documentation",
        "outcome_performance": "not inspected",
    }
    (destination / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=21600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def prepare_all() -> dict:
    import json

    frozen_source()
    _, out = paths()
    records = list(prepare_bucket.map(range(BUCKETS), return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in records if isinstance(r, Exception)]
    if failures:
        raise RuntimeError(f"Candidate preparation failed: {failures}")
    volume.reload()
    result = {
        "status": "complete",
        "phase": "candidate dataset preparation; independent audit and confirmatory freeze still required",
        "source_digest": source_snapshot()["sha256"],
        "buckets": records,
        "users": sum(r["users"] for r in records),
        "rows_kept": sum(r["rows_kept"] for r in records),
        "timestamp_bundles": sum(r["timestamp_bundles"] for r in records),
        "outcome_performance": "not inspected",
    }
    if result["users"] != 26285:
        raise ValueError("Final user count does not prove complete pilot exclusion")
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(1024, 1024),
    timeout=120,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def smoke() -> dict:
    import json
    import subprocess

    saved = source_snapshot()
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_human_preparation.py",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    report = {
        "status": "complete" if result.returncode == 0 else "failed",
        "source_digest": saved["sha256"],
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "scope": "synthetic fixture integration of full-record dedup, timestamp bundles, split boundaries and source validation",
    }
    destination = Path("/artifacts/data-preparation-smoke-v1") / saved["sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    (destination / "source.json").write_text(json.dumps(saved, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(stage: str = "convert"):
    import json

    if stage not in {"convert", "identify", "prepare", "smoke"}:
        raise ValueError("Stage must be convert, identify, prepare or smoke")
    function = {"convert": convert, "identify": identify, "prepare": prepare_all, "smoke": smoke}[
        stage
    ]
    call = function.spawn()
    print(
        json.dumps(
            {"stage": stage, "function_call_id": call.object_id, "function_id": function.object_id}
        ),
        flush=True,
    )
    result = call.get()
    if result.get("status") != "complete":
        raise RuntimeError("Cloud preparation stage did not pass: " + json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "buckets"}, indent=2), flush=True)
