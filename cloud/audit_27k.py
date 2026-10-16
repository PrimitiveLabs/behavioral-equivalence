"""Full-data independent audit of the frozen candidate 27K preparation."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-kuairand27k-independent-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "duckdb==1.3.2", "numpy==2.2.6", "pyarrow==20.0.0", "polars==1.31.0", "pytest==8.4.1"
    )
    .env({"PYTHONPATH": "/research/src", "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_data_audit.py", remote_path="/research/tests/test_data_audit.py"
    )
)
DATA = Path("/artifacts/datasets/kuairand-27k/prepared-candidate-v2")


def source_files() -> dict:
    return {
        "audit_27k.py": Path(__file__).read_text(),
        **{
            name: Path("/research/src/behavioral_equivalence" + "/" + name).read_text()
            for name in ["data_audit.py", "human_protocol.py", "human_preparation.py"]
        },
        "test_data_audit.py": Path("/research/tests/test_data_audit.py").read_text(),
    }


def source_digest() -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(source_files(), sort_keys=True).encode()).hexdigest()


def file_digest(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=1200,
    retries=0,
    max_containers=4,
    volumes={"/artifacts": volume},
)
def audit_one(bucket: int) -> dict:
    import json
    import time
    from behavioral_equivalence.data_audit import audit_bucket

    if not 0 <= bucket < 32:
        raise ValueError("Invalid data bucket")
    source = json.loads((DATA / "report.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed candidate preparation required")
    cohort = json.loads((DATA / "cohort-private.json").read_text())
    raw = sorted((DATA / "raw" / f"bucket-{bucket:02d}").glob("*.parquet"))
    if len(raw) != 5:
        raise ValueError("All five source shards required per bucket")
    prepared = DATA / "data" / f"bucket-{bucket:02d}"
    events, bundles = prepared / "interactions.parquet", prepared / "timestamp-bundles.parquet"
    inputs = {f.name: file_digest(f) for f in [*raw, events, bundles]}
    bucket_report = json.loads((prepared / "report.json").read_text())
    if (
        inputs["interactions.parquet"] != bucket_report["interactions_sha256"]
        or inputs["timestamp-bundles.parquet"] != bucket_report["bundles_sha256"]
    ):
        raise ValueError("Prepared artifacts differ from recorded hashes")
    destination = DATA / ("independent-audit-" + source_digest()[:16]) / f"bucket-{bucket:02d}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["input_sha256"] != inputs:
            raise ValueError("Cached audit inputs changed")
        return saved
    started = time.monotonic()
    identities = json.loads((DATA / "identity" / f"bucket-{bucket:02d}.json").read_text())[
        "records"
    ]
    result = audit_bucket(
        raw,
        events,
        bundles,
        cohort["excluded_user_ids"],
        {int(k): v for k, v in cohort["user_assignment"].items()},
        identities,
    )
    result.update(
        bucket=bucket,
        source_digest=source_digest(),
        input_sha256=inputs,
        wall_seconds=time.monotonic() - started,
    )
    if (
        result["raw_rows"] != bucket_report["raw_rows"]
        or result["rows_audited"] != bucket_report["rows_kept"]
        or result["bundles_audited"] != bucket_report["timestamp_bundles"]
    ):
        raise ValueError("Independent row counts differ from preparation manifest")
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(destination)
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=14400,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def audit_all() -> dict:
    import hashlib
    import json

    cohort = json.loads((DATA / "cohort-private.json").read_text())
    source = json.loads((DATA / "report.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed candidate preparation required")
    identities = []
    for b in range(32):
        identities.extend(
            json.loads((DATA / "identity" / f"bucket-{b:02d}.json").read_text())["records"]
        )
    by_hash = {r["sha256"]: r for r in identities}
    pilot = json.loads(
        Path("/artifacts/datasets/kuairand-1k/prepared-v4/pilot-user-fingerprints.json").read_text()
    )["records"]
    if len(identities) != 27285 or len(by_hash) != 27285 or len(pilot) != 1000:
        raise ValueError("Nonunique or incomplete identity universe")
    excluded = {
        by_hash[r["sha256"]]["user_id"]
        for r in pilot
        if r["sha256"] in by_hash and by_hash[r["sha256"]]["rows"] == r["rows"]
    }
    if len(excluded) != 1000 or excluded != set(cohort["excluded_user_ids"]):
        raise ValueError("Pilot exclusion does not match the independent identity lookup")
    eligible = sorted(r["user_id"] for r in identities if r["user_id"] not in excluded)
    ranked = sorted(eligible, key=lambda u: hashlib.sha256(f"20261015:{u}".encode()).digest())
    expected = {
        str(uid): "train"
        if i < int(0.7 * len(ranked))
        else "validation"
        if i < int(0.8 * len(ranked))
        else "calibration"
        if i < int(0.9 * len(ranked))
        else "test"
        for i, uid in enumerate(ranked)
    }
    if expected != cohort["user_assignment"]:
        raise ValueError("Outcome-independent user split assignment changed")
    snapshot = DATA / ("independent-audit-" + source_digest()[:16])
    snapshot.mkdir(exist_ok=True)
    (snapshot / "source.json").write_text(json.dumps(source_files(), indent=2))
    volume.commit()
    records = list(audit_one.map(range(32), return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in records if isinstance(r, Exception)]
    if failures:
        raise RuntimeError(
            "Independent bucket audits failed; successful hashes remain cached: " + str(failures)
        )
    if (
        sum(r["eligible_users"] for r in records) != 26285
        or sum(r["excluded_pilot_users"] for r in records) != 1000
        or sum(r["rows_audited"] for r in records) != source["rows_kept"]
    ):
        raise ValueError("Incomplete cohort or row coverage")
    result = {
        "status": "complete",
        "phase": "independent full candidate-data audit, not model evaluation",
        "source_digest": source_digest(),
        "prepared_report_sha256": file_digest(DATA / "report.json"),
        "cohort_sha256": file_digest(DATA / "cohort-private.json"),
        "raw_rows_audited": sum(r["raw_rows"] for r in records),
        "rows_audited": sum(r["rows_audited"] for r in records),
        "bundles_audited": sum(r["bundles_audited"] for r in records),
        "users_audited": 26285,
        "pilot_users_excluded": 1000,
        "bucket_reports": records,
        "limits": [
            "Checks converted source records, not a second complete CSV parse.",
            "Checks timing proxies and leakage guards; does not prove real-time feedback availability.",
            "Does not inspect predictive performance or establish human Markov sufficiency.",
        ],
    }
    (DATA / "independent-audit.json").write_text(json.dumps(result, indent=2))
    (DATA / "independent-audit-source.py").write_bytes(Path(__file__).read_bytes())
    volume.commit()
    return {k: v for k, v in result.items() if k != "bucket_reports"}


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=180,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def smoke() -> dict:
    import subprocess
    import json

    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_data_audit.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = {
        "status": "complete" if result.returncode == 0 else "failed",
        "source_digest": source_digest(),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    root = Path("/artifacts/data-audit-smoke-v1") / source_digest()
    root.mkdir(parents=True, exist_ok=True)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    (root / "source.json").write_text(json.dumps(source_files(), indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(stage: str = "smoke"):
    import json

    if stage not in {"smoke", "audit"}:
        raise ValueError("Stage must be smoke or audit")
    function = smoke if stage == "smoke" else audit_all
    call = function.spawn()
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": function.object_id}),
        flush=True,
    )
    result = call.get()
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("Independent audit stage failed")
