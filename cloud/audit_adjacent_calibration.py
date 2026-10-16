"""Independent full-source audit of adjacent calibration selection and histories."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = Path(
    "/artifacts/datasets/kuairand-27k/prepared-candidate-v2/adjacent-calibration-314b2e408cb56897"
)
REPORT_SHA = "ded969e03e56d879bb7a2dbff3aa637c69eee821e45024125103a7a21ea09b19"
app = modal.App("behavioral-equivalence-adjacent-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pyarrow==20.0.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "ARROW_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_adjacent_audit.py", remote_path="/research/test_adjacent_audit.py"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def sources():
    import hashlib
    import json

    remote = Path("/research/test_adjacent_audit.py").exists()
    base = Path("/research") if remote else ROOT
    saved = {
        "audit_adjacent_calibration.py": Path(__file__).read_text(),
        "test_adjacent_audit.py": (
            base / ("test_adjacent_audit.py" if remote else "tests/test_adjacent_audit.py")
        ).read_text(),
    }
    for name in [
        "adjacent_audit.py",
        "adjacent_calibration.py",
        "features.py",
        "calendar_splits.py",
    ]:
        saved[name] = (base / "src/behavioral_equivalence" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def inputs():
    import hashlib
    import json

    if checksum(CALIBRATION / "report.json") != REPORT_SHA:
        raise ValueError("Calibration report changed")
    report = json.loads((CALIBRATION / "report.json").read_text())
    original_source = json.loads((CALIBRATION / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(original_source, sort_keys=True).encode()).hexdigest()
        != report["source_digest"]
    ):
        raise ValueError("Calibration source changed")
    if report["status"] != "complete" or len(report["buckets"]) != 32:
        raise ValueError("All calibration inputs required")
    return report, json.loads(original_source["config.json"])


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=180,
    retries=0,
    volumes={"/artifacts": volume},
)
def smoke():
    import subprocess
    import json

    saved, digest = sources()
    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/test_adjacent_audit.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = CALIBRATION / ("independent-audit-" + digest[:16])
    out.mkdir(exist_ok=True)
    report = dict(
        status="complete", audit_source_digest=digest, stdout=result.stdout, stderr=result.stderr
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "smoke.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=900,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_bucket(bucket: int):
    import json
    import time
    import numpy as np
    import pyarrow.parquet as pq
    from behavioral_equivalence.adjacent_audit import audit_adjacent_cases

    if not isinstance(bucket, int) or not 0 <= bucket < 32:
        raise ValueError("Known prepared bucket required")
    volume.reload()
    started = time.monotonic()
    report, config = inputs()
    _, digest = sources()
    out = CALIBRATION / ("independent-audit-" + digest[:16])
    prepared = Path("/artifacts") / config["prepared_root"]
    if (
        checksum(prepared / "report.json") != config["prepared_report_sha256"]
        or checksum(prepared / "independent-audit.json") != config["independent_audit_sha256"]
    ):
        raise ValueError("Independently audited source report changed")
    source_report = json.loads((prepared / "report.json").read_text())["buckets"][bucket]
    source = prepared / "data" / f"bucket-{bucket:02d}"
    for name, expected in [
        ("interactions.parquet", source_report["interactions_sha256"]),
        ("timestamp-bundles.parquet", source_report["bundles_sha256"]),
    ]:
        if checksum(source / name) != expected:
            raise ValueError("Source physical rows changed")
    cr = report["buckets"][bucket]
    case_path = CALIBRATION / f"bucket-{bucket:02d}" / "calibration.npz"
    if checksum(case_path) != cr["output_sha256"]["calibration.npz"]:
        raise ValueError("Calibration cases changed")
    target = out / f"bucket-{bucket:02d}.json"
    if target.exists():
        cached = json.loads(target.read_text())
        if cached["calibration_npz_sha256"] != cr["output_sha256"]["calibration.npz"]:
            raise ValueError("Cached audited input differs")
        return cached

    def read(path, columns):
        table = pq.read_table(path, columns=columns)
        return {name: table[name].to_numpy(zero_copy_only=False) for name in columns}

    events = read(
        source / "interactions.parquet",
        ["user_id", "time_ms", "tab", "video_id", "is_rand", "is_click"],
    )
    bundles = read(
        source / "timestamp-bundles.parquet",
        ["user_id", "time_ms", "available_sum_ms", "clicks", "exposures", "user_split"],
    )
    with np.load(case_path, allow_pickle=False) as a:
        cases = {name: a[name] for name in a.files}
    result = audit_adjacent_cases(events, bundles, cases, bucket, config)
    metadata = Path("/artifacts") / config["metadata_root"]
    if checksum(metadata / "report.json") != config["metadata_report_sha256"]:
        raise ValueError("Metadata report changed")
    mr = json.loads((metadata / "report.json").read_text())
    for name in ["duration-ms.npy", "upload-day.npy"]:
        if checksum(metadata / name) != mr["output_sha256"][name]:
            raise ValueError("Metadata arrays changed")
    ids = cases["video_id"]
    duration = np.load(metadata / "duration-ms.npy", mmap_mode="r", allow_pickle=False)[ids]
    upload = np.load(metadata / "upload-day.npy", mmap_mode="r", allow_pickle=False)[ids]
    known = (
        np.isfinite(duration)
        & (duration > 0)
        & (upload != np.iinfo(np.int32).min)
        & (upload < cases["time_ms"] // 86400000)
    )
    context = np.zeros((len(ids), 2), dtype=np.float32)
    context[known, 0] = np.log1p(duration[known] / 1000)
    context[:, 1] = ~known
    if not np.array_equal(context, cases["intrinsic_context"]):
        raise ValueError("Intrinsic context differs from independent source formula")
    result.update(
        audit_source_digest=digest,
        calibration_report_sha256=REPORT_SHA,
        calibration_npz_sha256=cr["output_sha256"]["calibration.npz"],
        intrinsic_context_checked=True,
        wall_seconds=time.monotonic() - started,
    )
    target.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=86400,
    retries=0,
    volumes={"/artifacts": volume},
)
def full():
    import json

    volume.reload()
    _, digest = sources()
    out = CALIBRATION / ("independent-audit-" + digest[:16])
    if json.loads((out / "smoke.json").read_text())["status"] != "complete":
        raise ValueError("Passing source-bound independent-audit smoke required")
    results = list(audit_bucket.map(range(32), return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in results if isinstance(r, Exception)]
    if failures:
        raise RuntimeError("Adjacent calibration audit failed: " + str(failures))
    result = dict(
        status="complete",
        audit_source_digest=digest,
        calibration_report_sha256=REPORT_SHA,
        buckets=results,
        bucket_count=len(results),
        queries=sum(r["queries_audited"] for r in results),
        known_transitions=sum(r["transitions_audited"] for r in results),
        known_users=sum(r["known_users"] for r in results),
        unseen_transitions=sum(r["unseen_transitions"] for r in results),
        unseen_users=sum(r["unseen_users"] for r in results),
        root=str(out),
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "buckets"}


@app.local_entrypoint()
def main(stage: str = "smoke"):
    import json
    import sys

    if stage == "smoke":
        print(json.dumps(smoke.remote()), flush=True)
        return
    if stage != "full":
        raise ValueError("Smoke or full stage required")
    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = sources()
    reservation = BudgetLedger(ROOT / ".campaign/adjacent-audit-budget.sqlite", 5).reserve(
        digest,
        32 * 900 * (2 * 1.31e-5 + 8 * 2.22e-6) + 86400 * (0.5 * 1.31e-5 + 2.22e-6) + 0.1,
    )
    state = ROOT / ".campaign" / ("adjacent-audit-" + digest[:16] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = full.spawn()
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    function_id=full.object_id,
                    audit_source_digest=digest,
                    reservation=reservation,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
