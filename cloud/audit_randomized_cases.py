"""Independent full-source audit of randomized exposures, histories and contexts."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
RANDOM_COHORT = Path(
    "/artifacts/datasets/kuairand-27k/prepared-candidate-v2/randomized-cases-60b69598b1185ce7"
)
app = modal.App("behavioral-equivalence-randomized-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pyarrow==20.0.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "ARROW_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
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

    remote = Path("/research/tests/test_randomized_audit.py").exists()
    base = Path("/research") if remote else ROOT
    saved = {
        "audit_randomized_cases.py": Path(__file__).read_text(),
        "test_randomized_audit.py": (base / "tests/test_randomized_audit.py").read_text(),
        "test_adjacent_audit.py": (base / "tests/test_adjacent_audit.py").read_text(),
    }
    for name in [
        "randomized_audit.py",
        "randomized_cohort.py",
        "adjacent_audit.py",
        "adjacent_calibration.py",
        "features.py",
        "calendar_splits.py",
        "intrinsic_context.py",
    ]:
        saved[name] = (base / "src/behavioral_equivalence" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def inputs(report_sha256: str):
    import hashlib
    import json

    if checksum(RANDOM_COHORT / "report.json") != report_sha256:
        raise ValueError("Randomized cohort report changed")
    report = json.loads((RANDOM_COHORT / "report.json").read_text())
    original_source = json.loads((RANDOM_COHORT / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(original_source, sort_keys=True).encode()).hexdigest()
        != report["source_digest"]
    ):
        raise ValueError("Randomized cohort source changed")
    if report["status"] != "complete" or len(report["buckets"]) != 32:
        raise ValueError("All randomized inputs required")
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
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_randomized_audit.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = RANDOM_COHORT / ("independent-audit-" + digest[:16])
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
    timeout=600,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_bucket(bucket: int, report_sha256: str):
    import json
    import time
    import numpy as np
    import pyarrow.parquet as pq
    from behavioral_equivalence.randomized_audit import audit_randomized_cases

    if not isinstance(bucket, int) or not 0 <= bucket < 32:
        raise ValueError("Known prepared bucket required")
    volume.reload()
    started = time.monotonic()
    report, config = inputs(report_sha256)
    _, digest = sources()
    out = RANDOM_COHORT / ("independent-audit-" + digest[:16])
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
    case_path = RANDOM_COHORT / f"bucket-{bucket:02d}" / "randomized.npz"
    if checksum(case_path) != cr["output_sha256"]["randomized.npz"]:
        raise ValueError("Randomized cohort cases changed")
    target = out / f"bucket-{bucket:02d}.json"
    if target.exists():
        cached = json.loads(target.read_text())
        if cached["cohort_npz_sha256"] != cr["output_sha256"]["randomized.npz"]:
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
    result = audit_randomized_cases(events, bundles, cases, bucket, config)
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
    group = np.full(len(ids), 4, dtype=np.int8)
    group[known & (duration <= 7000)] = 0
    group[known & (duration > 7000) & (duration <= 18000)] = 1
    group[known & (duration > 18000) & (duration <= 60000)] = 2
    group[known & (duration > 60000)] = 3
    environments = 5 * cases["tab"] + group
    if not np.array_equal(group, cases["duration_group"]) or not np.array_equal(
        environments, cases["environment"]
    ):
        raise ValueError("Environment groups differ from independent threshold formula")
    for cell in cr["support"]:
        role = ("train", "validation", "calibration", "test").index(cell["role"])
        chosen = cases["role_code"] == role
        if cell["protocol"] == "unseen_users":
            chosen &= cases["user_role_code"] == role
        if cell["exposures"] != int(chosen.sum()) or cell["users"] != len(
            np.unique(cases["user_id"][chosen])
        ):
            raise ValueError("Reported role support differs from source")
        if len(cell["environments"]) != 75 or [
            c["environment"] for c in cell["environments"]
        ] != list(range(75)):
            raise ValueError("Complete environment support grid required")
        for e in cell["environments"]:
            match = chosen & (environments == e["environment"])
            if e["exposures"] != int(match.sum()) or e["users"] != len(
                np.unique(cases["user_id"][match])
            ):
                raise ValueError("Reported environment support differs from source")
    result.update(
        audit_source_digest=digest,
        cohort_report_sha256=report_sha256,
        cohort_npz_sha256=cr["output_sha256"]["randomized.npz"],
        intrinsic_context_checked=True,
        environment_and_support_checked=True,
        wall_seconds=time.monotonic() - started,
    )
    target.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=3600,
    retries=0,
    volumes={"/artifacts": volume},
)
def full(report_sha256: str):
    import json

    volume.reload()
    _, digest = sources()
    out = RANDOM_COHORT / ("independent-audit-" + digest[:16])
    if json.loads((out / "smoke.json").read_text())["status"] != "complete":
        raise ValueError("Passing source-bound independent-audit smoke required")
    results = list(
        audit_bucket.map(
            range(32), [report_sha256] * 32, return_exceptions=True, order_outputs=True
        )
    )
    failures = [repr(r) for r in results if isinstance(r, Exception)]
    if failures:
        raise RuntimeError("Randomized cohort audit failed: " + str(failures))
    result = dict(
        status="complete",
        audit_source_digest=digest,
        cohort_report_sha256=report_sha256,
        buckets=results,
        bucket_count=len(results),
        random_exposures=sum(r["random_exposures_audited"] for r in results),
        query_bundles=sum(r["query_bundles_audited"] for r in results),
        support=[
            dict(
                protocol=p,
                role=role,
                exposures=sum(
                    next(
                        s["exposures"]
                        for s in r["support"]
                        if s["protocol"] == p and s["role"] == role
                    )
                    for r in results
                ),
                users=sum(
                    next(
                        s["users"] for s in r["support"] if s["protocol"] == p and s["role"] == role
                    )
                    for r in results
                ),
            )
            for p in ["known_users", "unseen_users"]
            for role in ["train", "validation", "calibration", "test"]
        ],
        root=str(out),
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "buckets"}


@app.local_entrypoint()
def main(stage: str = "smoke", report_sha256: str = ""):
    import json
    import sys

    if stage not in {"smoke", "full"}:
        raise ValueError("Smoke or full stage required")
    if stage == "full" and (
        len(report_sha256) != 64 or any(c not in "0123456789abcdef" for c in report_sha256)
    ):
        raise ValueError("Pinned complete cohort report SHA256 required")
    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = sources()
    amount = (
        32 * 600 * (2 * 1.31e-5 + 8 * 2.22e-6)
        + 3600 * (0.5 * 1.31e-5 + 2.22e-6)
        + 180 * (1.31e-5 + 2 * 2.22e-6)
        + 0.11
    )
    reservation = BudgetLedger(ROOT / ".campaign/randomized-audit-budget.sqlite", 1).reserve(
        digest, amount
    )
    state = (
        ROOT
        / ".campaign"
        / ("randomized-audit-" + digest[:16] + "-" + stage + "-" + report_sha256[:12] + ".json")
    )
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = smoke.spawn() if stage == "smoke" else full.spawn(report_sha256)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    source_digest=digest,
                    cohort_report_sha256=report_sha256,
                    reservation=reservation,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
