"""Independently audit every constructed case, without predictive scoring."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/artifacts/datasets/kuairand-27k/prepared-candidate-v2")
CASES = DATA / "confirmatory-cases-d05a64f50a264877"
app = modal.App("behavioral-equivalence-confirmatory-case-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "polars==1.31.0", "pyarrow==20.0.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_case_audit.py", remote_path="/research/tests/test_case_audit.py"
    )
)


def snapshot() -> tuple[dict, str]:
    import hashlib
    import json

    sources = {
        "audit_confirmatory_cases.py": Path(__file__).read_text(),
        "test_case_audit.py": Path("/research/tests/test_case_audit.py").read_text(),
    }
    for name in [
        "case_audit.py",
        "confirmatory_preparation.py",
        "calendar_splits.py",
        "features.py",
    ]:
        sources[name] = (Path("/research/src/behavioral_equivalence") / name).read_text()
    return sources, hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=180,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def smoke() -> dict:
    import json
    import subprocess

    saved, digest = snapshot()
    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_case_audit.py"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = CASES / ("independent-audit-" + digest[:16])
    out.mkdir(exist_ok=True)
    report = {
        "status": "complete" if result.returncode == 0 else "failed",
        "source_digest": digest,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    (out / "smoke.json").write_text(json.dumps(report, indent=2))
    (out / "source.json").write_text(json.dumps(saved, indent=2))
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
def audit_bucket(bucket: int) -> dict:
    import json
    import time
    import numpy as np
    import pyarrow.parquet as pq
    from behavioral_equivalence.case_audit import audit_cases

    if not 0 <= bucket < 32:
        raise ValueError("Valid bucket required")
    began = time.monotonic()
    _, digest = snapshot()
    source_report = json.loads((CASES / "report.json").read_text())
    config = source_report["config"]
    record = source_report["buckets"][bucket]
    source = DATA / "data" / f"bucket-{bucket:02d}"
    packed = CASES / f"bucket-{bucket:02d}"
    if any(sha256(source / name) != checksum for name, checksum in record["input_sha256"].items()):
        raise ValueError("Audited prepared source changed")
    if any(sha256(packed / name) != checksum for name, checksum in record["output_sha256"].items()):
        raise ValueError("Case artifacts changed after construction")
    out = CASES / ("independent-audit-" + digest[:16]) / f"bucket-{bucket:02d}.json"
    if out.exists():
        cached = json.loads(out.read_text())
        if (
            cached["case_report_sha256"] != sha256(CASES / "report.json")
            or cached["source_digest"] != digest
        ):
            raise ValueError("Cached audit inputs changed")
        return cached
    # PyArrow physical-row reads are independent of the Polars preparation joins.
    events_table = pq.read_table(
        source / "interactions.parquet",
        columns=["user_id", "time_ms", "tab", "video_id", "is_rand", "is_click"],
    )
    bundle_table = pq.read_table(
        source / "timestamp-bundles.parquet",
        columns=["user_id", "time_ms", "available_sum_ms", "clicks", "exposures", "user_split"],
    )
    events = {
        name: events_table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in events_table.column_names
    }
    bundles = {
        name: bundle_table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in bundle_table.column_names
    }
    records = []
    for hours in config["embargo_hours"]:
        with np.load(packed / f"embargo-{hours:02d}h.npz", allow_pickle=False) as saved:
            cases = {name: saved[name] for name in saved.files}
        records.append(audit_cases(events, bundles, cases, bucket, config, hours))
    report = {
        "status": "complete",
        "bucket": bucket,
        "source_digest": digest,
        "case_report_sha256": sha256(CASES / "report.json"),
        "input_sha256": record["input_sha256"],
        "case_artifact_sha256": record["output_sha256"],
        "records": records,
        "wall_seconds": time.monotonic() - began,
    }
    out.write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=7200,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_all() -> dict:
    import json

    _, digest = snapshot()
    out = CASES / ("independent-audit-" + digest[:16])
    if json.loads((out / "smoke.json").read_text())["status"] != "complete":
        raise ValueError("Passing source-bound corruption and reconstruction test required")
    source = json.loads((CASES / "report.json").read_text())
    if source["status"] != "complete" or len(source["buckets"]) != 32:
        raise ValueError("Completed full case construction required")
    records = list(audit_bucket.map(range(32), return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in records if isinstance(r, Exception)]
    if failures:
        raise RuntimeError(
            "Independent case audit failed; successful reports remain cached: " + str(failures)
        )
    counts = []
    for i, hours in enumerate(source["config"]["embargo_hours"]):
        actual = sum(r["records"][i]["cases_audited"] for r in records)
        expected = sum(
            r["cases"]
            for r in source["support"]
            if r["protocol"] == "known_users" and r["embargo_hours"] == hours
        )
        if actual != expected:
            raise ValueError("Case audit coverage differs from construction manifest")
        counts.append(
            {
                "embargo_hours": hours,
                "cases_audited": actual,
                "unique_bundle_queries_audited": sum(
                    r["records"][i]["unique_bundle_queries_audited"] for r in records
                ),
            }
        )
    result = {
        "status": "complete",
        "phase": "independent full sampled-case audit, not predictive scoring",
        "source_digest": digest,
        "case_report_sha256": sha256(CASES / "report.json"),
        "counts": counts,
        "buckets": records,
        "scope": "Every physical source record binding, priority/cap sample, shared-calendar eligibility and history token under each declared embargo reconstructed.",
        "shared_primitive": "Pinned Polars case-ID hash, not the grouping, selection, calendar or history algorithms.",
        "limits": "Real feedback observation times remain unavailable; no human model fidelity or causal transfer claim.",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "buckets"} | {
        "report": str(out / "report.json")
    }


@app.local_entrypoint()
def main(stage: str = "smoke"):
    import json
    import sys

    if stage not in {"smoke", "audit"}:
        raise ValueError("Stage must be smoke or audit")
    if stage == "audit":
        sys.path.insert(0, str(ROOT / "src"))
        from behavioral_equivalence.budget import BudgetLedger

        amount = 32 * 600 * (2 * 1.31e-5 + 8 * 2.22e-6) + 7200 * (0.5 * 1.31e-5 + 2.22e-6) + 1
        print(
            json.dumps(
                BudgetLedger(ROOT / ".campaign/human-case-audit-budget.sqlite", 4).reserve(
                    "human-case-audit-v1", amount
                )
            ),
            flush=True,
        )
    function = smoke if stage == "smoke" else audit_all
    call = function.spawn()
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": function.object_id}),
        flush=True,
    )
    result = call.get()
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("Case audit did not pass")
