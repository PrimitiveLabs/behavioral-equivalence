"""Outcome-free shared-calendar support audit of all eligible 27K bundles."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/artifacts/datasets/kuairand-27k/prepared-candidate-v2")
app = modal.App("behavioral-equivalence-calendar-support-27k-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "polars==1.31.0")
    .env({"PYTHONPATH": "/research/src", "POLARS_MAX_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def snapshot() -> tuple[dict, str]:
    import hashlib
    import json

    sources = {"calendar_support_27k.py": Path(__file__).read_text()}
    for name in ["calendar_splits.py", "calendar_support.py"]:
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
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=300,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def inspect_bucket(bucket: int) -> dict:
    import json
    import time
    import polars as pl
    from behavioral_equivalence.calendar_splits import CONFIRMATORY_CANDIDATE
    from behavioral_equivalence.calendar_support import support_by_role

    if not 0 <= bucket < 32:
        raise ValueError("Known data bucket required")
    start = time.monotonic()
    _, digest = snapshot()
    path = DATA / "data" / f"bucket-{bucket:02d}" / "timestamp-bundles.parquet"
    source = json.loads((path.parent / "report.json").read_text())
    if sha256(path) != source["bundles_sha256"]:
        raise ValueError("Prepared bundle input changed")
    out = DATA / ("calendar-support-" + digest[:16])
    out.mkdir(exist_ok=True)
    destination = out / f"bucket-{bucket:02d}.json"
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached["source_digest"] != digest or cached["input_sha256"] != source["bundles_sha256"]:
            raise ValueError("Cached support inputs changed")
        return cached
    # Deliberately do not read click counts, event targets or model predictions.
    bundles = pl.read_parquet(
        path,
        columns=[
            "user_id",
            "time_ms",
            "available_ms",
            "available_sum_ms",
            "user_split",
            "exposures",
        ],
    ).sort(["user_id", "time_ms"])
    users, times, roles, exposures = [
        bundles[name].to_numpy() for name in ["user_id", "time_ms", "user_split", "exposures"]
    ]
    records = []
    for proxy in ["available_ms", "available_sum_ms"]:
        availability = bundles[proxy].to_numpy()
        for hours in [0, 24, 48]:
            for protocol in ["known_users", "unseen_users"]:
                record = support_by_role(
                    CONFIRMATORY_CANDIDATE,
                    users,
                    times,
                    availability,
                    exposures,
                    roles,
                    protocol,
                    hours * 3600 * 1000,
                )
                record["availability_proxy"] = proxy
                records.append(record)
    result = {
        "status": "complete",
        "bucket": bucket,
        "source_digest": digest,
        "input_sha256": source["bundles_sha256"],
        "users": int(bundles["user_id"].n_unique()),
        "bundles": len(bundles),
        "cases": int(exposures.sum()),
        "records": records,
        "wall_seconds": time.monotonic() - start,
    }
    (destination).write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=3600,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run() -> dict:
    import json
    from dataclasses import asdict
    from behavioral_equivalence.calendar_splits import CONFIRMATORY_CANDIDATE

    audit = json.loads((DATA / "independent-audit.json").read_text())
    if audit["status"] != "complete" or audit["prepared_report_sha256"] != sha256(
        DATA / "report.json"
    ):
        raise ValueError("Complete independent audit of the current prepared data required")
    sources, digest = snapshot()
    out = DATA / ("calendar-support-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(sources, indent=2))
    volume.commit()
    buckets = list(inspect_bucket.map(range(32), return_exceptions=True, order_outputs=True))
    failures = [repr(b) for b in buckets if isinstance(b, Exception)]
    if failures:
        raise RuntimeError("Calendar support audit failed: " + str(failures))
    if (
        sum(b["users"] for b in buckets) != audit["users_audited"]
        or sum(b["bundles"] for b in buckets) != audit["bundles_audited"]
        or sum(b["cases"] for b in buckets) != audit["rows_audited"]
    ):
        raise ValueError("Support diagnostics did not cover all audited data")
    records = []
    for i in range(len(buckets[0]["records"])):
        entries = [b["records"][i] for b in buckets]
        merged = {key: entries[0][key] for key in ["protocol", "embargo_ms", "availability_proxy"]}
        merged.update(
            {key: sum(e[key] for e in entries) for key in ["excluded_cases", "excluded_bundles"]}
        )
        merged["roles"] = []
        for r in range(4):
            parts = [e["roles"][r] for e in entries]
            summary = {"role": parts[0]["role"]}
            summary.update(
                {key: sum(p[key] for p in parts) for key in ["cases", "bundles", "users"]}
            )
            for key in [
                "first_timestamp_ms",
                "last_timestamp_ms",
                "latest_label_available_plus_embargo_ms",
            ]:
                present = [p[key] for p in parts if p[key] is not None]
                summary[key] = (
                    (min(present) if key == "first_timestamp_ms" else max(present))
                    if present
                    else None
                )
            summary["cases_by_available_history_0_1to7_8to31_32plus"] = [
                sum(p["cases_by_available_history_0_1to7_8to31_32plus"][j] for p in parts)
                for j in range(4)
            ]
            merged["roles"].append(summary)
        records.append(merged)
    result = {
        "status": "complete",
        "phase": "outcome-free support diagnostics, not final confirmatory scoring",
        "source_digest": digest,
        "calendar": asdict(CONFIRMATORY_CANDIDATE),
        "prepared_report_sha256": audit["prepared_report_sha256"],
        "independent_audit_sha256": sha256(DATA / "independent-audit.json"),
        "cases": audit["rows_audited"],
        "bundles": audit["bundles_audited"],
        "users": audit["users_audited"],
        "records": records,
        "bucket_reports": buckets,
        "limits": [
            "Actual feedback observation timestamps are unavailable; timing proxies remain assumptions.",
            "Support counts do not establish statistical power or human model fidelity.",
            "Final model choices and case sampling must be frozen before final test scoring.",
        ],
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "bucket_reports"} | {
        "report": str(out / "report.json")
    }


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    # Existing protocol rate snapshot, conservative timeout reservation, not actual billing.
    amount = 32 * 300 * (2 * 1.31e-5 + 4 * 2.22e-6) + 3600 * (0.5 * 1.31e-5 + 2.22e-6) + 0.5
    reservation = BudgetLedger(ROOT / ".campaign/calendar-support-budget.sqlite", 2).reserve(
        "calendar-support-v1", amount
    )
    print(json.dumps(reservation), flush=True)
    call = run.spawn()
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": run.object_id}), flush=True
    )
    print(json.dumps(call.get(), indent=2), flush=True)
