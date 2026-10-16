"""Audit standard/random sampled duration context and write private sidecars."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/artifacts/datasets/kuairand-27k")
CASES = BASE / "prepared-candidate-v2/confirmatory-cases-d05a64f50a264877"
META = BASE / "item-metadata-v1-14656b2136c2c3d0"
app = modal.App("behavioral-equivalence-case-context-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pyarrow==20.0.0")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def snapshot():
    import json
    import hashlib

    saved = {
        "audit_case_context.py": Path(__file__).read_text(),
        "intrinsic_context.py": Path(
            "/research/src/behavioral_equivalence/intrinsic_context.py"
        ).read_text(),
    }
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=300,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_bucket(bucket: int):
    import json
    import numpy as np
    import pyarrow.parquet as pq
    from behavioral_equivalence.intrinsic_context import duration_context

    volume.reload()
    _, digest = snapshot()
    source = json.loads((CASES / "report.json").read_text())
    if not 0 <= bucket < 32 or source["status"] != "complete":
        raise ValueError("Valid bucket and completed cases required")
    record = source["buckets"][bucket]
    metadata = json.loads((META / "report.json").read_text())
    for name in ["duration-ms.npy", "upload-day.npy"]:
        if sha256(META / name) != metadata["output_sha256"][name]:
            raise ValueError("Intrinsic metadata changed")
    duration = np.load(META / "duration-ms.npy", mmap_mode="r", allow_pickle=False)
    upload = np.load(META / "upload-day.npy", mmap_mode="r", allow_pickle=False)
    raw = BASE / "prepared-candidate-v2/data" / f"bucket-{bucket:02d}" / "interactions.parquet"
    if sha256(raw) != record["input_sha256"]["interactions.parquet"]:
        raise ValueError("Physical source changed")
    log_duration = (
        pq.read_table(raw, columns=["duration_ms"])["duration_ms"].combine_chunks().to_numpy()
    )
    out = CASES / ("intrinsic-context-" + digest[:16]) / f"bucket-{bucket:02d}"
    out.mkdir(parents=True, exist_ok=True)
    records, outputs = [], {}
    for hours in [0, 24, 48]:
        name = f"embargo-{hours:02d}h.npz"
        path = CASES / f"bucket-{bucket:02d}" / name
        if sha256(path) != record["output_sha256"][name]:
            raise ValueError("Case source changed")
        with np.load(path, allow_pickle=False) as saved:
            ids, item, time = saved["case_id"], saved["video_id"], saved["time_ms"]
            user, role, user_role, random = (
                saved[k] for k in ["user_id", "role_code", "user_role_code", "is_random"]
            )
        rows = (ids & np.uint64(2**32 - 1)).astype(np.int64)
        if np.any(item < 0) or np.any(item >= len(duration)) or np.any(rows >= len(log_duration)):
            raise ValueError("Item or source-row context lookup outside audited bounds")
        values = duration[item]
        features, known = duration_context(values, upload[item], time)
        np.savez_compressed(out / name, case_id=ids, context=features)
        outputs[name] = sha256(out / name)
        comparable = np.isfinite(values) & (values > 0) & (log_duration[rows] > 0)
        mismatch = comparable & (values != log_duration[rows])
        for protocol in ["known_users", "unseen_users"]:
            for r, role_name in enumerate(["train", "validation", "calibration", "test"]):
                selected = role == r
                if protocol == "unseen_users":
                    selected &= user_role == r
                for is_random in [0, 1]:
                    chosen = selected & (random == is_random)
                    records.append(
                        {
                            "embargo_hours": hours,
                            "protocol": protocol,
                            "role": role_name,
                            "random_exposure": is_random,
                            "cases": int(chosen.sum()),
                            "users": int(len(np.unique(user[chosen]))),
                            "usable_prior_date_context": int((chosen & known).sum()),
                            "comparable_durations": int((chosen & comparable).sum()),
                            "exact_duration_mismatches": int((chosen & mismatch).sum()),
                        }
                    )
    report = {
        "status": "complete",
        "bucket": bucket,
        "source_digest": digest,
        "records": records,
        "output_sha256": outputs,
        "case_report_sha256": sha256(CASES / "report.json"),
        "metadata_report_sha256": sha256(META / "report.json"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=3600,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run():
    import json

    saved, digest = snapshot()
    out = CASES / ("intrinsic-context-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    volume.commit()
    buckets = list(audit_bucket.map(range(32), return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in buckets if isinstance(r, Exception)]
    if failures:
        raise RuntimeError("Intrinsic context audit failed: " + str(failures))
    records = []
    for i in range(len(buckets[0]["records"])):
        parts = [b["records"][i] for b in buckets]
        identity = {
            k: parts[0][k] for k in ["embargo_hours", "protocol", "role", "random_exposure"]
        }
        identity.update(
            {
                k: sum(p[k] for p in parts)
                for k in [
                    "cases",
                    "users",
                    "usable_prior_date_context",
                    "comparable_durations",
                    "exact_duration_mismatches",
                ]
            }
        )
        records.append(identity)
    report = {
        "status": "complete",
        "source_digest": digest,
        "records": records,
        "buckets": buckets,
        "case_report_sha256": sha256(CASES / "report.json"),
        "metadata_report_sha256": sha256(META / "report.json"),
        "feature_policy": "log1p(positive intrinsic duration in seconds) plus missing flag; require a known strictly prior UTC upload day; missing values map to zero plus flag one",
        "limits": [
            "Metadata consistency is not full online provenance. Treat intrinsic-feature availability as an explicit assumption.",
            "No current response or model prediction is used in this audit.",
            "Timing-policy case sets overlap; counts are not independent trials.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k != "buckets"} | {
        "report": str(out / "report.json")
    }


@app.local_entrypoint()
def main():
    import json

    call = run.spawn()
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": run.object_id}), flush=True
    )
    result = call.get()
    print(json.dumps(result, indent=2), flush=True)
