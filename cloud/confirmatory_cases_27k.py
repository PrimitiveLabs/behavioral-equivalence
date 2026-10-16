"""Private bounded human case artifacts, without fitting or final test scoring."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/artifacts/datasets/kuairand-27k/prepared-candidate-v2")
app = modal.App("behavioral-equivalence-confirmatory-cases-27k-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "polars==1.31.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-case-construction-v1.json", remote_path="/research/case-config.json"
    )
    .add_local_file(
        ROOT / "tests/test_confirmatory_preparation.py",
        remote_path="/research/tests/test_confirmatory_preparation.py",
    )
)


def snapshot() -> tuple[dict, str]:
    import hashlib
    import json

    sources = {
        "confirmatory_cases_27k.py": Path(__file__).read_text(),
        "human-case-construction-v1.json": Path("/research/case-config.json").read_text(),
        "test_confirmatory_preparation.py": Path(
            "/research/tests/test_confirmatory_preparation.py"
        ).read_text(),
    }
    for name in ["calendar_splits.py", "confirmatory_preparation.py", "features.py"]:
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
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def smoke() -> dict:
    import json
    import subprocess

    sources, digest = snapshot()
    check = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_confirmatory_preparation.py",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = {
        "status": "complete" if check.returncode == 0 else "failed",
        "source_digest": digest,
        "returncode": check.returncode,
        "stdout": check.stdout,
        "stderr": check.stderr,
    }
    out = DATA / ("confirmatory-cases-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "smoke.json").write_text(json.dumps(report, indent=2))
    (out / "source.json").write_text(json.dumps(sources, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=900,
    retries=0,
    max_containers=4,
    volumes={"/artifacts": volume},
)
def prepare_bucket(bucket: int) -> dict:
    import json
    import time
    import numpy as np
    import polars as pl
    from behavioral_equivalence.confirmatory_preparation import prepare_sample
    from behavioral_equivalence.calendar_splits import EvaluationCalendar, ROLE_NAMES

    if not 0 <= bucket < 32:
        raise ValueError("Valid bucket required")
    started = time.monotonic()
    _, digest = snapshot()
    config = json.loads(Path("/research/case-config.json").read_text())
    out = DATA / ("confirmatory-cases-" + digest[:16]) / f"bucket-{bucket:02d}"
    source = DATA / "data" / f"bucket-{bucket:02d}"
    report = json.loads((source / "report.json").read_text())
    expected = {
        "interactions.parquet": report["interactions_sha256"],
        "timestamp-bundles.parquet": report["bundles_sha256"],
    }
    if any(sha256(source / name) != checksum for name, checksum in expected.items()):
        raise ValueError("Prepared bucket differs from independently audited hashes")
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if cached["source_digest"] != digest or cached["input_sha256"] != expected:
            raise ValueError("Cached case source changed")
        if any(
            sha256(out / name) != checksum for name, checksum in cached["output_sha256"].items()
        ):
            raise ValueError("Cached case artifact changed")
        return cached
    events = (
        pl.read_parquet(
            source / "interactions.parquet",
            columns=["user_id", "time_ms", "tab", "video_id", "is_rand", "is_click"],
        )
        .with_row_index("row_id")
        .with_columns((pl.col("row_id").cast(pl.UInt64) + (bucket << 32)).alias("case_id"))
        .drop("row_id")
    )
    if len(events) >= 2**32:
        raise ValueError("Case ordinal exceeds bucket namespace")
    bundles = pl.read_parquet(source / "timestamp-bundles.parquet")
    out.mkdir(parents=True, exist_ok=True)
    outputs, support = {}, []
    for hours in config["embargo_hours"]:
        frame, tokens = prepare_sample(
            events,
            bundles,
            EvaluationCalendar(**config["calendar"]),
            hours * 3600 * 1000,
            config["per_user_role_caps"],
            config["sampling_seed"],
        )
        user_codes = (
            frame["user_split"]
            .replace_strict({name: i for i, name in enumerate(ROLE_NAMES)})
            .to_numpy()
            .astype(np.int8)
        )
        role_codes = frame["role_code"].to_numpy()
        arrays = {
            name: frame[column].to_numpy()
            for name, column in {
                "case_id": "case_id",
                "user_id": "user_id",
                "time_ms": "time_ms",
                "available_sum_ms": "available_sum_ms",
                "tab": "tab",
                "video_id": "video_id",
                "is_random": "is_rand",
                "target": "is_click",
                "bundle_index": "feature_id",
            }.items()
        }
        arrays.update(tokens=tokens, role_code=role_codes, user_role_code=user_codes)
        name = f"embargo-{hours:02d}h.npz"
        np.savez_compressed(out / name, **arrays)
        outputs[name] = sha256(out / name)
        for protocol in ["known_users", "unseen_users"]:
            for r, role in enumerate(ROLE_NAMES):
                selected = role_codes == r
                if protocol == "unseen_users":
                    selected &= user_codes == r
                support.append(
                    {
                        "protocol": protocol,
                        "embargo_hours": hours,
                        "role": role,
                        "cases": int(selected.sum()),
                        "users": int(len(np.unique(arrays["user_id"][selected]))),
                    }
                )
        volume.commit()
    result = {
        "status": "complete",
        "bucket": bucket,
        "source_digest": digest,
        "input_sha256": expected,
        "output_sha256": outputs,
        "support": support,
        "wall_seconds": time.monotonic() - started,
        "test_predictive_performance": "not evaluated",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=10800,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def prepare_all() -> dict:
    import json

    _, digest = snapshot()
    config = json.loads(Path("/research/case-config.json").read_text())
    out = DATA / ("confirmatory-cases-" + digest[:16])
    if json.loads((out / "smoke.json").read_text())["status"] != "complete":
        raise ValueError("Passing source-bound cloud integration tests required")
    if (
        sha256(DATA / "report.json") != config["prepared_report_sha256"]
        or sha256(DATA / "independent-audit.json") != config["independent_audit_sha256"]
    ):
        raise ValueError("Candidate dataset or independent audit changed")
    records = list(prepare_bucket.map(range(32), return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in records if isinstance(r, Exception)]
    if failures:
        raise RuntimeError(
            "Case preparation failed; successful buckets remain cached: " + str(failures)
        )
    support = []
    for i in range(len(records[0]["support"])):
        parts = [r["support"][i] for r in records]
        combined = {key: parts[0][key] for key in ["protocol", "embargo_hours", "role"]}
        combined.update({key: sum(p[key] for p in parts) for key in ["cases", "users"]})
        support.append(combined)
    result = {
        "status": "complete",
        "phase": "private final-candidate case construction, not predictive scoring",
        "source_digest": digest,
        "config": config,
        "support": support,
        "buckets": records,
        "limits": [
            "Feedback availability remains a declared proxy assumption.",
            "Model and analysis preregistration and independent sampled-case audit remain required.",
            "Case targets and histories are private; no test predictive scores have been inspected.",
        ],
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

    if stage not in {"smoke", "prepare"}:
        raise ValueError("Stage must be smoke or prepare")
    if stage == "prepare":
        sys.path.insert(0, str(ROOT / "src"))
        from behavioral_equivalence.budget import BudgetLedger

        amount = 32 * 900 * (2 * 1.31e-5 + 8 * 2.22e-6) + 10800 * (0.5 * 1.31e-5 + 2.22e-6) + 1
        print(
            json.dumps(
                BudgetLedger(ROOT / ".campaign/human-cases-budget.sqlite", 4).reserve(
                    "human-cases-v1", amount
                )
            ),
            flush=True,
        )
    function = smoke if stage == "smoke" else prepare_all
    call = function.spawn()
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": function.object_id}),
        flush=True,
    )
    result = call.get()
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("Source-bound case preparation check failed")
