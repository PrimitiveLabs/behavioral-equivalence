"""Recover ordered binary futures on private storage, bound to audited cohorts."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-trajectory-sequences-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "polars==1.31.0", "pydantic==2.11.7")
    .env(
        {
            "PYTHONPATH": "/research/tools:/research/src",
            "POLARS_MAX_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tools/trajectory_targets.py", remote_path="/research/tools/trajectory_targets.py"
    )
    .add_local_file(
        ROOT / "configs/human-multistep-cohort-v1.json", remote_path="/research/cohort.json"
    )
    .add_local_file(
        ROOT / "configs/human-final-horizon-v1.json", remote_path="/research/horizon.json"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def identity():
    import hashlib
    import json

    base = Path("/research") if Path("/research/cohort.json").exists() else ROOT
    names = {
        "cloud.py": Path(__file__).read_text(),
        "targets.py": (base / ("tools/trajectory_targets.py")).read_text(),
        "cohort.json": (
            base / ("cohort.json" if base != ROOT else "configs/human-multistep-cohort-v1.json")
        ).read_text(),
        "horizon.json": (
            base / ("horizon.json" if base != ROOT else "configs/human-final-horizon-v1.json")
        ).read_text(),
    }
    for name in ["adjacent_calibration.py", "calendar_splits.py", "features.py"]:
        names[name] = (base / "src/behavioral_equivalence" / name).read_text()
    return names, hashlib.sha256(json.dumps(names, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def prepare():
    import json
    import time
    import numpy as np
    import polars as pl
    from behavioral_equivalence.adjacent_calibration import representative_indices
    from behavioral_equivalence.calendar_splits import EvaluationCalendar, calendar_roles
    from trajectory_targets import reconstruct_sequences

    volume.reload()
    started = time.monotonic()
    source, digest = identity()
    cfg, hc = json.loads(source["cohort.json"]), json.loads(source["horizon.json"])
    data = Path("/artifacts") / cfg["prepared_root"]
    cohort = Path("/artifacts") / hc["cohort_root"]
    for path, sha in [
        (data / "report.json", cfg["prepared_report_sha256"]),
        (cohort / "report.json", hc["cohort_report_sha256"]),
        (cohort / hc["audit_relative"], hc["audit_report_sha256"]),
    ]:
        if checksum(path) != sha:
            raise ValueError("Audited input manifest changed")
    parent = json.loads((cohort / "report.json").read_text())
    prepared = json.loads((data / "report.json").read_text())
    if len(parent["buckets"]) != 32 or len(prepared["buckets"]) != 32:
        raise ValueError("Complete 32-bucket sources required")
    out = Path("/artifacts") / ("trajectory-sequences-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(source, sort_keys=True))
    records = []
    for bucket in range(32):
        prefix = data / "data" / f"bucket-{bucket:02d}"
        expected = prepared["buckets"][bucket]
        for name, key in [
            ("interactions.parquet", "interactions_sha256"),
            ("timestamp-bundles.parquet", "bundles_sha256"),
        ]:
            if checksum(prefix / name) != expected[key]:
                raise ValueError("Prepared source bucket changed")
        packed = cohort / f"bucket-{bucket:02d}" / "test-cohort.npz"
        if checksum(packed) != parent["buckets"][bucket]["output_sha256"]["test-cohort.npz"]:
            raise ValueError("Audited origin cohort changed")
        target = out / f"bucket-{bucket:02d}.npz"
        events = pl.read_parquet(
            prefix / "interactions.parquet", columns=["user_id", "time_ms", "is_click"]
        ).with_row_index("physical_row")
        bundles = (
            pl.read_parquet(prefix / "timestamp-bundles.parquet")
            .sort(["user_id", "time_ms"])
            .with_row_index("bundle_index")
        )
        events = events.join(
            bundles.select("user_id", "time_ms", "bundle_index"),
            on=["user_id", "time_ms"],
            how="left",
            validate="m:1",
        ).sort(["bundle_index", "physical_row"])
        if events["bundle_index"].null_count():
            raise ValueError("Source exposure missing bundle")
        selected = representative_indices(
            events["bundle_index"].to_numpy(), cfg["representative_seed_base"] + bucket
        )
        response = events["is_click"].to_numpy()[selected]
        physical = events["physical_row"].to_numpy()[selected].astype(np.uint64) + (bucket << 32)
        users, times = bundles["user_id"].to_numpy(), bundles["time_ms"].to_numpy()
        if not np.array_equal(events["user_id"].to_numpy()[selected], users) or not np.array_equal(
            events["time_ms"].to_numpy()[selected], times
        ):
            raise ValueError("Representative ordering differs from audited bundles")
        roles = calendar_roles(
            EvaluationCalendar(**cfg["calendar"]),
            times,
            bundles["available_sum_ms"].to_numpy(),
            bundles["user_split"].to_numpy(),
            "known_users",
            cfg["embargo_hours"] * 3600000,
        )
        with np.load(packed, allow_pickle=False) as a:
            origins = a["bundle_index"]
            if not np.array_equal(a["case_id"], physical[origins]) or not np.array_equal(
                a["user_id"], users[origins]
            ):
                raise ValueError("Original representative identity differs from stored origins")
            if not np.array_equal(a["decision_horizons"], cfg["decision_horizons"]):
                raise ValueError("Horizon order changed")
            sequence = reconstruct_sequences(
                users,
                roles,
                response,
                origins,
                a["horizon_observed"],
                a["horizon_positive_count"],
                a["horizon_last_positive"],
                cfg["decision_horizons"],
            )
            np.savez_compressed(
                target,
                case_id=a["case_id"],
                user_id=a["user_id"],
                user_role_code=a["user_role_code"],
                outcomes=sequence,
                observed=a["horizon_observed"],
            )
            n = len(origins)
        records.append(
            dict(
                bucket=bucket,
                origins=n,
                source_cohort_sha256=checksum(packed),
                file=target.name,
                sha256=checksum(target),
            )
        )
        print(json.dumps(dict(bucket=bucket, origins=n)), flush=True)
        del events, bundles, sequence
    report = dict(
        status="complete",
        source_digest=digest,
        artifact_root=str(out),
        buckets=records,
        origins=sum(r["origins"] for r in records),
        wall_seconds=time.monotonic() - started,
        scope="Source-bound representative sequence recovery and independent per-origin support/count/last checks. No predictive scoring or causal inference.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import fcntl
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = identity()
    lock = (ROOT / ".campaign/trajectory-sequences.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reservation = BudgetLedger(ROOT / ".campaign/calibrated-parent-budget.sqlite", 25).reserve(
        "trajectory-sequences-" + digest, 0.30
    )
    path = ROOT / ".campaign" / ("trajectory-sequences-" + digest[:16] + ".json")
    if path.exists():
        call = modal.FunctionCall.from_id(json.loads(path.read_text())["function_call_id"])
    else:
        call = prepare.spawn()
        path.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, source_digest=digest, reservation=reservation
                ),
                indent=2,
            )
        )
    print(path.read_text(), flush=True)
    result = call.get()
    (ROOT / "reports/preflight" / ("trajectory-sequences-" + digest[:16] + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)
