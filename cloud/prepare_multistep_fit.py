"""Training/validation origins for direct multihorizon prediction heads."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-multistep-fit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "polars==1.31.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_training_horizons.py",
        remote_path="/research/tests/test_training_horizons.py",
    )
    .add_local_file(
        ROOT / "configs/human-multistep-fit-v1.json",
        remote_path="/research/multistep-config.json",
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def snapshot():
    import hashlib
    import json

    base = Path("/research") if Path("/research/multistep-config.json").exists() else ROOT
    config = base / (
        "multistep-config.json"
        if base == Path("/research")
        else "configs/human-multistep-fit-v1.json"
    )
    saved = {
        "prepare_multistep_fit.py": Path(__file__).read_text(),
        "config.json": config.read_text(),
    }
    for name in [
        "adjacent_calibration.py",
        "multistep_cohort.py",
        "training_horizons.py",
        "features.py",
        "calendar_splits.py",
        "intrinsic_context.py",
    ]:
        saved[name] = (base / "src/behavioral_equivalence" / name).read_text()
    saved["test_training_horizons.py"] = (base / "tests/test_training_horizons.py").read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=600,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def prepare_bucket(bucket: int):
    import json
    import time
    import numpy as np
    import polars as pl
    from behavioral_equivalence.adjacent_calibration import (
        representative_indices,
        query_history_tokens,
    )
    from behavioral_equivalence.multistep_cohort import recorded_horizon_targets
    from behavioral_equivalence.training_horizons import training_role_cohort
    from behavioral_equivalence.calendar_splits import (
        EvaluationCalendar,
        calendar_roles,
        ROLE_NAMES,
    )
    from behavioral_equivalence.intrinsic_context import duration_context

    if not isinstance(bucket, int) or not 0 <= bucket < 32:
        raise ValueError("Known prepared-data bucket required")
    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    config = json.loads(Path("/research/multistep-config.json").read_text())
    data = Path("/artifacts") / config["prepared_root"]
    if (
        checksum(data / "report.json") != config["prepared_report_sha256"]
        or checksum(data / "independent-audit.json") != config["independent_audit_sha256"]
    ):
        raise ValueError("Audited source dataset changed")
    source_report = json.loads((data / "report.json").read_text())["buckets"][bucket]
    source = data / "data" / f"bucket-{bucket:02d}"
    expected = {
        "interactions.parquet": source_report["interactions_sha256"],
        "timestamp-bundles.parquet": source_report["bundles_sha256"],
    }
    if any(checksum(source / name) != sha for name, sha in expected.items()):
        raise ValueError("Prepared bucket content differs from independent audit")
    metadata = Path("/artifacts") / config["metadata_root"]
    if checksum(metadata / "report.json") != config["metadata_report_sha256"]:
        raise ValueError("Intrinsic metadata report changed")
    meta_report = json.loads((metadata / "report.json").read_text())
    if any(
        checksum(metadata / name) != meta_report["output_sha256"][name]
        for name in ["duration-ms.npy", "upload-day.npy"]
    ):
        raise ValueError("Intrinsic metadata arrays changed")
    root = data / ("multistep-fit-" + digest[:16])
    out = root / f"bucket-{bucket:02d}"
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if (
            cached["source_digest"] != digest
            or cached["input_sha256"] != expected
            or any(checksum(out / name) != sha for name, sha in cached["output_sha256"].items())
        ):
            raise ValueError("Cached multistep cohort changed")
        return cached
    events = pl.read_parquet(
        source / "interactions.parquet",
        columns=["user_id", "time_ms", "tab", "video_id", "is_rand", "is_click"],
    ).with_row_index("physical_row")
    events = events.with_columns(
        (pl.col("physical_row").cast(pl.UInt64) + (bucket << 32)).alias("case_id")
    ).drop("physical_row")
    bundles = (
        pl.read_parquet(source / "timestamp-bundles.parquet")
        .sort(["user_id", "time_ms"])
        .with_row_index("bundle_index")
    )
    events = events.join(
        bundles.select(["user_id", "time_ms", "bundle_index"]),
        on=["user_id", "time_ms"],
        how="left",
        validate="m:1",
    ).sort(["bundle_index", "case_id"])
    if events["bundle_index"].null_count():
        raise ValueError("Exposure without an audited timestamp bundle")
    representatives = representative_indices(
        events["bundle_index"].to_numpy(), config["representative_seed_base"] + bucket
    )
    rep = {
        key: events[key].to_numpy()[representatives]
        for key in ["case_id", "user_id", "time_ms", "tab", "video_id", "is_rand", "is_click"]
    }
    del events
    users, times = bundles["user_id"].to_numpy(), bundles["time_ms"].to_numpy()
    if not np.array_equal(rep["user_id"], users) or not np.array_equal(rep["time_ms"], times):
        raise AssertionError("Representative does not match its source bundle")
    roles = calendar_roles(
        EvaluationCalendar(**config["calendar"]),
        times,
        bundles["available_sum_ms"].to_numpy(),
        bundles["user_split"].to_numpy(),
        "known_users",
        config["embargo_hours"] * 3600000,
    )
    user_role = (
        bundles["user_split"]
        .replace_strict({name: i for i, name in enumerate(ROLE_NAMES)})
        .to_numpy()
        .astype(np.int8)
    )
    duration = np.load(metadata / "duration-ms.npy", mmap_mode="r", allow_pickle=False)
    upload = np.load(metadata / "upload-day.npy", mmap_mode="r", allow_pickle=False)
    outputs, support = {}, []
    out.mkdir(parents=True, exist_ok=True)
    for split, role in [("train", 0), ("validation", 1)]:
        queries, cohorts = training_role_cohort(
            users,
            times,
            roles,
            role,
            config["decision_horizons"],
            config["origin_cap_per_user"],
            config["origin_seed"],
        )
        targets = [recorded_horizon_targets(rep["is_click"], c) for c in cohorts]
        tokens = query_history_tokens(
            users,
            times,
            bundles["available_sum_ms"].to_numpy(),
            bundles["clicks"].to_numpy(),
            bundles["exposures"].to_numpy(),
            queries,
            config["embargo_hours"] * 3600000,
        )
        context, known = duration_context(
            duration[rep["video_id"][queries]], upload[rep["video_id"][queries]], times[queries]
        )
        arrays = {
            key: rep[name][queries]
            for key, name in {
                "case_id": "case_id",
                "user_id": "user_id",
                "time_ms": "time_ms",
                "tab": "tab",
                "video_id": "video_id",
                "is_random": "is_rand",
                "target": "is_click",
            }.items()
        }
        arrays.update(
            tokens=tokens,
            intrinsic_context=context,
            bundle_index=queries,
            user_role_code=user_role[queries],
            role_code=roles[queries],
            decision_horizons=np.asarray(config["decision_horizons"], dtype=np.int16),
            horizon_observed=np.column_stack([c["observed_mask"] for c in cohorts]),
            horizon_end_bundle=np.column_stack([c["end_indices"] for c in cohorts]),
            horizon_positive_count=np.column_stack([t["positive_count"] for t in targets]).astype(
                np.int16
            ),
            horizon_last_positive=np.column_stack([t["last_positive"] for t in targets]).astype(
                np.int8
            ),
        )
        if not np.all(roles[queries] == role):
            raise AssertionError("Origin crossed its actual fitting/validation role")
        name = split + "-cohort.npz"
        np.savez_compressed(out / name, **arrays)
        outputs[name] = checksum(out / name)
        for protocol in config["protocols"]:
            selected = (
                np.ones(len(queries), dtype=bool)
                if protocol == "known_users"
                else user_role[queries] == role
            )
            support.append(
                dict(
                    protocol=protocol,
                    split=split,
                    origins=int(selected.sum()),
                    users=int(len(np.unique(users[queries[selected]]))),
                    known_duration_origins=int((selected & known).sum()),
                    horizons=[
                        dict(
                            horizon=c["horizon"],
                            observed_origins=int((selected & c["observed_mask"]).sum()),
                            observed_users=int(
                                len(np.unique(users[queries[selected & c["observed_mask"]]]))
                            ),
                            censored_origins=int((selected & ~c["observed_mask"]).sum()),
                        )
                        for c in cohorts
                    ],
                )
            )
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    result = dict(
        status="complete",
        source_digest=digest,
        bucket=bucket,
        input_sha256=expected,
        output_sha256=outputs,
        complete_source_bundles=len(bundles),
        support=support,
        wall_seconds=time.monotonic() - started,
        scope=config["scope"],
        final_test_predictions_generated=0,
        final_test_outcome_targets_exported=0,
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
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
def run(smoke):
    import json
    import subprocess

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    root = Path("/artifacts") / config["prepared_root"] / ("multistep-fit-" + digest[:16])
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    if smoke:
        subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-W",
                "error",
                "/research/tests/test_training_horizons.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    elif json.loads((root / "smoke.json").read_text())["status"] != "complete":
        raise ValueError("Passing same-source smoke required")
    volume.commit()
    results = list(
        prepare_bucket.map([0] if smoke else range(32), order_outputs=True, return_exceptions=True)
    )
    successes = [x for x in results if not isinstance(x, Exception)]
    failures = [repr(x) for x in results if isinstance(x, Exception)]
    report = dict(
        status="incomplete" if failures else "complete",
        phase="smoke" if smoke else "full",
        source_digest=digest,
        config=config,
        buckets=successes,
        failures=failures,
        support=[
            dict(
                protocol=p,
                split=split,
                origins=sum(
                    next(
                        s["origins"]
                        for s in row["support"]
                        if s["protocol"] == p and s["split"] == split
                    )
                    for row in successes
                ),
                users=sum(
                    next(
                        s["users"]
                        for s in row["support"]
                        if s["protocol"] == p and s["split"] == split
                    )
                    for row in successes
                ),
            )
            for p in config["protocols"]
            for split in ["train", "validation"]
        ],
        root=str(root),
        final_test_predictions_generated=0,
        scope=config["scope"],
    )
    (root / ("smoke.json" if smoke else "report.json")).write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k != "buckets"}


@app.local_entrypoint()
def main(smoke: bool = True):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    digest = snapshot()[1]
    phase = "smoke" if smoke else "full"
    amount = (
        (1 if smoke else 32) * 600 * (2 * 1.31e-5 + 8 * 2.22e-6)
        + 3600 * (0.5 * 1.31e-5 + 2.22e-6)
        + 0.25
    )
    reservation = BudgetLedger(ROOT / ".campaign/human-cases-budget.sqlite", 4).reserve(
        "multistep-fit-" + digest + "-" + phase, amount
    )
    state = ROOT / ".campaign" / ("multistep-fit-" + digest[:20] + "-" + phase + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = run.spawn(smoke)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, source_digest=digest, reservation=reservation
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
