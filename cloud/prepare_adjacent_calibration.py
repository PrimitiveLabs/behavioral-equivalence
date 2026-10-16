"""Private adjacent-bundle calibration inputs, independent of fitted model seeds."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-adjacent-calibration-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "polars==1.31.0")
    .env({"PYTHONPATH": "/research/src", "POLARS_MAX_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-adjacent-calibration-v1.json",
        remote_path="/research/adjacent-config.json",
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

    base = Path("/research") if Path("/research/adjacent-config.json").exists() else ROOT
    config = base / (
        "adjacent-config.json"
        if base == Path("/research")
        else "configs/human-adjacent-calibration-v1.json"
    )
    saved = {
        "prepare_adjacent_calibration.py": Path(__file__).read_text(),
        "config.json": config.read_text(),
    }
    for name in [
        "adjacent_calibration.py",
        "features.py",
        "calendar_splits.py",
        "intrinsic_context.py",
    ]:
        saved[name] = (base / "src/behavioral_equivalence" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=900,
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
        adjacent_origins,
        query_history_tokens,
    )
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
    config = json.loads(Path("/research/adjacent-config.json").read_text())
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
    root = data / ("adjacent-calibration-" + digest[:16])
    out = root / f"bucket-{bucket:02d}"
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if (
            cached["source_digest"] != digest
            or cached["input_sha256"] != expected
            or checksum(out / "calibration.npz") != cached["output_sha256"]["calibration.npz"]
        ):
            raise ValueError("Cached adjacent calibration changed")
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
    origins = adjacent_origins(
        users, times, roles, config["origin_cap_per_user"], config["origin_seed"]
    )
    queries = np.unique(np.r_[origins, origins + 1])
    tokens = query_history_tokens(
        users,
        times,
        bundles["available_sum_ms"].to_numpy(),
        bundles["clicks"].to_numpy(),
        bundles["exposures"].to_numpy(),
        queries,
        config["embargo_hours"] * 3600000,
    )
    duration = np.load(metadata / "duration-ms.npy", mmap_mode="r", allow_pickle=False)
    upload = np.load(metadata / "upload-day.npy", mmap_mode="r", allow_pickle=False)
    context, known = duration_context(
        duration[rep["video_id"][queries]], upload[rep["video_id"][queries]], times[queries]
    )
    user_role = (
        bundles["user_split"]
        .replace_strict({name: i for i, name in enumerate(ROLE_NAMES)})
        .to_numpy()
        .astype(np.int8)
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
        pair_current=np.searchsorted(queries, origins),
        pair_next=np.searchsorted(queries, origins + 1),
        pair_user_id=users[origins],
        pair_user_role_code=user_role[origins],
    )
    if not np.all(roles[queries] == 2) or not np.all(users[origins] == users[origins + 1]):
        raise AssertionError("A transition crossed a user or calibration boundary")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "calibration.npz", **arrays)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    support = []
    for protocol in ["known_users", "unseen_users"]:
        selected = (
            np.ones(len(origins), dtype=bool)
            if protocol == "known_users"
            else user_role[origins] == 2
        )
        support.append(
            dict(
                protocol=protocol,
                transitions=int(selected.sum()),
                users=int(len(np.unique(users[origins[selected]]))),
            )
        )
    result = dict(
        status="complete",
        source_digest=digest,
        bucket=bucket,
        input_sha256=expected,
        output_sha256={"calibration.npz": checksum(out / "calibration.npz")},
        complete_source_bundles=len(bundles),
        query_exposures=len(queries),
        known_duration_queries=int(known.sum()),
        support=support,
        wall_seconds=time.monotonic() - started,
        scope="Actual adjacent timestamp bundles with outcome-independent representative exposures and origin sampling. Complete histories enforce feedback availability. Point-model calibration only; no human confidence guarantee or test prediction.",
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=10800,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run(smoke: bool):
    import json

    saved, digest = snapshot()
    config = json.loads(Path("/research/adjacent-config.json").read_text())
    root = Path("/artifacts") / config["prepared_root"] / ("adjacent-calibration-" + digest[:16])
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    volume.commit()
    results = list(
        prepare_bucket.map([0] if smoke else range(32), order_outputs=True, return_exceptions=True)
    )
    failures = [repr(r) for r in results if isinstance(r, Exception)]
    successful = [r for r in results if not isinstance(r, Exception)]
    report = dict(
        status="incomplete" if failures else "complete",
        phase="single-bucket smoke" if smoke else "full adjacent calibration preparation",
        source_digest=digest,
        config=config,
        buckets=successful,
        failures=failures,
        support=[
            dict(
                protocol=p,
                transitions=sum(
                    next(s["transitions"] for s in r["support"] if s["protocol"] == p)
                    for r in successful
                ),
                users=sum(
                    next(s["users"] for s in r["support"] if s["protocol"] == p) for r in successful
                ),
            )
            for p in ["known_users", "unseen_users"]
        ],
        root=str(root),
        test_predictive_performance="not evaluated",
    )
    (root / ("smoke.json" if smoke else "report.json")).write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(smoke: bool = True):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = snapshot()
    phase = "smoke" if smoke else "full"
    reservation = (
        (1 if smoke else 32) * 900 * (2 * 1.31e-5 + 8 * 2.22e-6)
        + 10800 * (0.5 * 1.31e-5 + 2.22e-6)
        + 0.25
    )
    reserved = BudgetLedger(ROOT / ".campaign/adjacent-calibration-budget.sqlite", 3).reserve(
        digest + "-" + phase, reservation
    )
    state = ROOT / ".campaign" / ("adjacent-calibration-" + digest[:12] + "-" + phase + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = run.spawn(smoke)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    function_id=run.object_id,
                    source_digest=digest,
                    reservation=reserved,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
