"""Fit fixed direct horizon comparators on audited origin-only features."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-horizon-heads-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1", "pytest==8.4.1"
    )
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(
        ROOT / "configs/human-horizon-heads-v1.json", remote_path="/research/head-spec.json"
    )
    .add_local_file(
        ROOT / "configs/human-horizon-training-v1.json", remote_path="/research/config.json"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(8 * 1024**2):
            h.update(b)
    return h.hexdigest()


def snapshot():
    import json
    import hashlib

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/human-horizon-training-v1.json"
    )
    head = (
        base / "head-spec.json"
        if base == Path("/research")
        else base / "configs/human-horizon-heads-v1.json"
    )
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": config.read_text(),
        "head-spec.json": head.read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in ["test_horizon_heads.py", "test_horizon_targets.py"]:
        saved["tests/" + name] = (base / "tests" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def context():
    import json
    import hashlib

    saved, digest = snapshot()
    cfg = json.loads(saved["config.json"])
    spec = json.loads(saved["head-spec.json"])
    if hashlib.sha256(saved["head-spec.json"].encode()).hexdigest() != cfg["head_spec_sha256"]:
        raise ValueError("Fixed head specification changed")
    data = Path("/artifacts") / cfg["cohort_root"]
    if (
        checksum(data / "report.json") != cfg["cohort_report_sha256"]
        or checksum(data / cfg["audit_relative"]) != cfg["audit_report_sha256"]
    ):
        raise ValueError("Audited fitting cohort changed")
    report = json.loads((data / "report.json").read_text())
    audit = json.loads((data / cfg["audit_relative"]).read_text())
    if (
        report["status"] != "complete"
        or audit["status"] != "complete"
        or len(report["buckets"]) != 32
        or audit["bucket_count"] != 32
    ):
        raise ValueError("All audited input buckets required")
    return saved, digest, cfg, spec, data, report


def parent_path(cfg, protocol, architecture, seed):
    source = next(x for x in cfg["training_sources"] if seed in x["seeds"])
    return Path("/artifacts") / source["root"] / "fits" / f"{protocol}-{architecture}-{seed}"


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=900,
    max_containers=2,
    retries=0,
    volumes={"/artifacts": volume},
)
def fit(protocol, architecture, seed):
    import json
    import hashlib
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.item_context import encode_items
    from behavioral_equivalence.content_scaling import scale_duration
    from behavioral_equivalence.horizon_targets import horizon_event_targets
    from behavioral_equivalence.horizon_heads import fit_horizon_heads, predict_horizon_heads
    from behavioral_equivalence.streaming_scores import metric_block, merge_metric_blocks

    volume.reload()
    started = time.monotonic()
    saved, digest, cfg, spec, data, cohort = context()
    if (
        protocol not in spec["protocols"]
        or architecture not in spec["architectures"]
        or seed not in spec["seeds"]
    ):
        raise ValueError("Fixed30fit matrix required")
    parent = parent_path(cfg, protocol, architecture, seed)
    parent_sha = checksum(parent / "report.json")
    pr = json.loads((parent / "report.json").read_text())
    source = next(x for x in cfg["training_sources"] if seed in x["seeds"])
    root = Path("/artifacts") / source["root"]
    original = json.loads((root / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        != source["source_digest"]
        or pr["source_digest"] != source["source_digest"]
        or pr["status"] != "complete"
        or pr["protocol"] != protocol
        or pr["architecture"] != architecture
        or pr["seed"] != seed
    ):
        raise ValueError("Exact completed frozen parent encoder required")
    if any(checksum(parent / name) != sha for name, sha in pr["output_sha256"].items()):
        raise ValueError("Parent encoder artifacts changed")
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "item_context.py",
        "content_scaling.py",
    ]:
        key = "src/behavioral_equivalence/" + name
        if original[key] != saved[key]:
            raise ValueError("Frozen encoder/preprocessing implementation changed")
    training = next(
        json.loads(v)
        for k, v in original.items()
        if k.startswith("configs/human-training-calendar")
    )
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    if (
        checksum(inputs / "report.json") != pr["input_report_sha256"]
        or pr["input_report_sha256"] != training["input_artifacts"][protocol]["report_sha256"]
    ):
        raise ValueError("Training-only preprocessing report changed")
    ir = json.loads((inputs / "report.json").read_text())
    for name in ["item-vocabulary.npy", "duration-scaling.json"]:
        if checksum(inputs / name) != ir["output_sha256"][name]:
            raise ValueError("Preprocessing arrays changed")
    vocab = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    out = (
        Path("/artifacts")
        / ("human-horizon-heads-" + digest[:16])
        / f"{protocol}-{architecture}-{seed}"
    )
    if (out / "report.json").exists():
        old = json.loads((out / "report.json").read_text())
        if (
            old["parent_fit_report_sha256"] != parent_sha
            or old["source_digest"] != digest
            or any(checksum(out / name) != sha for name, sha in old["output_sha256"].items())
        ):
            raise ValueError("Cached horizon model changed")
        return {k: v for k, v in old.items() if k != "validation_metrics"}
    torch.set_num_threads(8)
    model = build_content_model(architecture, ir["vocabulary_size"], training["hidden"])
    model.load_state_dict(torch.load(parent / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()

    def encode(a, indices):
        items = encode_items(vocab, a["video_id"][indices])
        ctx = scale_duration(a["intrinsic_context"][indices], scaling)
        pieces = []
        for offset in range(0, len(indices), cfg["inference_batch_size"]):
            sl = slice(offset, offset + cfg["inference_batch_size"])
            ix = indices[sl]
            args = [
                torch.as_tensor(a["tokens"][ix], dtype=torch.long),
                torch.as_tensor(a["tab"][ix], dtype=torch.long),
                torch.as_tensor(items[sl], dtype=torch.long),
                torch.as_tensor(ctx[sl], dtype=torch.float32),
            ]
            with torch.inference_mode():
                pieces.append(
                    model.output[:-1](model.representation(*args)).numpy().astype(np.float64)
                )
        return np.concatenate(pieces) if pieces else np.empty((0, training["hidden"]))

    total = next(
        x["origins"]
        for x in cohort["support"]
        if x["protocol"] == protocol and x["split"] == "train"
    )
    chosen = np.sort(
        np.random.default_rng(cfg["origin_sampling_seed"]).choice(
            total, min(total, spec["maximum_training_origins"]), replace=False
        )
    )
    position = 0
    pieces = []
    ids_sha = hashlib.sha256()
    for bucket in cohort["buckets"]:
        path = data / f"bucket-{bucket['bucket']:02d}" / "train-cohort.npz"
        if checksum(path) != bucket["output_sha256"]["train-cohort.npz"]:
            raise ValueError("Training-origin bucket changed")
        with np.load(path, allow_pickle=False) as a:
            eligible = np.flatnonzero(
                (a["role_code"] == 0)
                & ((a["user_role_code"] == 0) if protocol == "unseen_users" else True)
            )
            local = (
                chosen[
                    np.searchsorted(chosen, position) : np.searchsorted(
                        chosen, position + len(eligible)
                    )
                ]
                - position
            )
            position += len(eligible)
            ix = eligible[local]
            if not len(ix):
                continue
            targets, events = horizon_event_targets(
                spec["decision_horizons"],
                a["horizon_observed"][ix],
                a["horizon_positive_count"][ix],
                a["horizon_last_positive"][ix],
            )
            pieces.append(dict(features=encode(a, ix), targets=targets, users=a["user_id"][ix]))
            ids_sha.update(a["case_id"][ix].astype("<u8").tobytes())
    if position != total or sum(len(x["users"]) for x in pieces) != len(chosen):
        raise ValueError("Outcome-blind global training sample differs from audited support")
    arrays = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    del pieces
    estimator = spec["head_estimator"]
    parameters, records = fit_horizon_heads(
        arrays["features"],
        arrays["targets"],
        arrays["users"],
        spec["decision_horizons"],
        C=estimator["C"],
        max_iter=estimator["max_iter"],
        tol=estimator["tol"],
        solver=estimator["solver"],
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    np.savez_compressed(out / "model.npz", **parameters)
    outputs = {"model.npz": checksum(out / "model.npz")}
    del arrays
    scores = [[] for _ in events]
    validation_cases = 0
    for bucket in cohort["buckets"]:
        path = data / f"bucket-{bucket['bucket']:02d}" / "validation-cohort.npz"
        if checksum(path) != bucket["output_sha256"]["validation-cohort.npz"]:
            raise ValueError("Validation-origin bucket changed")
        with np.load(path, allow_pickle=False) as a:
            ix = np.flatnonzero(
                (a["role_code"] == 1)
                & ((a["user_role_code"] == 1) if protocol == "unseen_users" else True)
            )
            ids = a["case_id"][ix]
            users = a["user_id"][ix]
            targets, _ = horizon_event_targets(
                spec["decision_horizons"],
                a["horizon_observed"][ix],
                a["horizon_positive_count"][ix],
                a["horizon_last_positive"][ix],
            )
            probabilities = predict_horizon_heads(encode(a, ix), parameters)
        validation_cases += len(ids)
        for j in range(len(events)):
            keep = targets[:, j] >= 0
            scores[j].append(
                metric_block(ids[keep], users[keep], targets[keep, j], probabilities[keep, j])
            )
        name = f"validation-bucket-{bucket['bucket']:02d}.npz"
        np.savez_compressed(
            out / name, case_id=ids, user_id=users, targets=targets, probabilities=probabilities
        )
        outputs[name] = checksum(out / name)
    from dataclasses import asdict

    metrics = [
        dict(event=asdict(e), **merge_metric_blocks(blocks)["metrics"])
        for e, blocks in zip(events, scores, strict=True)
    ]
    result = dict(
        status="complete",
        source_digest=digest,
        protocol=protocol,
        architecture=architecture,
        seed=seed,
        parent_fit_report_sha256=parent_sha,
        cohort_report_sha256=cfg["cohort_report_sha256"],
        audit_report_sha256=cfg["audit_report_sha256"],
        sampled_train_origins=len(chosen),
        sampled_train_case_id_sha256=ids_sha.hexdigest(),
        fit_events=records,
        validation_origins=validation_cases,
        validation_metrics=metrics,
        encoder_parameters=pr["parameters"],
        fitted_head_parameters=sum(r["kind"] == "logistic" for r in records)
        * (training["hidden"] + 1)
        + sum(r["kind"] == "jeffreys_constant" for r in records),
        output_sha256=outputs,
        wall_seconds=time.monotonic() - started,
        final_test_predictions_generated=0,
        final_test_metrics_generated=0,
        scope=cfg["scope"],
        root=str(out),
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "validation_metrics"}


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=14400,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def run(smoke):
    import json
    import itertools
    import time
    import subprocess

    volume.reload()
    saved, digest, cfg, spec, data, cohort = context()
    out = Path("/artifacts") / ("human-horizon-heads-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    volume.commit()
    if smoke:
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-W",
                "error",
                "/research/tests/test_horizon_heads.py",
                "/research/tests/test_horizon_targets.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout, flush=True)
        jobs = [("known_users", "context", 20261015)]
    else:
        if json.loads((out / "smoke.json").read_text())["status"] != "complete":
            raise ValueError("Source-bound full-fit smoke required")
        jobs = list(itertools.product(spec["protocols"], spec["architectures"], spec["seeds"]))
        while True:
            volume.reload()
            missing = [j for j in jobs if not (parent_path(cfg, *j) / "report.json").exists()]
            if not missing:
                break
            print(
                json.dumps(dict(status="waiting_for_original_encoders", missing=missing)),
                flush=True,
            )
            time.sleep(30)
    reports = []
    failures = []
    for value in fit.starmap(jobs, order_outputs=True, return_exceptions=True):
        if isinstance(value, Exception):
            failures.append(repr(value))
        else:
            reports.append(value)
        (out / "progress.json").write_text(
            json.dumps(dict(complete=len(reports), expected=len(jobs), failures=failures))
        )
        volume.commit()
    result = dict(
        status="incomplete" if failures else "complete",
        source_digest=digest,
        phase="smoke" if smoke else "full",
        fits=reports,
        failures=failures,
        root=str(out),
        final_test_predictions_generated=0,
        scope=cfg["scope"],
    )
    (out / ("smoke.json" if smoke else "manifest.json")).write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "fits"}


@app.local_entrypoint()
def main(smoke: bool = True):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger, check_attempt_allowance

    digest = snapshot()[1]
    cfg = json.loads((ROOT / "configs/human-horizon-training-v1.json").read_text())
    bundle = cfg["budget_bundle"]
    prior_states = [
        json.loads(p.read_text()) for p in (ROOT / ".campaign").glob("horizon-heads-*.json")
    ]
    reservation = BudgetLedger(
        ROOT / ".campaign/human-training-calendar-v3-budget.sqlite", 60
    ).reserve(bundle["reservation_id"], bundle["reserved_usd"])
    state = (
        ROOT
        / ".campaign"
        / ("horizon-heads-" + digest[:20] + ("-smoke.json" if smoke else "-full.json"))
    )
    check_attempt_allowance(
        digest,
        [s["source_digest"] for s in prior_states],
        maximum_versions=bundle["maximum_source_versions"],
        maximum_calls=bundle["maximum_coordinator_calls"],
        new_call=not state.exists(),
    )
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
