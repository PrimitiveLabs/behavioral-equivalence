"""Final observed-response refinement and held-out-calibration-group evaluation."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("response_all_groups", "response_heldout_groups")
app = modal.App("behavioral-equivalence-human-final-responses-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
)
for phase in PHASES:
    image = image.add_local_file(
        ROOT / ("configs/human-final-" + phase.replace("_", "-") + "-v1.json"),
        remote_path="/research/" + phase + ".json",
    )


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def snapshot(phase):
    import json
    import hashlib

    if phase not in PHASES:
        raise ValueError("Known final response phase required")
    base = Path("/research") if Path("/research").exists() else ROOT
    config = (
        base / (phase + ".json")
        if base == Path("/research")
        else base / ("configs/human-final-" + phase.replace("_", "-") + "-v1.json")
    )
    saved = {"cloud.py": Path(__file__).read_text(), "config.json": config.read_text()}
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in [
        "test_streaming_scores.py",
        "test_response_replicas.py",
        "test_case_inference.py",
        "test_analysis_freeze.py",
        "test_horizon_inference.py",
    ]:
        saved["tests/" + name] = (base / "tests" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=3600,
    max_containers=2,
    retries=0,
    volumes={"/artifacts": volume},
)
def evaluate(phase, seed):
    import json
    import hashlib
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.analysis_freeze import verify_analysis_freeze
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.content_scaling import scale_duration
    from behavioral_equivalence.item_context import encode_items
    from behavioral_equivalence.quantization import assign_centers
    from behavioral_equivalence.case_inference import conditional_response_forecasts
    from behavioral_equivalence.response_replicas import response_replicas
    from behavioral_equivalence.heldout_response import response_group_folds
    from behavioral_equivalence.streaming_scores import (
        metric_block,
        merge_metric_blocks,
        shared_user_bootstrap,
    )

    started = time.monotonic()
    volume.reload()
    saved, digest = snapshot(phase)
    cfg = json.loads(saved["config.json"])
    frozen = json.loads((Path("/artifacts") / cfg["freeze_path"]).read_text())
    freeze_digest = verify_analysis_freeze(
        frozen,
        source_digest=digest,
        config_bytes=saved["config.json"].encode(),
        read_artifact=lambda name: (Path("/artifacts") / name).read_bytes(),
        phase=phase,
    )
    spec = next(s for s in cfg["probes"] if s["seed"] == seed)
    out = Path("/artifacts") / ("human-final-" + phase + "-" + digest[:16]) / str(seed)
    if (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())
        if (
            report["source_digest"] != digest
            or report["analysis_freeze_digest"] != freeze_digest
            or any(checksum(out / n) != h for n, h in report["output_sha256"].items())
        ):
            raise ValueError("Cached response evaluation changed")
        return {k: v for k, v in report.items() if k not in {"metrics", "comparisons"}}
    history = Path("/artifacts") / spec["history_root"]
    frontier = Path("/artifacts") / spec["frontier_root"]
    if (
        checksum(history / "report.json") != spec["history_report_sha256"]
        or checksum(frontier / "report.json") != spec["frontier_report_sha256"]
    ):
        raise ValueError("Frozen history or response frontier changed")
    hr = json.loads((history / "report.json").read_text())
    fr = json.loads((frontier / "report.json").read_text())
    hs = json.loads((history / "source.json").read_text())
    if (
        hr["seed"] != seed
        or hr["protocol"] != "known_users"
        or fr["seed"] != seed
        or fr["input_reports"]["history"] != spec["history_report_sha256"]
    ):
        raise ValueError("Response history/frontier identities disagree")
    if (
        checksum(history / "model.npz") != hr["output_sha256"]["model.npz"]
        or checksum(frontier / "models.npz") != fr["output_sha256"]["models.npz"]
    ):
        raise ValueError("History quantizer or response table changed")
    with np.load(history / "model.npz", allow_pickle=False) as a:
        centers, mean, scale = [a[k] for k in ["centers", "mean", "scale"]]
    with np.load(frontier / "models.npz", allow_pickle=False) as a:
        parameters = {key: a[key] for key in a.files}
    root = Path("/artifacts") / spec["training_root"]
    source = json.loads((root / "source.json").read_text())
    training = next(
        json.loads(v) for k, v in source.items() if k.startswith("configs/human-training-calendar")
    )
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "content_scaling.py",
        "item_context.py",
    ]:
        key = "src/behavioral_equivalence/" + name
        if source[key] != saved[key]:
            raise ValueError("Frozen neural implementation changed")
    inputs = Path("/artifacts") / training["input_artifacts"]["known_users"]["path"]
    if (
        checksum(inputs / "report.json")
        != training["input_artifacts"]["known_users"]["report_sha256"]
    ):
        raise ValueError("Training preprocessing report changed")
    ir = json.loads((inputs / "report.json").read_text())
    for name in ["item-vocabulary.npy", "duration-scaling.json", "markov.json"]:
        if checksum(inputs / name) != ir["output_sha256"][name]:
            raise ValueError("Training-only baseline/preprocessing changed")
    vocabulary = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    markov = np.asarray(json.loads((inputs / "markov.json").read_text())["probabilities"])
    torch.set_num_threads(8)
    models = {}
    for arch, sha in spec["fit_report_sha256"].items():
        path = root / "fits" / f"known_users-{arch}-{seed}"
        if checksum(path / "report.json") != sha:
            raise ValueError("Neural comparator report changed")
        report = json.loads((path / "report.json").read_text())
        if checksum(path / "best.pt") != report["output_sha256"]["best.pt"]:
            raise ValueError("Neural comparator weights changed")
        model = build_content_model(arch, ir["vocabulary_size"], training["hidden"])
        model.load_state_dict(torch.load(path / "best.pt", weights_only=True, map_location="cpu"))
        model.eval()
        models[arch] = model
    if spec["fit_report_sha256"]["gru"] != hr["parent_fit_report_sha256"]:
        raise ValueError("History encoder and GRU comparator differ")
    hc = json.loads(hs["config.json"])
    cohort = Path("/artifacts") / hc["random_cohort_root"]
    if (
        checksum(cohort / "report.json") != hc["random_cohort_report_sha256"]
        or checksum(cohort / hc["random_audit_relative"]) != hc["random_audit_sha256"]
    ):
        raise ValueError("Audited randomized response cohort changed")
    cr = json.loads((cohort / "report.json").read_text())
    heldout = phase == "response_heldout_groups"
    folds = cfg["folds"] if heldout else [dict(name="all", heldout_groups=[0, 1, 2, 3])]
    if heldout and folds != [
        dict(
            name="q" + "".join(map(str, f["query"])),
            query_groups=list(f["query"]),
            heldout_groups=list(f["heldout"]),
        )
        for f in response_group_folds()
    ]:
        raise ValueError("All six fixed folds required")
    names = list(dict.fromkeys(r["name"] for r in fr["records"]))
    if cfg["primary_candidate"] not in names or cfg["primary_reference"] not in names:
        raise ValueError("Frozen primary models absent")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    blocks = {}
    outputs = {}
    cases = 0
    case_sha = hashlib.sha256()
    users_seen = set()
    for bucket in cr["buckets"]:
        path = cohort / f"bucket-{bucket['bucket']:02d}" / "randomized.npz"
        if checksum(path) != bucket["output_sha256"]["randomized.npz"]:
            raise ValueError("Randomized case bucket changed")
        with np.load(path, allow_pickle=False) as a:
            ix = np.flatnonzero((a["role_code"] == 3) & (a["duration_group"] < 4))
            ids, users, y, groups, env = [
                a[key][ix]
                for key in ["case_id", "user_id", "target", "duration_group", "environment"]
            ]
            tokens, tabs = a["tokens"][ix], a["tab"][ix]
            items = encode_items(vocabulary, a["video_id"][ix])
            context = scale_duration(a["intrinsic_context"][ix], scaling)
        current = set(map(int, np.unique(users)))
        if current & users_seen:
            raise ValueError("Original users cross source buckets")
        users_seen.update(current)
        case_sha.update(ids.astype("<u8").tobytes())
        cases += len(ids)
        states = []
        baseline = {"markov": markov[tokens[:, -1].astype(int) * 15 + tabs]}
        for arch, model in models.items():
            predictions = []
            for offset in range(0, len(ids), cfg["inference_batch_size"]):
                sl = slice(offset, offset + cfg["inference_batch_size"])
                args = [
                    torch.as_tensor(value[sl], dtype=torch.float32 if j == 3 else torch.long)
                    for j, value in enumerate([tokens, tabs, items, context])
                ]
                with torch.inference_mode():
                    predictions.append(torch.sigmoid(model(*args)).numpy())
                    if arch == "gru":
                        states.append(
                            assign_centers(model.encode(args[0]).numpy(), mean, scale, centers)
                        )
            baseline[arch] = np.concatenate(predictions) if predictions else np.empty(0)
        state = np.concatenate(states) if states else np.empty(0, dtype=np.int64)
        fold_forecasts = {}
        for fold in folds:
            fold_name = fold["name"]
            forecast = dict(baseline)
            for name in names:
                key = fold_name + "_" + name if heldout else name
                forecast[name] = conditional_response_forecasts(
                    state,
                    env,
                    parameters["fine_to_coarse"],
                    parameters[key + "_mapping"],
                    parameters[key + "_abstract"],
                )
            fold_forecasts[fold_name] = forecast
            keep = np.isin(groups, fold["heldout_groups"])
            if heldout:
                for name, p in forecast.items():
                    blocks.setdefault((fold_name, name), []).append(
                        metric_block(ids[keep], users[keep], y[keep], p[keep])
                    )
        for name in list(baseline) + names:
            p = (
                response_replicas(
                    groups, {fold["name"]: fold_forecasts[fold["name"]][name] for fold in folds}
                )
                if heldout
                else fold_forecasts["all"][name]
            )
            blocks.setdefault(("aggregate", name), []).append(metric_block(ids, users, y, p))
        name = f"bucket-{bucket['bucket']:02d}-test-origins.npz"
        np.savez_compressed(
            out / name,
            case_id=ids,
            user_id=users,
            target=y,
            duration_group=groups,
            environment=env,
            fine_state=state,
            **baseline,
        )
        outputs[name] = checksum(out / name)
    if (
        cases != cfg["expected_cases"]
        or len(users_seen) != cfg["expected_users"]
        or case_sha.hexdigest() != cfg["expected_case_id_sha256"]
    ):
        raise ValueError("Outcome-free eligibility contract changed")
    merged = {key: merge_metric_blocks(value) for key, value in blocks.items()}
    metrics = {fold + "/" + name: value["metrics"] for (fold, name), value in merged.items()}
    comparisons = {}
    private = {}
    for fold_name in sorted({key[0] for key in merged}):
        members = {name: value for (fold, name), value in merged.items() if fold == fold_name}
        base = members[cfg["primary_reference"]]["private_user_statistics"]
        for name, value in members.items():
            stats = value["private_user_statistics"]
            if any(
                not np.array_equal(base[key], stats[key]) for key in ["user_ids", "case_counts"]
            ):
                raise ValueError(
                    "Response comparisons must retain identical original user/case supports"
                )
            private[fold_name + "__" + name + "__nll"] = stats["user_nll"]
        private[fold_name + "__users"] = base["user_ids"]
        private[fold_name + "__cases"] = base["case_counts"]
        if (
            len(base["user_ids"]) < cfg["minimum_cohort_users"]
            or base["case_counts"].sum() < cfg["minimum_cohort_cases"]
        ):
            comparisons[fold_name] = dict(status="unsupported")
            continue
        contrasts = {
            name + "-vs-reference": (name, cfg["primary_reference"])
            for name in members
            if name != cfg["primary_reference"]
        }
        contrasts.update(
            {
                cfg["primary_candidate"] + "-vs-" + name: (cfg["primary_candidate"], name)
                for name in ["full_parent", "unrefined", "gru", "transformer", "context", "markov"]
            }
        )
        comparisons[fold_name] = shared_user_bootstrap(
            base["user_ids"],
            base["case_counts"],
            {name: value["private_user_statistics"]["user_nll"] for name, value in members.items()},
            contrasts,
            replications=cfg["bootstrap_replications"],
            seed=cfg["bootstrap_seed"],
            margin=cfg["noninferiority_margin_nats"],
        )
    np.savez_compressed(out / "private-user-losses.npz", **private)
    outputs["private-user-losses.npz"] = checksum(out / "private-user-losses.npz")
    report = dict(
        status="complete",
        source_digest=digest,
        analysis_freeze_digest=freeze_digest,
        phase=phase,
        seed=seed,
        protocol="known_users",
        cases=cases,
        users=len(users_seen),
        ordered_case_id_sha256=case_sha.hexdigest(),
        primary_candidate=cfg["primary_candidate"],
        primary_reference=cfg["primary_reference"],
        metrics=metrics,
        comparisons=comparisons,
        frontier_records=fr["records"],
        output_sha256=outputs,
        root=str(out),
        wall_seconds=time.monotonic() - started,
        scope=cfg["scope"],
        forecast_storage="Fine history states and immutable environments plus bound frontier tables; neural/Markov forecasts saved directly. Reconstruct each fold, average losses across three held-out folds per original case before user aggregation.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k not in {"metrics", "comparisons"}}


@app.function(image=image, cpu=(2, 2), memory=(2048, 2048), timeout=300, retries=0)
def check_implementation(phase):
    import subprocess

    saved, digest = snapshot(phase)
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            *[
                f"/research/tests/{name}"
                for name in [
                    "test_streaming_scores.py",
                    "test_response_replicas.py",
                    "test_case_inference.py",
                    "test_analysis_freeze.py",
                    "test_horizon_inference.py",
                ]
            ],
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(
        status="complete",
        source_digest=digest,
        tests=result.stdout,
        scope="Data-free original-user fold scoring, response lookup and neural inference tests; no human test data opened",
    )


@app.local_entrypoint()
def main(phase: str = "response_all_groups", seed: int = 20261015, smoke: bool = False):
    import json
    import sys
    import hashlib

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    saved, digest = snapshot(phase)
    cfg = json.loads(saved["config.json"])
    if seed not in {p["seed"] for p in cfg["probes"]}:
        raise ValueError("Only the five fixed known-user seeds may run")
    if not smoke:
        path = ROOT / "reports/final-analysis-freeze.json"
        if not path.exists():
            raise ValueError("Joint analysis lock required before final scoring")
        frozen = json.loads(path.read_text())
        binding = frozen.get("controllers", {}).get(phase, {})
        if (
            frozen.get("status") != "frozen"
            or binding.get("source_digest") != digest
            or binding.get("config_sha256")
            != hashlib.sha256(saved["config.json"].encode()).hexdigest()
        ):
            raise ValueError("Frozen response controller/configuration changed")
    identity = f"final-{phase}-{digest}-{seed}-" + ("smoke" if smoke else "test")
    reservation = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40).reserve(
        identity, 0.03 if smoke else 3600 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    )
    state = ROOT / ".campaign" / (identity + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = check_implementation.spawn(phase) if smoke else evaluate.spawn(phase, seed)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, source_digest=digest, reservation=reservation
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    result = call.get()
    if smoke:
        (
            ROOT
            / "reports/preflight"
            / ("final-response-check-" + phase + "-" + digest[:16] + ".json")
        ).write_text(json.dumps(result, indent=2))
        (
            ROOT
            / "reports/source-snapshots"
            / ("final-response-check-" + phase + "-" + digest[:16] + ".json")
        ).write_text(json.dumps(saved, indent=2))
    print(json.dumps(result), flush=True)
