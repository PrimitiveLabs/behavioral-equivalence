"""Final origin-only marginal horizon comparisons on audited human trajectories."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-human-final-horizons-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(
        ROOT / "configs/human-final-horizon-v1.json", remote_path="/research/config.json"
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
    import json
    import hashlib

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/human-final-horizon-v1.json"
    )
    saved = {"cloud.py": Path(__file__).read_text(), "config.json": config.read_text()}
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in [
        "test_analysis_freeze.py",
        "test_horizon_targets.py",
        "test_streaming_scores.py",
        "test_horizon_inference.py",
        "test_horizon_archive.py",
    ]:
        saved["tests/" + name] = (base / "tests" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=3600,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def evaluate(protocol, seed):
    import json
    import hashlib
    import time
    from dataclasses import asdict
    import numpy as np
    import torch
    from behavioral_equivalence.analysis_freeze import verify_analysis_freeze
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.content_scaling import scale_duration
    from behavioral_equivalence.item_context import encode_items
    from behavioral_equivalence.quantization import assign_centers
    from behavioral_equivalence.horizon_heads import predict_horizon_heads
    from behavioral_equivalence.horizon_targets import (
        horizon_event_targets,
        origin_iid_event_probabilities,
    )
    from behavioral_equivalence.streaming_scores import (
        metric_block,
        merge_metric_blocks,
        shared_user_bootstrap,
    )
    from behavioral_equivalence.property_bounds import event_predictions
    from behavioral_equivalence.decision_model import decision_chain
    from behavioral_equivalence.core import Chain

    started = time.monotonic()
    volume.reload()
    saved, digest = snapshot()
    cfg = json.loads(saved["config.json"])
    freeze = json.loads((Path("/artifacts") / cfg["freeze_path"]).read_text())
    freeze_digest = verify_analysis_freeze(
        freeze,
        source_digest=digest,
        config_bytes=saved["config.json"].encode(),
        read_artifact=lambda name: (Path("/artifacts") / name).read_bytes(),
        phase="direct_horizon",
    )
    spec = next(p for p in cfg["probes"] if (p["protocol"], p["seed"]) == (protocol, seed))
    data = Path("/artifacts") / cfg["cohort_root"]
    if (
        checksum(data / "report.json") != cfg["cohort_report_sha256"]
        or checksum(data / cfg["audit_relative"]) != cfg["audit_report_sha256"]
    ):
        raise ValueError("Audited test trajectory identity changed")
    cohort = json.loads((data / "report.json").read_text())
    out = Path("/artifacts") / ("human-final-horizons-" + digest[:16]) / f"{protocol}-{seed}"
    if (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())
        if (
            report["source_digest"] != digest
            or report["analysis_freeze_digest"] != freeze_digest
            or any(checksum(out / n) != h for n, h in report["output_sha256"].items())
        ):
            raise ValueError("Cached final horizon analysis changed")
        return {k: v for k, v in report.items() if k not in {"metrics", "comparisons"}}
    root = Path("/artifacts") / spec["training_root"]
    original = json.loads((root / "source.json").read_text())
    training = next(
        json.loads(v)
        for k, v in original.items()
        if k.startswith("configs/human-training-calendar")
    )
    for name in [
        "content_neural.py",
        "content_scaling.py",
        "item_context.py",
        "neural.py",
        "exposure_neural.py",
    ]:
        key = "src/behavioral_equivalence/" + name
        if original[key] != saved[key]:
            raise ValueError("Frozen encoder implementation changed")
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    if checksum(inputs / "report.json") != training["input_artifacts"][protocol]["report_sha256"]:
        raise ValueError("Frozen input preprocessing changed")
    ir = json.loads((inputs / "report.json").read_text())
    for name in ["item-vocabulary.npy", "duration-scaling.json", "markov.json"]:
        if checksum(inputs / name) != ir["output_sha256"][name]:
            raise ValueError("Training-only vocabulary, scaling or Markov baseline changed")
    vocabulary = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    markov = np.asarray(json.loads((inputs / "markov.json").read_text())["probabilities"])
    torch.set_num_threads(8)
    models, heads = {}, {}
    for architecture, sha in spec["fit_report_sha256"].items():
        fit = root / "fits" / f"{protocol}-{architecture}-{seed}"
        if checksum(fit / "report.json") != sha:
            raise ValueError("Frozen neural fit changed")
        fr = json.loads((fit / "report.json").read_text())
        if checksum(fit / "best.pt") != fr["output_sha256"]["best.pt"]:
            raise ValueError("Frozen encoder weights changed")
        model = build_content_model(architecture, ir["vocabulary_size"], training["hidden"])
        model.load_state_dict(torch.load(fit / "best.pt", weights_only=True, map_location="cpu"))
        model.eval()
        models[architecture] = model
        hs = spec["heads"][architecture]
        path = Path("/artifacts") / hs["root"]
        if (
            checksum(path / "report.json") != hs["report_sha256"]
            or checksum(path / "model.npz") != hs["model_sha256"]
        ):
            raise ValueError("Frozen direct horizon heads changed")
        hr = json.loads((path / "report.json").read_text())
        if hr["source_digest"] != hs["source_digest"] or hr["parent_fit_report_sha256"] != sha:
            raise ValueError("Head and encoder identities disagree")
        with np.load(path / "model.npz", allow_pickle=False) as a:
            heads[architecture] = {key: a[key] for key in a.files}
    nested = Path("/artifacts") / spec["nested_root"]
    if checksum(nested / "report.json") != spec["nested_report_sha256"]:
        raise ValueError("Frozen nested frontier changed")
    nr = json.loads((nested / "report.json").read_text())
    parent = Path("/artifacts") / nr["parent_root"]
    if checksum(parent / "report.json") != nr["parent_report_sha256"]:
        raise ValueError("Frozen calibrated parent changed")
    pr = json.loads((parent / "report.json").read_text())
    quant = parent.parent
    if checksum(quant / "report.json") != pr["quantization_report_sha256"]:
        raise ValueError("Frozen quantization changed")
    qr = json.loads((quant / "report.json").read_text())
    if qr["parent_fit_report_sha256"] != spec["fit_report_sha256"]["gru"]:
        raise ValueError("Quantizer belongs to a different encoder")
    quantizers = {}
    for count in [16, 32, 64, 128, 256]:
        name = f"k{count}-model.npz"
        if checksum(quant / name) != qr["output_sha256"][name]:
            raise ValueError("Frozen quantizer arrays changed")
        with np.load(quant / name, allow_pickle=False) as a:
            quantizers[count] = {
                key: a[key] for key in ["mean", "scale", "centers", "probabilities"]
            }
    from behavioral_equivalence.nested_design import human_event_suite

    events = human_event_suite(tuple(cfg["decision_horizons"]))
    finite = {}
    name = "k256-parent.npz"
    if checksum(parent / name) != pr["output_sha256"][name]:
        raise ValueError("Frozen finite-state dynamics changed")
    with np.load(parent / name, allow_pickle=False) as a:
        chain = decision_chain(a["transitions"], a["emissions"], a["initial"])
    finite["parent_k256"] = np.clip(event_predictions(chain, events)[:, ::3].T, 0, 1)
    for row in nr["records"]:
        name = row["name"] + "-model.npz"
        if checksum(nested / name) != row["output_sha256"][name]:
            raise ValueError("Frozen nested dynamics changed")
        with np.load(nested / name, allow_pickle=False) as a:
            labels = tuple(
                label
                for _ in range(len(a["initial"]) // 3)
                for label in ["pending", "negative", "positive"]
            )
            chain = Chain(a["transitions"], labels, a["initial"])
            finite["nested." + row["name"]] = np.clip(
                event_predictions(chain, events)[:, 3 * a["latent_mapping"]].T, 0, 1
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    np.savez_compressed(out / "finite-model-event-tables.npz", **finite)
    blocks, outputs = (
        {},
        {"finite-model-event-tables.npz": checksum(out / "finite-model-event-tables.npz")},
    )
    origins = 0
    case_digest = hashlib.sha256()
    for bucket in cohort["buckets"]:
        path = data / f"bucket-{bucket['bucket']:02d}" / "test-cohort.npz"
        if checksum(path) != bucket["output_sha256"]["test-cohort.npz"]:
            raise ValueError("Audited origin rows changed")
        with np.load(path, allow_pickle=False) as a:
            ix = np.flatnonzero(
                (a["role_code"] == 3)
                & ((a["user_role_code"] == 3) if protocol == "unseen_users" else True)
            )
            ids, users = a["case_id"][ix], a["user_id"][ix]
            targets, _ = horizon_event_targets(
                cfg["decision_horizons"],
                a["horizon_observed"][ix],
                a["horizon_positive_count"][ix],
                a["horizon_last_positive"][ix],
            )
            tokens, tabs = a["tokens"][ix], a["tab"][ix]
            items = encode_items(vocabulary, a["video_id"][ix])
            context = scale_duration(a["intrinsic_context"][ix], scaling)
        forecasts = {
            "markov.iid": origin_iid_event_probabilities(
                markov[tokens[:, -1].astype(int) * 15 + tabs], cfg["decision_horizons"]
            )
        }
        states = {k: [] for k in quantizers}
        for architecture, model in models.items():
            zs, ps = [], []
            for offset in range(0, len(ix), cfg["inference_batch_size"]):
                sl = slice(offset, offset + cfg["inference_batch_size"])
                args = [
                    torch.as_tensor(v[sl], dtype=torch.float32 if i == 3 else torch.long)
                    for i, v in enumerate([tokens, tabs, items, context])
                ]
                with torch.inference_mode():
                    hidden = model.output[:-1](model.representation(*args))
                    z = hidden.numpy().astype(np.float64)
                    ps.append(torch.sigmoid(model.output[-1](hidden)).flatten().numpy())
                zs.append(z)
                if architecture == "gru":
                    for k, q in quantizers.items():
                        states[k].append(assign_centers(z, q["mean"], q["scale"], q["centers"]))
            z = np.concatenate(zs) if zs else np.empty((0, training["hidden"]))
            p = np.concatenate(ps) if ps else np.empty(0)
            forecasts[architecture + ".direct"] = predict_horizon_heads(z, heads[architecture])
            forecasts[architecture + ".iid"] = origin_iid_event_probabilities(
                p, cfg["decision_horizons"]
            )
        states = {
            k: np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
            for k, parts in states.items()
        }
        forecasts.update({name: table[states[256]] for name, table in finite.items()})
        for k, q in quantizers.items():
            forecasts[f"teacher_k{k}.iid"] = origin_iid_event_probabilities(
                q["probabilities"][states[k]], cfg["decision_horizons"]
            )
        for j, event in enumerate(events):
            keep = targets[:, j] >= 0
            for name, probabilities in forecasts.items():
                blocks.setdefault((j, name), []).append(
                    metric_block(ids[keep], users[keep], targets[keep, j], probabilities[keep, j])
                )
        name = f"bucket-{bucket['bucket']:02d}-test-forecasts.npz"
        factorized = {
            **{"direct." + arch: forecasts[arch + ".direct"] for arch in models},
            **{
                "iid_origin." + key: value[:, 0]
                for key, value in forecasts.items()
                if key.endswith(".iid")
            },
        }
        np.savez_compressed(
            out / name,
            case_id=ids,
            user_id=users,
            targets=targets,
            state_k256=states[256],
            **factorized,
        )
        outputs[name] = checksum(out / name)
        origins += len(ids)
        case_digest.update(ids.astype("<u8").tobytes())
        print(json.dumps(dict(bucket=bucket["bucket"], origins=origins)), flush=True)
    merged = {key: merge_metric_blocks(value) for key, value in blocks.items()}
    metrics = {f"{j}/{name}": value["metrics"] for (j, name), value in merged.items()}
    comparisons = {}
    private = {}
    for j in range(len(events)):
        members = {name: value for (event, name), value in merged.items() if event == j}
        base = members["parent_k256"]["private_user_statistics"]
        for name, value in members.items():
            stats = value["private_user_statistics"]
            if any(
                not np.array_equal(base[key], stats[key]) for key in ["user_ids", "case_counts"]
            ):
                raise ValueError("Horizon comparator supports differ")
            private[f"{j}__{name}__nll"] = stats["user_nll"]
        private[f"{j}__users"] = base["user_ids"]
        private[f"{j}__cases"] = base["case_counts"]
        if (
            len(base["user_ids"]) < cfg["minimum_cohort_users"]
            or base["case_counts"].sum() < cfg["minimum_cohort_cases"]
        ):
            comparisons[str(j)] = {"status": "unsupported"}
            continue
        contrasts = {
            name + "-vs-parent": (name, "parent_k256") for name in members if name != "parent_k256"
        }
        contrasts.update(
            {
                arch + ".direct-vs-gru.direct": (arch + ".direct", "gru.direct")
                for arch in ["context", "transformer"]
            }
        )
        comparisons[str(j)] = shared_user_bootstrap(
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
        protocol=protocol,
        seed=seed,
        events=[asdict(e) for e in events],
        origins=origins,
        ordered_case_id_sha256=case_digest.hexdigest(),
        metrics=metrics,
        comparisons=comparisons,
        output_sha256=outputs,
        root=str(out),
        wall_seconds=time.monotonic() - started,
        scope=cfg["scope"],
        forecast_storage="Lossless direct arrays, iid origin probabilities, K256state assignments and shared finite event tables",
        probability_numerics="float64 finite-model screening clipped to [0,1]; formal numeric certificates remain separate; proper-loss probabilities clipped at1e-6 consistently across methods",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k not in {"metrics", "comparisons"}}


@app.function(
    image=image, cpu=(2, 2), memory=(2048, 2048), timeout=300, retries=0, max_containers=1
)
def check_implementation():
    import subprocess

    saved, digest = snapshot()
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_horizon_inference.py",
            "/research/tests/test_horizon_archive.py",
            "/research/tests/test_horizon_targets.py",
            "/research/tests/test_analysis_freeze.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(
        status="complete",
        source_digest=digest,
        tests=result.stdout,
        scope="Synthetic numerical bridge and integrity tests; no human data opened, no final predictions generated",
    )


@app.local_entrypoint()
def main(protocol: str = "known_users", seed: int = 20261015, smoke: bool = False):
    import json
    import sys
    import hashlib

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    config_path = ROOT / "configs/human-final-horizon-v1.json"
    saved, digest = snapshot()
    if smoke:
        reservation = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40).reserve(
            "horizon-final-smoke-" + digest, 0.03
        )
        state = ROOT / ".campaign" / ("horizon-final-smoke-" + digest[:16] + ".json")
        if state.exists():
            call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
        else:
            call = check_implementation.spawn()
            state.write_text(
                json.dumps(
                    dict(
                        function_call_id=call.object_id,
                        source_digest=digest,
                        reservation=reservation,
                    ),
                    indent=2,
                )
            )
        print(state.read_text(), flush=True)
        result = call.get()
        (
            ROOT / "reports/preflight" / ("horizon-final-implementation-" + digest[:16] + ".json")
        ).write_text(json.dumps(result, indent=2))
        (
            ROOT
            / "reports/source-snapshots"
            / ("horizon-final-implementation-" + digest[:16] + ".json")
        ).write_text(json.dumps(saved, indent=2))
        print(json.dumps(result), flush=True)
        return
    freeze_path = ROOT / "reports/final-analysis-freeze.json"
    if not freeze_path.exists():
        raise ValueError("Complete joint analysis freeze required before reserving compute")
    frozen = json.loads(freeze_path.read_text())
    binding = frozen.get("controllers", {}).get("direct_horizon", {})
    if (
        frozen.get("status") != "frozen"
        or binding.get("source_digest") != digest
        or binding.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Local final horizon analysis freeze changed")
    reservation = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40).reserve(
        f"final-horizon-{digest}-{protocol}-{seed}", 3600 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    )
    state = ROOT / ".campaign" / f"final-horizon-{digest[:16]}-{protocol}-{seed}.json"
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = evaluate.spawn(protocol, seed)
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
