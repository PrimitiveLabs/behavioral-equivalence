"""Full comparator bridge on validation only, before the final human analysis freeze."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-comparative-inference-validation-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(
        ROOT / "configs/comparative-inference-validation-v1.json",
        remote_path="/research/config.json",
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def snapshot():
    import hashlib
    import json

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config_path = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/comparative-inference-validation-v1.json"
    )
    saved = {"cloud.py": Path(__file__).read_text(), "config.json": config_path.read_text()}
    for path in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + path.name] = path.read_text()
    for name in ["test_streaming_scores.py", "test_case_inference.py"]:
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
def validate(protocol, seed):
    import hashlib
    import json
    import subprocess
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.case_inference import (
        forecast_indices,
        available_history_lengths,
        normalized_positive_emissions,
    )
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.content_scaling import scale_duration
    from behavioral_equivalence.item_context import encode_items
    from behavioral_equivalence.quantization import assign_centers
    from behavioral_equivalence.streaming_scores import (
        metric_block,
        merge_metric_blocks,
        shared_user_bootstrap,
    )

    started = time.monotonic()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    if config["split"] != "validation":
        raise ValueError("Comparative bridge validation cannot generate final test predictions")
    spec = next(s for s in config["probes"] if s["protocol"] == protocol and s["seed"] == seed)
    tests = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_streaming_scores.py",
            "/research/tests/test_case_inference.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    volume.reload()
    out = (
        Path("/artifacts")
        / ("comparative-inference-validation-" + digest[:16])
        / f"{protocol}-{seed}"
    )
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if cached["source_digest"] != digest or any(
            checksum(out / n) != sha for n, sha in cached["output_sha256"].items()
        ):
            raise ValueError("Cached validation inference changed")
        return {k: v for k, v in cached.items() if k not in {"metrics", "comparisons"}}
    root = Path("/artifacts") / spec["training_root"]
    original = json.loads((root / "source.json").read_text())
    source_digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    training = next(
        json.loads(v)
        for k, v in original.items()
        if k.startswith("configs/human-training-calendar")
    )
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "content_scaling.py",
        "item_context.py",
    ]:
        key = "src/behavioral_equivalence/" + name
        if original[key] != saved[key]:
            raise ValueError("Frozen encoder/preprocessing implementation changed")
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    if checksum(inputs / "report.json") != training["input_artifacts"][protocol]["report_sha256"]:
        raise ValueError("Frozen input report changed")
    input_report = json.loads((inputs / "report.json").read_text())
    for name in ["item-vocabulary.npy", "duration-scaling.json", "markov.json"]:
        if checksum(inputs / name) != input_report["output_sha256"][name]:
            raise ValueError("Frozen training preprocessing/baseline changed")
    vocabulary = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    markov = np.asarray(json.loads((inputs / "markov.json").read_text())["probabilities"])
    models, fit_reports, previous = {}, {}, {}
    torch.set_num_threads(8)
    for architecture, sha in spec["fit_report_sha256"].items():
        path = root / "fits" / f"{protocol}-{architecture}-{seed}"
        if checksum(path / "report.json") != sha:
            raise ValueError("Frozen neural fit report changed")
        report = json.loads((path / "report.json").read_text())
        if (
            report["status"] != "complete"
            or report["source_digest"] != source_digest
            or report["input_report_sha256"] != checksum(inputs / "report.json")
        ):
            raise ValueError("Completed fit with common training-only preprocessing required")
        if any(checksum(path / name) != value for name, value in report["output_sha256"].items()):
            raise ValueError("Neural fit weights/predictions changed")
        model = build_content_model(
            architecture, input_report["vocabulary_size"], training["hidden"]
        )
        model.load_state_dict(torch.load(path / "best.pt", weights_only=True, map_location="cpu"))
        model.eval()
        models[architecture], fit_reports[architecture] = model, report
        with np.load(path / "validation-predictions.npz", allow_pickle=False) as a:
            previous[architecture] = {
                key: a[key] for key in ["case_id", "user_id", "target", "probability"]
            }
    if set(models) != {"context", "gru", "transformer"}:
        raise ValueError("All three fixed neural comparators required")
    nested = Path("/artifacts") / spec["nested_root"]
    if checksum(nested / "report.json") != spec["nested_report_sha256"]:
        raise ValueError("Nested frontier changed")
    nested_report = json.loads((nested / "report.json").read_text())
    parent = Path("/artifacts") / nested_report["parent_root"]
    if checksum(parent / "report.json") != nested_report["parent_report_sha256"]:
        raise ValueError("Calibrated point parent changed")
    parent_report = json.loads((parent / "report.json").read_text())
    parent_name = "k256-parent.npz"
    if checksum(parent / parent_name) != parent_report["output_sha256"][parent_name]:
        raise ValueError("Calibrated parent arrays changed")
    with np.load(parent / parent_name, allow_pickle=False) as a:
        parent_probability = normalized_positive_emissions(a["emissions"])
    quant = parent.parent
    if checksum(quant / "report.json") != parent_report["quantization_report_sha256"]:
        raise ValueError("Training quantization report changed")
    quant_report = json.loads((quant / "report.json").read_text())
    if (
        quant_report["status"] != "complete"
        or quant_report["protocol"] != protocol
        or quant_report["seed"] != seed
        or quant_report["parent_fit_report_sha256"] != spec["fit_report_sha256"]["gru"]
        or nested_report["protocol"] != protocol
        or nested_report["seed"] != seed
    ):
        raise ValueError("Quantizer and nested frontier must belong to this exact GRU fit")
    if (
        checksum(quant / "validation-predictions.npz")
        != quant_report["output_sha256"]["validation-predictions.npz"]
    ):
        raise ValueError("Quantizer validation reference changed")
    with np.load(quant / "validation-predictions.npz", allow_pickle=False) as a:
        previous_quant = {key: a[key] for key in a.files}
    quantizers = {}
    for count in [16, 32, 64, 128, 256]:
        name = f"k{count}-model.npz"
        if checksum(quant / name) != quant_report["output_sha256"][name]:
            raise ValueError("Training-only quantizer changed")
        with np.load(quant / name, allow_pickle=False) as a:
            quantizers[count] = {
                key: a[key] for key in ["centers", "mean", "scale", "probabilities", "counts"]
            }
    nested_probabilities = {}
    for row in nested_report["records"]:
        for name, sha in row["output_sha256"].items():
            if checksum(nested / name) != sha:
                raise ValueError("Stored nested model/certificate changed")
        with np.load(nested / (row["name"] + "-model.npz"), allow_pickle=False) as a:
            nested_probabilities["nested." + row["name"]] = a["next_positive"]
    primary = "nested." + nested_report["primary"]
    primary_record = next(
        r for r in nested_report["records"] if r["name"] == nested_report["primary"]
    )
    if primary_record["reduced_latent_states"] == 256 and not np.array_equal(
        nested_probabilities[primary], parent_probability
    ):
        raise ValueError("Identity primary must reproduce exact-normalized parent forecasts")
    cases = Path("/artifacts") / training["case_root"]
    intrinsic = Path("/artifacts") / training["context_root"]
    if (
        checksum(cases / "report.json") != training["case_report_sha256"]
        or checksum(cases / training["case_audit_relative"]) != training["case_audit_sha256"]
    ):
        raise ValueError("Independently audited raw cases changed")
    if checksum(intrinsic / "report.json") != training["context_report_sha256"]:
        raise ValueError("Audited context sidecar changed")
    case_report = json.loads((cases / "report.json").read_text())
    context_report = json.loads((intrinsic / "report.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    summaries = {}
    outputs = {}
    case_sha = hashlib.sha256()
    position = 0
    maximum_reproduction_errors = {a: 0.0 for a in models}
    batch = config["inference_batch_size"]
    for bucket in case_report["buckets"]:
        index = bucket["bucket"]
        name = f"embargo-{training['embargo_hours']:02d}h.npz"
        path = cases / f"bucket-{index:02d}" / name
        context_path = intrinsic / f"bucket-{index:02d}" / name
        if (
            checksum(path) != bucket["output_sha256"][name]
            or checksum(context_path) != context_report["buckets"][index]["output_sha256"][name]
        ):
            raise ValueError("Raw case/context bucket changed")
        with np.load(context_path, allow_pickle=False) as a:
            context_ids, context_values = a["case_id"], a["context"]
        with np.load(path, allow_pickle=False) as a:
            selected = forecast_indices(
                a["case_id"],
                a["user_id"],
                a["role_code"],
                a["user_role_code"],
                context_ids,
                protocol,
                "validation",
            )
            part = {
                key: a[key][selected]
                for key in ["case_id", "user_id", "tokens", "tab", "video_id", "target"]
            }
        ids, users, targets = [part[key] for key in ["case_id", "user_id", "target"]]
        case_sha.update(ids.astype("<u8").tobytes())
        context = scale_duration(context_values[selected], scaling)
        item_ids = encode_items(vocabulary, part["video_id"])
        forecast = {"markov": markov[part["tokens"][:, -1].astype(int) * 15 + part["tab"]]}
        states = {k: [] for k in quantizers}
        for architecture, model in models.items():
            pieces = []
            for offset in range(0, len(ids), batch):
                sl = slice(offset, offset + batch)
                args = [
                    torch.as_tensor(a[sl], dtype=torch.float32 if i == 3 else torch.long)
                    for i, a in enumerate([part["tokens"], part["tab"], item_ids, context])
                ]
                with torch.inference_mode():
                    hidden = model.output[:-1](model.representation(*args))
                    pieces.append(torch.sigmoid(model.output[-1](hidden).squeeze(-1)).numpy())
                if architecture == "gru":
                    z = hidden.numpy().astype(np.float64)
                    for count, q in quantizers.items():
                        states[count].append(assign_centers(z, q["mean"], q["scale"], q["centers"]))
            forecast[architecture] = np.concatenate(pieces) if pieces else np.empty(0)
            old = previous[architecture]
            sl = slice(position, position + len(ids))
            if any(
                not np.array_equal(old[key][sl], part[key])
                for key in ["case_id", "user_id", "target"]
            ):
                raise ValueError(
                    "Immutable validation cases/order differ from frozen neural predictions"
                )
            error = float(
                np.max(np.abs(old["probability"][sl] - forecast[architecture]), initial=0)
            )
            maximum_reproduction_errors[architecture] = max(
                maximum_reproduction_errors[architecture], error
            )
            if error > config["probability_absolute_tolerance"]:
                raise ValueError(
                    "Raw-case neural forecast does not reproduce frozen validation predictions"
                )
        position += len(ids)
        states = {
            k: np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)
            for k, pieces in states.items()
        }
        for count, q in quantizers.items():
            forecast[f"teacher_k{count}"] = q["probabilities"][states[count]]
            sl = slice(position - len(ids), position)
            if (
                not np.array_equal(previous_quant["case_id"][sl], ids)
                or not np.array_equal(previous_quant[f"k{count}_state"][sl], states[count])
                or not np.array_equal(
                    previous_quant[f"k{count}_probability"][sl], forecast[f"teacher_k{count}"]
                )
            ):
                raise ValueError(
                    "Raw-case quantizer assignments/forecasts do not reproduce the frozen validation artifact"
                )
        forecast["parent_k256"] = parent_probability[states[256]]
        forecast.update(
            {name: values[states[256]] for name, values in nested_probabilities.items()}
        )
        lengths = available_history_lengths(part["tokens"])
        cohorts = dict(
            all=np.ones(len(ids), dtype=bool),
            history_0=lengths == 0,
            history_1_8=(lengths >= 1) & (lengths <= 8),
            history_9_16=(lengths >= 9) & (lengths <= 16),
            history_17_32=lengths >= 17,
            context_missing=context[:, 1] == 1,
            context_known=context[:, 1] == 0,
            sparse_training_state=quantizers[256]["counts"][states[256]]
            < config["sparse_training_state_cases"],
        )
        cohorts.update({f"tab_{tab}": part["tab"] == tab for tab in range(15)})
        cohort_models = {"markov", "context", "gru", "transformer", "parent_k256", primary}
        for cohort, chosen in cohorts.items():
            for model_name, probabilities in forecast.items():
                if cohort != "all" and model_name not in cohort_models:
                    continue
                summaries.setdefault((cohort, model_name), []).append(
                    metric_block(ids[chosen], users[chosen], targets[chosen], probabilities[chosen])
                )
        name = f"bucket-{index:02d}-validation-forecasts.npz"
        np.savez_compressed(out / name, case_id=ids, user_id=users, target=targets, **forecast)
        outputs[name] = checksum(out / name)
        print(
            json.dumps(
                dict(
                    protocol=protocol,
                    seed=seed,
                    bucket=index,
                    cases=len(ids),
                    cumulative_cases=position,
                )
            ),
            flush=True,
        )
    if any(len(old["case_id"]) != position for old in previous.values()):
        raise ValueError("Incomplete validation inference")
    merged = {key: merge_metric_blocks(value) for key, value in summaries.items()}
    metrics = {f"{cohort}/{name}": value["metrics"] for (cohort, name), value in merged.items()}
    comparisons = {}
    sufficient_statistics = {}
    for cohort in sorted({key[0] for key in merged}):
        members = {name: value for (group, name), value in merged.items() if group == cohort}
        base = members["parent_k256"]["private_user_statistics"]
        for name, value in members.items():
            stats = value["private_user_statistics"]
            if any(
                not np.array_equal(base[key], stats[key]) for key in ["user_ids", "case_counts"]
            ):
                raise ValueError("Comparator user/case supports differ")
            sufficient_statistics[cohort + "__" + name + "__nll"] = stats["user_nll"]
        sufficient_statistics[cohort + "__users"] = base["user_ids"]
        sufficient_statistics[cohort + "__cases"] = base["case_counts"]
        supported = (
            len(base["user_ids"]) >= config["minimum_cohort_users"]
            and base["case_counts"].sum() >= config["minimum_cohort_cases"]
        )
        if not supported:
            comparisons[cohort] = dict(
                status="unsupported",
                users=len(base["user_ids"]),
                cases=int(base["case_counts"].sum()),
            )
            continue
        contrasts = {
            name + "-vs-parent": (name, "parent_k256") for name in members if name != "parent_k256"
        }
        contrasts.update(
            {
                "parent-vs-gru": ("parent_k256", "gru"),
                "primary-vs-gru": (primary, "gru"),
                "transformer-vs-gru": ("transformer", "gru"),
            }
        )
        comparisons[cohort] = shared_user_bootstrap(
            base["user_ids"],
            base["case_counts"],
            {name: value["private_user_statistics"]["user_nll"] for name, value in members.items()},
            contrasts,
            replications=config["bootstrap_replications"],
            seed=config["bootstrap_seed"],
            margin=config["noninferiority_margin_nats"],
        )
    np.savez_compressed(out / "private-user-losses.npz", **sufficient_statistics)
    outputs["private-user-losses.npz"] = checksum(out / "private-user-losses.npz")
    report = dict(
        status="complete",
        source_digest=digest,
        protocol=protocol,
        seed=seed,
        split="validation",
        cases=position,
        models=len(forecast),
        primary=primary,
        ordered_case_id_sha256=case_sha.hexdigest(),
        maximum_neural_reproduction_errors=maximum_reproduction_errors,
        fit_report_sha256=spec["fit_report_sha256"],
        quantization_report_sha256=parent_report["quantization_report_sha256"],
        nested_report_sha256=spec["nested_report_sha256"],
        parameter_counts={a: f["parameters"] for a, f in fit_reports.items()},
        metrics=metrics,
        comparisons=comparisons,
        tests=tests.stdout,
        output_sha256=outputs,
        wall_seconds=time.monotonic() - started,
        final_test_predictions_generated=0,
        final_test_metrics_generated=0,
        scope=config["scope"],
        root=str(out),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k not in {"metrics", "comparisons"}}


@app.local_entrypoint()
def main(protocol: str = "known_users", seed: int = 20261015):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    config = json.loads((ROOT / "configs/comparative-inference-validation-v1.json").read_text())
    if (protocol, seed) not in {(s["protocol"], s["seed"]) for s in config["probes"]}:
        raise ValueError("Only fixed validation probes may run")
    digest = snapshot()[1]
    reservation = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40).reserve(
        f"comparative-validation-{digest}-{protocol}-{seed}",
        3600 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1,
    )
    state = ROOT / ".campaign" / f"comparative-validation-{digest[:20]}-{protocol}-{seed}.json"
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = validate.spawn(protocol, seed)
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
