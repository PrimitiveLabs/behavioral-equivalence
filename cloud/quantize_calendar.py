"""Private training-only quantizers for immutable, completed neural fits."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-calendar-quantization-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "8"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-quantization-v1.json",
        remote_path="/research/quantization-config.json",
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

    base = Path("/research") if Path("/research/quantization-config.json").exists() else ROOT
    config = base / (
        "quantization-config.json"
        if base == Path("/research")
        else "configs/human-quantization-v1.json"
    )
    saved = {
        "cloud/quantize_calendar.py": Path(__file__).read_text(),
        "configs/human-quantization-v1.json": config.read_text(),
    }
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "statistics.py",
        "quantization.py",
    ]:
        saved["src/behavioral_equivalence/" + name] = (
            base / "src/behavioral_equivalence" / name
        ).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=21600,
    max_containers=2,
    retries=0,
    volumes={"/artifacts": volume},
)
def quantize(training_root: str, protocol: str, architecture: str, seed: int):
    import json
    import hashlib
    import time
    import numpy as np
    import torch
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.quantization import (
        fit_representation_scaling,
        assign_centers,
        emission_probabilities,
    )
    from behavioral_equivalence.statistics import prediction_scores

    if not training_root.replace("-", "").isalnum() or protocol not in {
        "known_users",
        "unseen_users",
    }:
        raise ValueError("Valid immutable training root and user protocol required")
    volume.reload()
    started = time.monotonic()
    root = Path("/artifacts") / training_root
    source = json.loads((root / "source.json").read_text())
    parent_digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    configs = [
        json.loads(text)
        for name, text in source.items()
        if name.startswith("configs/human-training-calendar")
    ]
    if len(configs) != 1:
        raise ValueError("One frozen parent training configuration required")
    training = configs[0]
    config = json.loads(Path("/research/quantization-config.json").read_text())
    if (
        architecture
        not in {config["allowed_final_architecture"], config["engineering_architecture"]}
        or seed not in training["seeds"]
    ):
        raise ValueError("Frozen parent architecture and seed required")
    for name in ["neural.py", "exposure_neural.py", "content_neural.py", "statistics.py"]:
        path = "src/behavioral_equivalence/" + name
        if source[path] != (Path("/research") / path).read_text():
            raise ValueError("Parent model or score implementation changed")
    fit = root / "fits" / f"{protocol}-{architecture}-{seed}"
    report_raw = (fit / "report.json").read_bytes()
    report = json.loads(report_raw)
    if (
        report["status"] != "complete"
        or report["source_digest"] != parent_digest
        or report["protocol"] != protocol
        or report["architecture"] != architecture
        or report["seed"] != seed
        or any(checksum(fit / name) != sha for name, sha in report["output_sha256"].items())
    ):
        raise ValueError("Complete immutable parent checkpoint required")
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    input_sha = checksum(inputs / "report.json")
    if (
        input_sha != training["input_artifacts"][protocol]["report_sha256"]
        or input_sha != report["input_report_sha256"]
    ):
        raise ValueError("Parent training inputs differ")
    input_report = json.loads((inputs / "report.json").read_text())
    if any(
        checksum(inputs / (split + ".npz")) != input_report["output_sha256"][split + ".npz"]
        for split in ["train", "validation"]
    ):
        raise ValueError("Train/validation artifacts changed")
    saved_source, digest = snapshot()
    fit_sha = hashlib.sha256(report_raw).hexdigest()
    out = fit / ("quantization-" + digest[:12] + "-" + fit_sha[:12])
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if (
            cached["source_digest"] != digest
            or cached["parent_fit_report_sha256"] != fit_sha
            or any(checksum(out / name) != sha for name, sha in cached["output_sha256"].items())
        ):
            raise ValueError("Cached quantization evidence changed")
        return cached
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved_source, indent=2))
    (out / "started.json").write_text(
        json.dumps(
            dict(
                status="running",
                function_call_id=modal.current_function_call_id(),
                source_digest=digest,
                parent_fit_report_sha256=fit_sha,
            )
        )
    )
    volume.commit()
    torch.set_num_threads(8)
    model = build_content_model(architecture, input_report["vocabulary_size"], training["hidden"])
    model.load_state_dict(torch.load(fit / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    data = {}
    for split in ["train", "validation"]:
        with np.load(inputs / (split + ".npz"), allow_pickle=False) as arrays:
            data[split] = {
                name: arrays[name]
                for name in [
                    "case_id",
                    "user_id",
                    "target",
                    "tokens",
                    "tab",
                    "item_id",
                    "intrinsic_context",
                ]
            }

    def infer(arrays, selected):
        args = [
            torch.as_tensor(
                arrays[key][selected],
                dtype=torch.float32 if key == "intrinsic_context" else torch.long,
            )
            for key in ["tokens", "tab", "item_id", "intrinsic_context"]
        ]
        with torch.inference_mode():
            hidden = model.output[:-1](model.representation(*args))
            probability = torch.sigmoid(model.output[-1](hidden).squeeze(-1))
        return hidden.numpy().astype(np.float64), probability.numpy().astype(np.float64)

    n = len(data["train"]["case_id"])
    chosen = np.sort(
        np.random.default_rng(config["sampling_seed"]).choice(
            n, min(n, config["normalization_sample_cases"]), replace=False
        )
    )
    embeddings = []
    batch = config["inference_batch_size"]
    for offset in range(0, len(chosen), batch):
        z, _ = infer(data["train"], chosen[offset : offset + batch])
        embeddings.append(z)
    sample = np.concatenate(embeddings)
    mean, scale = fit_representation_scaling(sample)
    normalized = (sample - mean) / scale
    centers = {}
    for k in config["state_counts"]:
        with threadpool_limits(limits=8):
            estimator = MiniBatchKMeans(
                n_clusters=k,
                **{key: value for key, value in config["fit"].items() if key != "algorithm"},
            )
            estimator.fit(normalized)
        centers[k] = estimator.cluster_centers_
    count = {k: np.zeros(k, dtype=np.int64) for k in centers}
    total = {k: np.zeros(k) for k in centers}
    teacher_total = 0.0
    for offset in range(0, n, batch):
        z, teacher = infer(data["train"], slice(offset, offset + batch))
        teacher_total += float(teacher.sum())
        for k, c in centers.items():
            state = assign_centers(z, mean, scale, c)
            count[k] += np.bincount(state, minlength=k)
            total[k] += np.bincount(state, weights=teacher, minlength=k)
    emissions = {k: emission_probabilities(count[k], total[k], teacher_total / n) for k in centers}
    val = data["validation"]
    states = {k: [] for k in centers}
    teacher_parts = []
    for offset in range(0, len(val["target"]), batch):
        z, teacher = infer(val, slice(offset, offset + batch))
        teacher_parts.append(teacher)
        for k, c in centers.items():
            states[k].append(assign_centers(z, mean, scale, c))
    teacher = np.concatenate(teacher_parts)
    with np.load(fit / "validation-predictions.npz", allow_pickle=False) as previous:
        if (
            not np.array_equal(previous["case_id"], val["case_id"])
            or not np.array_equal(previous["target"], val["target"])
            or not np.array_equal(previous["user_id"], val["user_id"])
            or not np.allclose(previous["probability"], teacher, atol=1e-6, rtol=0)
        ):
            raise AssertionError("Frozen teacher validation predictions do not reproduce")
        max_reproduction_error = float(np.abs(previous["probability"] - teacher).max())
    outputs, rows = {}, []
    predictions = dict(
        case_id=val["case_id"], user_id=val["user_id"], target=val["target"], teacher=teacher
    )
    for k, c in centers.items():
        state = np.concatenate(states[k])
        forecast = emissions[k][state]
        predictions[f"k{k}_state"] = state
        predictions[f"k{k}_probability"] = forecast
        name = f"k{k}-model.npz"
        np.savez_compressed(
            out / name,
            centers=c,
            mean=mean,
            scale=scale,
            probabilities=emissions[k],
            counts=count[k],
            normalization_sample_indices=chosen,
            normalization_sample_case_id=data["train"]["case_id"][chosen],
        )
        outputs[name] = checksum(out / name)
        rows.append(
            dict(
                nominal_states=k,
                occupied_training_states=int(np.count_nonzero(count[k])),
                empty_training_states=int(np.count_nonzero(count[k] == 0)),
                validation=prediction_scores(val["target"], forecast, val["user_id"]),
            )
        )
    np.savez_compressed(out / "validation-predictions.npz", **predictions)
    outputs["validation-predictions.npz"] = checksum(out / "validation-predictions.npz")
    result = dict(
        status="complete",
        source_digest=digest,
        parent_source_digest=parent_digest,
        parent_fit_report_sha256=fit_sha,
        input_report_sha256=input_sha,
        protocol=protocol,
        architecture=architecture,
        seed=seed,
        phase="engineering context-only quantization"
        if architecture == "context"
        else "training-only quantization and validation; no final test scoring",
        train_cases=n,
        normalization_sample_cases=len(chosen),
        teacher_validation=prediction_scores(val["target"], teacher, val["user_id"]),
        maximum_teacher_reproduction_error=max_reproduction_error,
        models=rows,
        output_sha256=outputs,
        wall_seconds=time.monotonic() - started,
        root=str(out),
        test_predictive_performance="not evaluated",
        scope="Separate smaller quantizers are unverified baselines. K256 is the fixed parent for future nested compression; no final human noninferiority claim.",
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    training_root: str,
    protocol: str = "known_users",
    architecture: str = "context",
    seed: int = 20261015,
):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = snapshot()
    identity = f"{training_root}-{protocol}-{architecture}-{seed}-{digest[:12]}"
    reserved = BudgetLedger(ROOT / ".campaign/calendar-quantization-budget.sqlite", 40).reserve(
        identity, 21600 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    )
    state = ROOT / ".campaign" / ("quantize-" + identity + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = quantize.spawn(training_root, protocol, architecture, seed)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    function_id=quantize.object_id,
                    source_digest=digest,
                    reservation=reserved,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    try:
        print(json.dumps(call.get()), flush=True)
    except TimeoutError:
        print("Observation expired; original quantization call remains recorded", flush=True)
