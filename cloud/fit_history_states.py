"""Private pre-exposure history quantization from frozen completed GRU fits."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-history-states-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1", "pytest==8.4.1"
    )
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_history_state.py", remote_path="/research/tests/test_history_state.py"
    )
    .add_local_file(
        ROOT / "configs/human-history-state-v1.json", remote_path="/research/config.json"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    import json
    import hashlib

    remote = Path("/research/config.json").exists()
    base = Path("/research") if remote else ROOT
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": (
            base / "config.json" if remote else base / "configs/human-history-state-v1.json"
        ).read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    saved["tests/test_history_state.py"] = (base / "tests/test_history_state.py").read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=180, retries=0)
def smoke():
    import subprocess
    import torch
    from behavioral_equivalence.content_neural import build_content_model

    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_history_state.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    torch.manual_seed(17)
    model = build_content_model("gru", 32, 128).eval()
    tokens = torch.ones((3, 32), dtype=torch.long)
    with torch.inference_mode():
        before = model.encode(tokens)
        model(tokens, torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.zeros((3, 2)))
        after = model.encode(tokens)
    assert (
        before.shape == (3, 128)
        and torch.equal(before, after)
        and torch.equal(before[0], before[1])
    )
    return dict(
        status="complete",
        source_digest=snapshot()[1],
        tests=result.stdout,
        history_context_invariance=True,
    )


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=1800,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def fit(training_root: str, protocol: str, seed: int):
    import json
    import hashlib
    import time
    import numpy as np
    import torch
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.history_state import (
        standard_training_sample,
        initial_predictive_blocks,
    )
    from behavioral_equivalence.quantization import (
        fit_representation_scaling,
        assign_centers,
        emission_probabilities,
    )

    if not training_root.replace("-", "").isalnum() or protocol not in {
        "known_users",
        "unseen_users",
    }:
        raise ValueError("Immutable training root and user protocol required")
    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    root = Path("/artifacts") / training_root
    source = json.loads((root / "source.json").read_text())
    parent_digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    configs = [
        json.loads(v) for k, v in source.items() if k.startswith("configs/human-training-calendar")
    ]
    if len(configs) != 1:
        raise ValueError("One parent training configuration required")
    training = configs[0]
    for name in ["neural.py", "exposure_neural.py", "content_neural.py"]:
        key = "src/behavioral_equivalence/" + name
        if source[key] != saved[key]:
            raise ValueError("Frozen neural implementation changed")
    parent = root / "fits" / f"{protocol}-gru-{seed}"
    raw = (parent / "report.json").read_bytes()
    report = json.loads(raw)
    parent_sha = hashlib.sha256(raw).hexdigest()
    if (
        report["status"] != "complete"
        or report["source_digest"] != parent_digest
        or report["protocol"] != protocol
        or report["seed"] != seed
        or report["architecture"] != "gru"
        or seed not in training["seeds"]
        or any(checksum(parent / name) != sha for name, sha in report["output_sha256"].items())
    ):
        raise ValueError("Complete checksum-bound parent GRU fit required")
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    if (
        checksum(inputs / "report.json") != training["input_artifacts"][protocol]["report_sha256"]
        or checksum(inputs / "report.json") != report["input_report_sha256"]
    ):
        raise ValueError("Frozen inputs changed")
    input_report = json.loads((inputs / "report.json").read_text())
    if checksum(inputs / "train.npz") != input_report["output_sha256"]["train.npz"]:
        raise ValueError("Training cases changed")
    cohort = Path("/artifacts") / config["random_cohort_root"]
    if (
        checksum(cohort / "report.json") != config["random_cohort_report_sha256"]
        or checksum(cohort / config["random_audit_relative"]) != config["random_audit_sha256"]
    ):
        raise ValueError("Audited random cohort changed")
    cohort_report = json.loads((cohort / "report.json").read_text())
    random_ids = []
    for bucket in cohort_report["buckets"]:
        path = cohort / f"bucket-{bucket['bucket']:02d}" / "randomized.npz"
        if checksum(path) != bucket["output_sha256"]["randomized.npz"]:
            raise ValueError("Random cohort bucket changed")
        with np.load(path, allow_pickle=False) as arrays:
            random_ids.append(arrays["case_id"])
    with np.load(inputs / "train.npz", allow_pickle=False) as arrays:
        selected = standard_training_sample(
            arrays["case_id"],
            np.concatenate(random_ids),
            config["sample_cases"],
            config["sampling_seed"],
        )
        sample = {
            k: arrays[k][selected]
            for k in ["case_id", "tokens", "tab", "item_id", "intrinsic_context"]
        }
    out = parent / ("history-states-" + digest[:12] + "-" + parent_sha[:12])
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if (
            cached["source_digest"] != digest
            or cached["parent_fit_report_sha256"] != parent_sha
            or checksum(out / "model.npz") != cached["output_sha256"]["model.npz"]
        ):
            raise ValueError("Cached model changed")
        return cached
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    torch.set_num_threads(8)
    model = build_content_model("gru", input_report["vocabulary_size"], training["hidden"])
    model.load_state_dict(torch.load(parent / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    representations = []
    teachers = []
    batch = config["inference_batch_size"]
    for offset in range(0, len(selected), batch):
        sl = slice(offset, offset + batch)
        args = [
            torch.as_tensor(
                sample[k][sl], dtype=torch.float32 if k == "intrinsic_context" else torch.long
            )
            for k in ["tokens", "tab", "item_id", "intrinsic_context"]
        ]
        with torch.inference_mode():
            representations.append(model.encode(args[0]).numpy().astype(np.float64))
            teachers.append(torch.sigmoid(model(*args)).numpy().astype(np.float64))
    z = np.concatenate(representations)
    teacher = np.concatenate(teachers)
    mean, scale = fit_representation_scaling(z)
    with threadpool_limits(limits=8):
        estimator = MiniBatchKMeans(n_clusters=config["state_count"], **config["fit"]).fit(
            (z - mean) / scale
        )
    state = assign_centers(z, mean, scale, estimator.cluster_centers_)
    counts = np.bincount(state, minlength=config["state_count"])
    totals = np.bincount(state, weights=teacher, minlength=config["state_count"])
    probabilities = emission_probabilities(counts, totals, float(teacher.mean()))
    blocks = initial_predictive_blocks(probabilities, config["initial_block_count"])
    np.savez_compressed(
        out / "model.npz",
        mean=mean,
        scale=scale,
        centers=estimator.cluster_centers_,
        sample_case_id=sample["case_id"],
        sample_state=state,
        sample_teacher_mean=probabilities,
        sample_state_counts=counts,
    )
    result = dict(
        status="complete",
        source_digest=digest,
        parent_fit_report_sha256=parent_sha,
        protocol=protocol,
        seed=seed,
        sampled_standard_training_cases=len(selected),
        occupied_sample_states=int(np.count_nonzero(counts)),
        initial_blocks=blocks,
        random_cohort_report_sha256=config["random_cohort_report_sha256"],
        output_sha256={"model.npz": checksum(out / "model.npz")},
        wall_seconds=time.monotonic() - started,
        root=str(out),
        scope=config["scope"],
        final_test_scoring="none",
        provider_inference_calls=0,
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    stage: str = "smoke",
    training_root: str = "",
    protocol: str = "known_users",
    seed: int = 20261015,
):
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    if stage not in {"smoke", "fit"}:
        raise ValueError("Smoke or fit stage required")
    digest = snapshot()[1]
    identity = hashlib.sha256(
        json.dumps([stage, digest, training_root, protocol, seed]).encode()
    ).hexdigest()
    amount = 0.03 if stage == "smoke" else 1800 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    reservation = BudgetLedger(ROOT / ".campaign/history-state-budget.sqlite", 4).reserve(
        identity, amount
    )
    state = ROOT / ".campaign" / ("history-state-" + identity[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = smoke.spawn() if stage == "smoke" else fit.spawn(training_root, protocol, seed)
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
