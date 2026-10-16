"""Training-only teacher priors for calibration-group holdout experiments."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-heldout-teacher-priors-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1", "pytest==8.4.1"
    )
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_heldout_response.py",
        remote_path="/research/tests/test_heldout_response.py",
    )
    .add_local_file(
        ROOT / "configs/human-heldout-response-v1.json", remote_path="/research/config.json"
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
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": (
            base / "config.json"
            if base == Path("/research")
            else base / "configs/human-heldout-response-v1.json"
        ).read_text(),
        "test_heldout_response.py": (base / "tests/test_heldout_response.py").read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=180, retries=0)
def smoke():
    import subprocess

    r = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_heldout_response.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(status="complete", source_digest=snapshot()[1], tests=r.stdout)


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=900,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def fit(seed: int):
    import json
    import hashlib
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.heldout_response import teacher_response_prior

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    spec = next(s for s in config["inputs"] if s["seed"] == seed)
    history = Path("/artifacts") / spec["history_root"]
    if checksum(history / "report.json") != spec["history_report_sha256"]:
        raise ValueError("History report changed")
    h = json.loads((history / "report.json").read_text())
    hs = json.loads((history / "source.json").read_text())
    if (
        h["status"] != "complete"
        or h["seed"] != seed
        or h["protocol"] != "known_users"
        or hashlib.sha256(json.dumps(hs, sort_keys=True).encode()).hexdigest() != h["source_digest"]
        or checksum(history / "model.npz") != h["output_sha256"]["model.npz"]
    ):
        raise ValueError("Completed frozen history model required")
    parent = history.parent
    if checksum(parent / "report.json") != h["parent_fit_report_sha256"]:
        raise ValueError("Parent fit changed")
    pr = json.loads((parent / "report.json").read_text())
    ts = json.loads((parent.parents[1] / "source.json").read_text())
    training = next(
        json.loads(v) for k, v in ts.items() if k.startswith("configs/human-training-calendar")
    )
    if hashlib.sha256(json.dumps(ts, sort_keys=True).encode()).hexdigest() != pr[
        "source_digest"
    ] or any(checksum(parent / name) != sha for name, sha in pr["output_sha256"].items()):
        raise ValueError("Parent source or weights changed")
    for name in ["neural.py", "exposure_neural.py", "content_neural.py"]:
        key = "src/behavioral_equivalence/" + name
        if saved[key] != hs[key] or saved[key] != ts[key]:
            raise ValueError("Frozen teacher implementation changed")
    inputs = Path("/artifacts") / training["input_artifacts"]["known_users"]["path"]
    if (
        checksum(inputs / "report.json")
        != training["input_artifacts"]["known_users"]["report_sha256"]
        or checksum(inputs / "report.json") != pr["input_report_sha256"]
    ):
        raise ValueError("Training-only input report changed")
    ir = json.loads((inputs / "report.json").read_text())
    for name in ["train.npz", "duration-scaling.json"]:
        if checksum(inputs / name) != ir["output_sha256"][name]:
            raise ValueError("Training features/scaling changed")
    support = Path("/artifacts") / spec["support_root"]
    if checksum(support / "report.json") != spec["support_report_sha256"]:
        raise ValueError("Training-rank state map support report changed")
    sr = json.loads((support / "report.json").read_text())
    if (
        sr["history_report_sha256"] != spec["history_report_sha256"]
        or checksum(support / "support.npz") != sr["output_sha256"]["support.npz"]
    ):
        raise ValueError("History coarsening map changed")
    with np.load(support / "support.npz", allow_pickle=False) as a:
        mapping = a["k32_map"]
    with np.load(history / "model.npz", allow_pickle=False) as a:
        ids = a["sample_case_id"]
        states = a["sample_state"]
        previous_counts = a["sample_state_counts"]
        previous_means = a["sample_teacher_mean"]
    with np.load(inputs / "train.npz", allow_pickle=False) as a:
        original = a["case_id"]
        order = np.argsort(original)
        positions = np.searchsorted(original[order], ids)
        if np.any(positions >= len(original)) or not np.array_equal(
            original[order[positions]], ids
        ):
            raise ValueError("Frozen standard-only sample escaped training inputs")
        selected = order[positions]
        sample = {
            key: a[key][selected]
            for key in ["tokens", "tab", "item_id", "intrinsic_context", "video_id"]
        }
    metadata = Path("/artifacts") / config["metadata_root"]
    if checksum(metadata / "report.json") != config["metadata_report_sha256"]:
        raise ValueError("Intrinsic metadata report changed")
    mr = json.loads((metadata / "report.json").read_text())
    if checksum(metadata / "duration-ms.npy") != mr["output_sha256"]["duration-ms.npy"]:
        raise ValueError("Duration metadata changed")
    duration = np.load(metadata / "duration-ms.npy", mmap_mode="r", allow_pickle=False)[
        sample["video_id"]
    ]
    known = sample["intrinsic_context"][:, 1] == 0
    groups = np.full(len(ids), 4, dtype=np.int64)
    if np.any(~np.isfinite(duration[known])) or np.any(duration[known] <= 0):
        raise ValueError("Known training context lacks usable metadata")
    groups[known] = np.searchsorted([7000, 18000, 60000], duration[known], side="left")
    environment = sample["tab"].astype(np.int64) * 5 + groups
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    reconstructed = (
        sample["intrinsic_context"][known, 0].astype(float) * scaling["scale"] + scaling["mean"]
    )
    if not np.allclose(reconstructed, np.log1p(duration[known] / 1000), atol=1e-5, rtol=0):
        raise ValueError("Intrinsic group metadata disagrees with training features")
    model = build_content_model("gru", ir["vocabulary_size"], training["hidden"])
    model.load_state_dict(torch.load(parent / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    torch.set_num_threads(8)
    parts = []
    for offset in range(0, len(ids), 2048):
        sl = slice(offset, offset + 2048)
        args = [
            torch.as_tensor(
                sample[key][sl], dtype=torch.float32 if key == "intrinsic_context" else torch.long
            )
            for key in ["tokens", "tab", "item_id", "intrinsic_context"]
        ]
        with torch.inference_mode():
            parts.append(torch.sigmoid(model(*args)).numpy().astype(float))
    teacher = np.concatenate(parts)
    counts = np.bincount(states, minlength=256)
    totals = np.bincount(states, weights=teacher, minlength=256)
    means = np.full(256, float(teacher.mean()))
    np.divide(totals, counts, out=means, where=counts > 0)
    error = float(np.max(np.abs(means - previous_means)))
    if not np.array_equal(counts, previous_counts) or error > 1e-6:
        raise ValueError("Frozen sample teacher forecasts do not reproduce")
    fine = teacher_response_prior(states, environment, teacher, 256)
    coarse = teacher_response_prior(mapping[states], environment, teacher, 32)
    out = history / ("heldout-teacher-prior-" + digest[:16])
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "priors.npz",
        fine_probabilities=fine["probabilities"],
        coarse_probabilities=coarse["probabilities"],
        coarse_counts=coarse["counts"],
        fine_counts=fine["counts"],
        fine_to_coarse=mapping,
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    report = dict(
        status="complete",
        source_digest=digest,
        seed=seed,
        protocol="known_users",
        history_report_sha256=spec["history_report_sha256"],
        sample_standard_training_cases=len(ids),
        maximum_sample_teacher_mean_reproduction_error=error,
        known_duration_sample_cases=int(known.sum()),
        unknown_duration_sample_cases=int((~known).sum()),
        output_sha256={"priors.npz": checksum(out / "priors.npz")},
        root=str(out),
        random_calibration_responses_used=False,
        validation_or_test_responses_used=False,
        scope=config["teacher_prior"],
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(stage: str = "smoke", seed: int = 20261015):
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    if stage not in {"smoke", "fit"}:
        raise ValueError("Smoke or fit stage required")
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    if seed not in [r["seed"] for r in config["inputs"]]:
        raise ValueError("One fixed known-user fit required")
    identity = hashlib.sha256(json.dumps([stage, digest, seed]).encode()).hexdigest()
    amount = 0.03 if stage == "smoke" else 900 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    reservation = BudgetLedger(ROOT / ".campaign/heldout-response-budget.sqlite", 4).reserve(
        identity, amount
    )
    state = ROOT / ".campaign" / ("heldout-prior-" + identity[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = smoke.spawn() if stage == "smoke" else fit.spawn(seed)
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
