"""Raw-case validation inference roundtrip, without final test scoring."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-case-inference-verification-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1", "pytest==8.4.1"
    )
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_case_inference.py", remote_path="/research/tests/test_case_inference.py"
    )
    .add_local_file(
        ROOT / "configs/case-inference-verification-v1.json", remote_path="/research/config.json"
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
            else base / "configs/case-inference-verification-v1.json"
        ).read_text(),
        "test_case_inference.py": (base / "tests/test_case_inference.py").read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=1200,
    retries=0,
    volumes={"/artifacts": volume},
)
def verify():
    import json
    import hashlib
    import subprocess
    import numpy as np
    import torch
    from behavioral_equivalence.case_inference import forecast_indices, available_history_lengths
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.item_context import encode_items
    from behavioral_equivalence.content_scaling import scale_duration

    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    if config["split"] != "validation":
        raise ValueError("This verification never opens final test predictions")
    tests = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_case_inference.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    volume.reload()
    root = Path("/artifacts") / config["training_root"]
    parent_source = json.loads((root / "source.json").read_text())
    parent_digest = hashlib.sha256(json.dumps(parent_source, sort_keys=True).encode()).hexdigest()
    training = next(
        json.loads(v)
        for k, v in parent_source.items()
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
        if saved[key] != parent_source[key]:
            raise ValueError("Frozen model or preprocessing implementation changed")
    fit = root / "fits" / f"{config['protocol']}-{config['architecture']}-{config['seed']}"
    if checksum(fit / "report.json") != config["fit_report_sha256"]:
        raise ValueError("Pinned completed fit required")
    fit_report = json.loads((fit / "report.json").read_text())
    if (
        fit_report["status"] != "complete"
        or fit_report["source_digest"] != parent_digest
        or any(checksum(fit / name) != sha for name, sha in fit_report["output_sha256"].items())
    ):
        raise ValueError("Frozen model outputs changed")
    inputs = Path("/artifacts") / training["input_artifacts"][config["protocol"]]["path"]
    if (
        checksum(inputs / "report.json")
        != training["input_artifacts"][config["protocol"]]["report_sha256"]
        or checksum(inputs / "report.json") != fit_report["input_report_sha256"]
    ):
        raise ValueError("Training-only preprocessing source changed")
    input_report = json.loads((inputs / "report.json").read_text())
    for name in ["item-vocabulary.npy", "duration-scaling.json"]:
        if checksum(inputs / name) != input_report["output_sha256"][name]:
            raise ValueError("Frozen preprocessing artifact changed")
    vocabulary = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    cases = Path("/artifacts") / training["case_root"]
    intrinsic = Path("/artifacts") / training["context_root"]
    if (
        checksum(cases / "report.json") != training["case_report_sha256"]
        or checksum(cases / training["case_audit_relative"]) != training["case_audit_sha256"]
        or checksum(intrinsic / "report.json") != training["context_report_sha256"]
    ):
        raise ValueError("Independently audited raw cases/context changed")
    case_report = json.loads((cases / "report.json").read_text())
    context_report = json.loads((intrinsic / "report.json").read_text())
    torch.set_num_threads(8)
    model = build_content_model(
        config["architecture"], input_report["vocabulary_size"], training["hidden"]
    )
    model.load_state_dict(torch.load(fit / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    pieces = []
    for bucket in case_report["buckets"]:
        name = f"embargo-{training['embargo_hours']:02d}h.npz"
        path = cases / f"bucket-{bucket['bucket']:02d}" / name
        context_path = intrinsic / f"bucket-{bucket['bucket']:02d}" / name
        if (
            checksum(path) != bucket["output_sha256"][name]
            or checksum(context_path)
            != context_report["buckets"][bucket["bucket"]]["output_sha256"][name]
        ):
            raise ValueError("Audited raw case or context bucket changed")
        with np.load(context_path, allow_pickle=False) as a:
            context_ids, context_values = a["case_id"], a["context"]
        with np.load(path, allow_pickle=False) as a:
            selected = forecast_indices(
                a["case_id"],
                a["user_id"],
                a["role_code"],
                a["user_role_code"],
                context_ids,
                config["protocol"],
                "validation",
            )
            part = {
                key: a[key][selected] for key in ["case_id", "user_id", "tokens", "tab", "video_id"]
            }
        history_lengths = available_history_lengths(part["tokens"])
        items = encode_items(vocabulary, part["video_id"])
        context = scale_duration(context_values[selected], scaling)
        probabilities = []
        batch = config["inference_batch_size"]
        for offset in range(0, len(selected), batch):
            sl = slice(offset, offset + batch)
            args = [
                torch.as_tensor(x[sl], dtype=torch.float32 if i == 3 else torch.long)
                for i, x in enumerate([part["tokens"], part["tab"], items, context])
            ]
            with torch.inference_mode():
                probabilities.append(torch.sigmoid(model(*args)).numpy())
        pieces.append(
            dict(
                case_id=part["case_id"],
                user_id=part["user_id"],
                history_length=history_lengths,
                probability=np.concatenate(probabilities) if probabilities else np.empty(0),
            )
        )
    result_arrays = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
    with np.load(fit / "validation-predictions.npz", allow_pickle=False) as previous:
        if any(
            not np.array_equal(previous[key], result_arrays[key]) for key in ["case_id", "user_id"]
        ):
            raise ValueError("Reconstructed forecast identities/order differ")
        error = float(np.max(np.abs(previous["probability"] - result_arrays["probability"])))
        if error > config["probability_absolute_tolerance"]:
            raise ValueError("Raw-case validation forecasts do not reproduce")
    out = fit / ("case-inference-verification-" + digest[:16])
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "validation-forecasts.npz", **result_arrays)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    lengths = result_arrays["history_length"]
    report = dict(
        status="complete",
        source_digest=digest,
        fit_report_sha256=config["fit_report_sha256"],
        protocol=config["protocol"],
        architecture=config["architecture"],
        seed=config["seed"],
        tests=tests.stdout,
        cases=len(lengths),
        users=len(np.unique(result_arrays["user_id"])),
        maximum_validation_forecast_reproduction_error=error,
        history_cohort_cases=dict(
            empty=int((lengths == 0).sum()),
            one_to_eight=int(((lengths >= 1) & (lengths <= 8)).sum()),
            nine_to_sixteen=int(((lengths >= 9) & (lengths <= 16)).sum()),
            seventeen_to_thirtytwo=int((lengths >= 17).sum()),
        ),
        output_sha256={"validation-forecasts.npz": checksum(out / "validation-forecasts.npz")},
        target_fields_accessed=False,
        final_test_predictions_generated=0,
        scope=config["scope"],
        root=str(out),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    digest = snapshot()[1]
    reservation = BudgetLedger(
        ROOT / ".campaign/case-inference-verification-budget.sqlite", 0.5
    ).reserve(digest, 1200 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1)
    state = ROOT / ".campaign" / ("case-inference-verification-" + digest[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = verify.spawn()
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
