"""Emission-aware point models from frozen encoders and true adjacent bundles."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-calibrated-parent-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "8"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-calibrated-parent-v1.json", remote_path="/research/parent-config.json"
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
    import hashlib
    import json

    base = Path("/research") if Path("/research/parent-config.json").exists() else ROOT
    config = base / (
        "parent-config.json"
        if base == Path("/research")
        else "configs/human-calibrated-parent-v1.json"
    )
    saved = {
        "estimate_calendar_parent.py": Path(__file__).read_text(),
        "config.json": config.read_text(),
    }
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "item_context.py",
        "content_scaling.py",
        "quantization.py",
        "latent_estimation.py",
        "decision_model.py",
        "core.py",
        "parent_identity.py",
    ]:
        saved[name] = (base / "src/behavioral_equivalence" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=10800,
    max_containers=2,
    retries=0,
    volumes={"/artifacts": volume},
)
def estimate(
    training_root: str, protocol: str, architecture: str, seed: int, quantization_name: str
):
    import hashlib
    import json
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.item_context import encode_items
    from behavioral_equivalence.content_scaling import scale_duration
    from behavioral_equivalence.quantization import assign_centers
    from behavioral_equivalence.latent_estimation import (
        adjacent_transition_counts,
        point_transitions,
        equal_user_initial_mass,
    )
    from behavioral_equivalence.decision_model import decision_chain

    if (
        not training_root.replace("-", "").isalnum()
        or not quantization_name.replace("-", "").isalnum()
        or protocol not in {"known_users", "unseen_users"}
        or architecture not in {"context", "gru"}
    ):
        raise ValueError("Known immutable training, quantization and protocol names required")
    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    config = json.loads(Path("/research/parent-config.json").read_text())
    root = Path("/artifacts") / training_root
    parent_source = json.loads((root / "source.json").read_text())
    source_digest = hashlib.sha256(json.dumps(parent_source, sort_keys=True).encode()).hexdigest()
    training_configs = [
        json.loads(text)
        for name, text in parent_source.items()
        if name.startswith("configs/human-training-calendar")
    ]
    if len(training_configs) != 1:
        raise ValueError("One bound training configuration required")
    training = training_configs[0]
    fit = root / "fits" / f"{protocol}-{architecture}-{seed}"
    fit_raw = (fit / "report.json").read_bytes()
    fit_report = json.loads(fit_raw)
    quant = fit / quantization_name
    quant_raw = (quant / "report.json").read_bytes()
    quant_report = json.loads(quant_raw)
    if (
        fit_report["status"] != "complete"
        or fit_report["source_digest"] != source_digest
        or quant_report["status"] != "complete"
        or quant_report["parent_fit_report_sha256"] != hashlib.sha256(fit_raw).hexdigest()
        or any(checksum(fit / name) != sha for name, sha in fit_report["output_sha256"].items())
        or any(checksum(quant / name) != sha for name, sha in quant_report["output_sha256"].items())
    ):
        raise ValueError("Completed fit or training-only quantizer changed")
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "content_scaling.py",
        "item_context.py",
    ]:
        if (
            parent_source["src/behavioral_equivalence/" + name]
            != (Path("/research/src/behavioral_equivalence") / name).read_text()
        ):
            raise ValueError("Inference implementation differs from frozen training")
    quant_source = json.loads((quant / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(quant_source, sort_keys=True).encode()).hexdigest()
        != quant_report["source_digest"]
        or quant_source["src/behavioral_equivalence/quantization.py"] != saved["quantization.py"]
    ):
        raise ValueError("Quantization source or assignment implementation changed")
    data = Path("/artifacts") / config["calibration_root"]
    if checksum(data / "report.json") != config["calibration_report_sha256"]:
        raise ValueError("Adjacent calibration dataset changed")
    calibration = json.loads((data / "report.json").read_text())
    if calibration["status"] != "complete" or len(calibration["buckets"]) != 32:
        raise ValueError("All calibration buckets required")
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    if checksum(inputs / "report.json") != fit_report["input_report_sha256"]:
        raise ValueError("Training normalization source changed")
    input_report = json.loads((inputs / "report.json").read_text())
    for name in ["duration-scaling.json", "item-vocabulary.npy"]:
        if checksum(inputs / name) != input_report["output_sha256"][name]:
            raise ValueError("Saved input vocabulary or normalization changed")
    out = quant / ("calibrated-parent-" + digest[:12])
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if cached["source_digest"] != digest or any(
            checksum(out / name) != sha for name, sha in cached["output_sha256"].items()
        ):
            raise ValueError("Cached calibrated model changed")
        return cached
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    volume.commit()
    torch.set_num_threads(8)
    model = build_content_model(architecture, input_report["vocabulary_size"], training["hidden"])
    model.load_state_dict(torch.load(fit / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    vocabulary = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    quantizers = {}
    for k in config["state_counts"]:
        with np.load(quant / f"k{k}-model.npz", allow_pickle=False) as q:
            quantizers[k] = {
                name: q[name] for name in ["centers", "mean", "scale", "probabilities"]
            }
    counts = {k: np.zeros((k, 2, k), dtype=np.int64) for k in quantizers}
    mass = {k: np.zeros(k) for k in quantizers}
    users_seen = set()
    outputs = {}
    support = []
    for bucket in calibration["buckets"]:
        path = data / f"bucket-{bucket['bucket']:02d}" / "calibration.npz"
        if checksum(path) != bucket["output_sha256"]["calibration.npz"]:
            raise ValueError("Adjacent calibration bucket changed")
        with np.load(path, allow_pickle=False) as arrays:
            a = {name: arrays[name] for name in arrays.files}
        chosen = (
            np.arange(len(a["target"]))
            if protocol == "known_users"
            else np.flatnonzero(a["user_role_code"] == 2)
        )
        pair_mask = (
            np.ones(len(a["pair_current"]), dtype=bool)
            if protocol == "known_users"
            else a["pair_user_role_code"] == 2
        )
        current = np.searchsorted(chosen, a["pair_current"][pair_mask])
        following = np.searchsorted(chosen, a["pair_next"][pair_mask])
        if (
            not np.array_equal(chosen[current], a["pair_current"][pair_mask])
            or not np.array_equal(chosen[following], a["pair_next"][pair_mask])
            or not np.all(a["role_code"][chosen] == 2)
        ):
            raise AssertionError("Pair endpoints escaped the calibration/user cohort")
        user = a["user_id"][chosen]
        origin_users = set(map(int, np.unique(user[current])))
        if origin_users & users_seen:
            raise AssertionError("User crosses prepared bucket boundaries")
        users_seen.update(origin_users)
        encoded_items = encode_items(vocabulary, a["video_id"][chosen])
        context = scale_duration(a["intrinsic_context"][chosen], scaling)
        states = {k: [] for k in quantizers}
        batch = config["inference_batch_size"]
        for offset in range(0, len(chosen), batch):
            section = slice(offset, offset + batch)
            idx = chosen[section]
            args = [
                torch.as_tensor(a["tokens"][idx], dtype=torch.long),
                torch.as_tensor(a["tab"][idx], dtype=torch.long),
                torch.as_tensor(encoded_items[section], dtype=torch.long),
                torch.as_tensor(context[section], dtype=torch.float32),
            ]
            with torch.inference_mode():
                z = model.output[:-1](model.representation(*args)).numpy()
            for k, q in quantizers.items():
                states[k].append(assign_centers(z, q["mean"], q["scale"], q["centers"]))
        stored = dict(
            source_query_indices=chosen,
            case_id=a["case_id"][chosen],
            pair_current=current,
            pair_next=following,
        )
        for k in quantizers:
            state = np.concatenate(states[k]) if len(chosen) else np.empty(0, dtype=np.int64)
            stored[f"k{k}_state"] = state
            counts[k] += adjacent_transition_counts(
                state, a["target"][chosen], user, a["bundle_index"][chosen], current, following, k
            )
            value, n = equal_user_initial_mass(state[current], user[current], k)
            if n != len(origin_users):
                raise AssertionError("User-weight normalization differs")
            mass[k] += value
        name = f"bucket-{bucket['bucket']:02d}-states.npz"
        np.savez_compressed(out / name, **stored)
        outputs[name] = checksum(out / name)
        support.append(
            dict(
                bucket=bucket["bucket"],
                queries=len(chosen),
                transitions=len(current),
                users=len(origin_users),
            )
        )
        volume.commit()
    if not users_seen:
        raise ValueError("At least one supported calibration user required")
    models = []
    for k, q in quantizers.items():
        transition = point_transitions(counts[k], config["transition_prior_mass_per_outcome_row"])
        emission = np.column_stack([1 - q["probabilities"], q["probabilities"]])
        initial = mass[k] / len(users_seen)
        parent = decision_chain(transition, emission, initial)
        name = f"k{k}-parent.npz"
        np.savez_compressed(
            out / name,
            transitions=transition,
            emissions=emission,
            initial=initial,
            counts=counts[k],
        )
        outputs[name] = checksum(out / name)
        models.append(
            dict(
                latent_states=k,
                labeled_states=parent.size,
                chain_digest=parent.digest(),
                transition_pairs=int(counts[k].sum()),
                unsupported_state_outcome_rows=int(np.count_nonzero(counts[k].sum(axis=2) == 0)),
                nonzero_graph_edges=int(np.count_nonzero(parent.transitions)),
            )
        )
    result = dict(
        status="complete",
        source_digest=digest,
        parent_source_digest=source_digest,
        quantization_report_sha256=hashlib.sha256(quant_raw).hexdigest(),
        calibration_report_sha256=config["calibration_report_sha256"],
        phase="engineering context-only point models"
        if architecture == "context"
        else "frozen-encoder adjacent calibrated point models",
        protocol=protocol,
        architecture=architecture,
        seed=seed,
        users=len(users_seen),
        support=support,
        models=models,
        output_sha256=outputs,
        wall_seconds=time.monotonic() - started,
        root=str(out),
        test_predictive_performance="not evaluated",
        scope="Emissions distill training teacher probabilities; outcome-conditioned transitions use real adjacent calibration bundles. The additive prior and Markov representation specify a point model, not a human confidence set. Positive/negative labels are nonterminal outcomes, not completion or abandonment.",
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    training_root: str = "human-calendar-training-v3-8c2afe99bb75",
    protocol: str = "known_users",
    architecture: str = "context",
    seed: int = 20261015,
    quantization_name: str = "quantization-eaa699cb5987-3cac5acd2e90",
):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger
    from behavioral_equivalence.parent_identity import parent_job_identity

    _, digest = snapshot()
    identity = parent_job_identity(
        training_root, protocol, architecture, seed, quantization_name, digest
    )
    reserved = BudgetLedger(ROOT / ".campaign/calibrated-parent-budget.sqlite", 25).reserve(
        identity, 10800 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    )
    state = ROOT / ".campaign" / ("parent-" + identity + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = estimate.spawn(training_root, protocol, architecture, seed, quantization_name)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    function_id=estimate.object_id,
                    source_digest=digest,
                    reservation=reserved,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
