"""Nested verified compression of learned human pilot states, not final evaluation."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-learned-certified-pilot-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "polars==1.31.0")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(12288, 12288),
    timeout=1800,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run(source_run: str) -> dict:
    import gzip
    import hashlib
    import json
    import time
    import numpy as np
    import polars as pl
    import torch
    from behavioral_equivalence.features import bundle_history_tokens
    from behavioral_equivalence.neural import SequencePredictor
    from behavioral_equivalence.decision_model import decision_chain
    from behavioral_equivalence.learned_abstraction import (
        certified_latent_refinement,
        next_positive_probabilities,
    )
    from behavioral_equivalence.integer_bounds import verify_abstraction
    from behavioral_equivalence.statistics import paired_user_nll

    if not source_run.replace("-", "").isalnum():
        raise ValueError("Invalid source run")
    root = Path("/artifacts") / source_run
    source = json.loads((root / "report.json").read_text())
    out = root / "learned-certified-pilot-v1"
    out.mkdir(exist_ok=False)
    snapshots = out / "source"
    snapshots.mkdir()
    files = sorted(Path("/research/src/behavioral_equivalence").glob("*.py"))
    for path in files:
        (snapshots / path.name).write_bytes(path.read_bytes())
    (snapshots / "learned_pilot.py").write_bytes(Path(__file__).read_bytes())
    for name in ["features.py", "neural.py"]:
        if (root / "source" / name).read_bytes() != (snapshots / name).read_bytes():
            raise ValueError("Teacher feature/architecture differs from attested training")
    volume.commit()
    start = time.monotonic()
    torch.set_num_threads(2)
    data = Path("/artifacts/datasets/kuairand-1k/prepared-v4")
    if (
        json.loads((data / "report.json").read_text())["output_sha256"]
        != source["prepared_source_sha256"]
    ):
        raise ValueError("Human source differs from baseline training")
    bundles = pl.read_parquet(data / "timestamp-bundles.parquet").sort(["user_id", "time_ms"])
    users = bundles["user_id"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    feature_parts = []
    for left, right in zip(boundaries, boundaries[1:]):
        b = bundles.slice(int(left), int(right - left))
        feature_parts.append(
            bundle_history_tokens(
                b["time_ms"].to_numpy(),
                b["available_ms"].to_numpy(),
                b["clicks"].to_numpy(),
                b["exposures"].to_numpy(),
            )
        )
    features = np.concatenate(feature_parts)
    indices = bundles.select(["user_id", "time_ms"]).with_row_index("feature_id")
    events = (
        pl.read_parquet(
            data / "interactions.parquet",
            columns=["user_id", "time_ms", "tab", "is_click", "chronological_split", "user_split"],
        )
        .with_row_index("case_id")
        .join(indices, on=["user_id", "time_ms"], how="left", validate="m:1")
        .sort("case_id")
    )
    feature_id = events["feature_id"].to_numpy()
    tabs = events["tab"].to_numpy()
    targets = events["is_click"].to_numpy()
    _, begins, counts = np.unique(feature_id, return_index=True, return_counts=True)
    if not np.all(feature_id[begins] == np.arange(len(bundles))):
        raise ValueError("Representative sampling requires sorted contiguous timestamp bundles")
    # Uniform independent index per bundle. No feedback value enters the draw.
    rng = np.random.default_rng(20261016)
    representative = begins + rng.integers(0, counts)
    np.save(out / "representative-case-indices.npy", representative)
    records = []
    for protocol, split_column in [
        ("chronological", "chronological_split"),
        ("users", "user_split"),
    ]:
        saved = np.load(root / f"{protocol}-predictions.npz")
        cases = np.load(root / f"{protocol}-case-indices.npz")
        quantized = np.load(root / "quantization-pilot-v3-cpu" / f"{protocol}-k256-state-model.npz")
        centers = torch.as_tensor(quantized["centers"], dtype=torch.float32)
        mean = torch.as_tensor(quantized["mean"], dtype=torch.float32)
        scale = torch.as_tensor(quantized["scale"], dtype=torch.float32)
        model = SequencePredictor("gru")
        model.load_state_dict(
            torch.load(root / f"{protocol}-gru.pt", weights_only=True, map_location="cpu")
        )
        model.eval()
        split = events[split_column].to_numpy()[representative]
        eligible_pairs = np.flatnonzero(
            (split[:-1] == "calibration") & (split[1:] == "calibration") & (users[:-1] == users[1:])
        )
        calibration_bundles = np.unique(np.r_[eligible_pairs, eligible_pairs + 1])
        calibration_cases = representative[calibration_bundles]

        def infer(selected):
            states, prediction = [], []
            for offset in range(0, len(selected), 1024):
                batch = selected[offset : offset + 1024]
                with torch.no_grad():
                    z = torch.cat(
                        [
                            model.encode(
                                torch.as_tensor(features[feature_id[batch]], dtype=torch.long)
                            ),
                            model.context(torch.as_tensor(tabs[batch], dtype=torch.long)),
                        ],
                        dim=1,
                    )
                    normalized = (z - mean) / scale
                    # Direct squared distances avoid reduced precision or an
                    # implicit change in the saved centroid assignment metric.
                    distance = (normalized[:, None, :] - centers[None, :, :]).square().sum(dim=2)
                    states.append(distance.argmin(dim=1).numpy())
                    prediction.append(torch.sigmoid(model.output(z).squeeze(-1)).numpy())
            return np.concatenate(states), np.concatenate(prediction)

        calibration_state, _ = infer(calibration_cases)
        state_by_bundle = np.full(len(bundles), -1)
        state_by_bundle[calibration_bundles] = calibration_state
        transition_counts = np.zeros((256, 2, 256), dtype=np.int64)
        np.add.at(
            transition_counts,
            (
                state_by_bundle[eligible_pairs],
                targets[representative[eligible_pairs]],
                state_by_bundle[eligible_pairs + 1],
            ),
            1,
        )
        transition = transition_counts.astype(float) + 1 / 256
        transition /= transition.sum(axis=2, keepdims=True)
        emission = np.column_stack([1 - quantized["probabilities"], quantized["probabilities"]])
        initial = np.zeros(256)
        initial[0] = 1
        parent = decision_chain(transition, emission, initial)
        test_state, teacher = infer(cases["test"])
        if (
            not np.array_equal(events["case_id"].to_numpy()[cases["test"]], saved["test_case_id"])
            or not np.array_equal(targets[cases["test"]], saved["test_target"])
            or not np.allclose(teacher, saved["gru_test"], atol=1e-6, rtol=0)
        ):
            raise AssertionError("Teacher reproduction or identical-case alignment failed")
        reference = quantized["probabilities"][test_state]
        prior_predictions = np.load(
            root / "quantization-pilot-v3-cpu" / f"{protocol}-test-predictions.npz"
        )
        if not np.allclose(reference, prior_predictions["k256"], atol=1e-12, rtol=0):
            raise AssertionError("Saved K256 assignment metric does not reproduce")
        np.savez_compressed(
            out / f"{protocol}-parent-model.npz",
            transitions=transition,
            emissions=emission,
            counts=transition_counts,
        )
        order = np.argsort(quantized["probabilities"], kind="stable")
        for candidate_k in [64, 128]:
            candidate = tuple(
                tuple(int(s) for s in part) for part in np.array_split(order, candidate_k)
            )
            for epsilon in [0.005, 0.03, 0.1]:
                began = time.monotonic()
                abstract, partition, evidence = certified_latent_refinement(
                    parent, candidate, epsilon
                )
                verify_abstraction(parent, abstract, evidence["certificate"])
                forecasts = next_positive_probabilities(abstract, partition)[test_state]
                paired = paired_user_nll(
                    saved["test_target"],
                    reference,
                    forecasts,
                    saved["test_user_id"],
                    replications=2000,
                    seed=20261015,
                )
                packed = gzip.compress(json.dumps(evidence, sort_keys=True).encode(), mtime=0)
                name = f"{protocol}-k{candidate_k}-eps{epsilon}.json.gz"
                (out / name).write_bytes(packed)
                record = {
                    "protocol": protocol,
                    "candidate_k": candidate_k,
                    "epsilon_budget": epsilon,
                    "reduced_latent_states": len(partition),
                    "compression_ratio": 256 / len(partition),
                    "parent_labeled_states": parent.size,
                    "abstract_labeled_states": abstract.size,
                    "calibration_pairs": len(eligible_pairs),
                    "zero_count_transition_rows": int((transition_counts.sum(axis=2) == 0).sum()),
                    "transition_prior": "one total uniform Dirichlet pseudocount per state/outcome row",
                    "paired_test_nll": paired,
                    "certificate": name,
                    "certificate_sha256": hashlib.sha256(packed).hexdigest(),
                    "seconds": time.monotonic() - began,
                }
                records.append(record)
                (out / "progress.json").write_text(json.dumps(records, indent=2))
                volume.commit()
                print(
                    json.dumps(
                        {
                            "protocol": protocol,
                            "candidate_k": candidate_k,
                            "epsilon": epsilon,
                            "reduced_states": len(partition),
                        }
                    ),
                    flush=True,
                )
    report = {
        "status": "complete",
        "phase": "exploratory 1K learned certified compression pilot",
        "source_run": source_run,
        "records": records,
        "wall_seconds": time.monotonic() - start,
        "limitations": [
            "Certificates apply to estimated point models, not the unknown human process.",
            "Within-user transition dependence does not satisfy fixed-iid-row confidence assumptions.",
            "One fitted seed, exploratory epsilon choices, pilot timing proxies, no final 27K evaluation.",
            "One uniform representative exposure per timestamp bundle defines the calibration transition estimand.",
            "Large error tolerances can produce vacuous long-horizon bounds; report the full frontier.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {
        "status": "complete",
        "records": len(records),
        "report": str(out / "report.json"),
        "wall_seconds": report["wall_seconds"],
    }


@app.local_entrypoint()
def main(source_run: str = "human-training-pilot-20261015-b"):
    import json

    call = run.spawn(source_run)
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": run.object_id}), flush=True
    )
    print(json.dumps(call.get(), indent=2), flush=True)
