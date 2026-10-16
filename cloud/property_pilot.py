"""Finite-property compression of the same learned pilot parent models.

This is an explicit alternative to bisimulation, not a replacement of its
negative result or a confirmatory test on the final human holdout.
"""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-property-compression-pilot-v1")
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
    timeout=3600,
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
    from behavioral_equivalence.decision_model import decision_chain, decision_horizon
    from behavioral_equivalence.learned_abstraction import next_positive_probabilities
    from behavioral_equivalence.property_bounds import LabelEvent, verify_properties
    from behavioral_equivalence.property_refinement import property_guided_latent_refinement
    from behavioral_equivalence.statistics import paired_user_nll

    if not source_run.replace("-", "").isalnum():
        raise ValueError("Invalid source run")
    root = Path("/artifacts") / source_run
    baseline = json.loads((root / "report.json").read_text())
    strict_root = root / "learned-certified-pilot-v1"
    strict = json.loads((strict_root / "report.json").read_text())
    if baseline["status"] != "complete" or strict["status"] != "complete":
        raise ValueError("Completed baseline and strict-compression evidence required")
    out = root / "property-compression-pilot-v1"
    out.mkdir(exist_ok=False)
    snapshots = out / "source"
    snapshots.mkdir()
    for path in sorted(Path("/research/src/behavioral_equivalence").glob("*.py")):
        (snapshots / path.name).write_bytes(path.read_bytes())
    (snapshots / "property_pilot.py").write_bytes(Path(__file__).read_bytes())
    for name in ["features.py", "neural.py"]:
        if (root / "source" / name).read_bytes() != (snapshots / name).read_bytes():
            raise ValueError("Teacher features/architecture differs from attested training")
    torch.set_num_threads(2)
    started = time.monotonic()
    data = Path("/artifacts/datasets/kuairand-1k/prepared-v4")
    if (
        json.loads((data / "report.json").read_text())["output_sha256"]
        != baseline["prepared_source_sha256"]
    ):
        raise ValueError("Data differs from baseline training")
    bundles = pl.read_parquet(data / "timestamp-bundles.parquet").sort(["user_id", "time_ms"])
    users = bundles["user_id"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    pieces = []
    for left, right in zip(boundaries, boundaries[1:]):
        b = bundles.slice(int(left), int(right - left))
        pieces.append(
            bundle_history_tokens(
                b["time_ms"].to_numpy(),
                b["available_ms"].to_numpy(),
                b["clicks"].to_numpy(),
                b["exposures"].to_numpy(),
            )
        )
    features = np.concatenate(pieces)
    indices = bundles.select(["user_id", "time_ms"]).with_row_index("feature_id")
    events = (
        pl.read_parquet(
            data / "interactions.parquet", columns=["user_id", "time_ms", "tab", "is_click"]
        )
        .with_row_index("case_id")
        .join(indices, on=["user_id", "time_ms"], how="left", validate="m:1")
        .sort("case_id")
    )
    feature_id, tabs = events["feature_id"].to_numpy(), events["tab"].to_numpy()
    suite = tuple(
        LabelEvent(mode, (target,), decision_horizon(h), threshold)
        for mode, target, threshold in [
            ("occupancy", "positive", 1),
            ("reachability", "positive", 1),
            ("reachability", "negative", 1),
            ("count_at_least", "positive", 2),
            ("count_at_least", "positive", 4),
        ]
        for h in [1, 5, 10, 25, 50]
    )
    protocol = {
        "phase": "exploratory pilot, not final human confirmation",
        "source_run": source_run,
        "candidate_k": [64, 128],
        "probability_error_budgets": [0.005, 0.01, 0.03],
        "properties": [
            {
                "mode": e.mode,
                "targets": e.targets,
                "graph_horizon": e.horizon,
                "threshold": e.threshold,
            }
            for e in suite
        ],
        "prediction_origins": "pending stage of every parent latent state",
        "interpretation": "finite enumerated-property certificates, not bisimulation or human confidence bounds",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    volume.commit()
    records = []
    for protocol_name in ["chronological", "users"]:
        model_npz = np.load(strict_root / f"{protocol_name}-parent-model.npz")
        initial = np.zeros(256)
        initial[0] = 1
        parent = decision_chain(model_npz["transitions"], model_npz["emissions"], initial)
        source_record = next(r for r in strict["records"] if r["protocol"] == protocol_name)
        packed = (strict_root / source_record["certificate"]).read_bytes()
        if hashlib.sha256(packed).hexdigest() != source_record["certificate_sha256"]:
            raise ValueError("Strict source certificate changed")
        if json.loads(gzip.decompress(packed))["certificate"]["concrete_digest"] != parent.digest():
            raise ValueError("Property experiment differs from the strict parent model")
        sample = np.load(root / f"{protocol_name}-case-indices.npz")
        saved = np.load(root / f"{protocol_name}-predictions.npz")
        quant = np.load(
            root / "quantization-pilot-v3-cpu" / f"{protocol_name}-k256-state-model.npz"
        )
        model = SequencePredictor("gru")
        model.load_state_dict(
            torch.load(root / f"{protocol_name}-gru.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        centers = torch.as_tensor(quant["centers"], dtype=torch.float32)
        mean, scale = (
            torch.as_tensor(quant["mean"], dtype=torch.float32),
            torch.as_tensor(quant["scale"], dtype=torch.float32),
        )
        states, prediction = [], []
        for offset in range(0, len(sample["test"]), 1024):
            cases = sample["test"][offset : offset + 1024]
            with torch.no_grad():
                z = torch.cat(
                    [
                        model.encode(
                            torch.as_tensor(features[feature_id[cases]], dtype=torch.long)
                        ),
                        model.context(torch.as_tensor(tabs[cases], dtype=torch.long)),
                    ],
                    dim=1,
                )
                normalized = (z - mean) / scale
                states.append(
                    ((normalized[:, None, :] - centers[None, :, :]).square().sum(dim=2))
                    .argmin(dim=1)
                    .numpy()
                )
                prediction.append(torch.sigmoid(model.output(z).squeeze(-1)).numpy())
        state = np.concatenate(states)
        if (
            not np.array_equal(events["case_id"].to_numpy()[sample["test"]], saved["test_case_id"])
            or not np.array_equal(
                events["is_click"].to_numpy()[sample["test"]], saved["test_target"]
            )
            or not np.allclose(np.concatenate(prediction), saved["gru_test"], atol=1e-6, rtol=0)
        ):
            raise AssertionError("Teacher or prediction cases do not reproduce")
        reference = model_npz["emissions"][state, 1]
        prior = np.load(
            root / "quantization-pilot-v3-cpu" / f"{protocol_name}-test-predictions.npz"
        )
        if not np.allclose(reference, prior["k256"], atol=1e-12, rtol=0):
            raise AssertionError("Parent quantization predictions do not reproduce")
        np.savez_compressed(
            out / f"{protocol_name}-cases.npz",
            case_id=saved["test_case_id"],
            user_id=saved["test_user_id"],
            target=saved["test_target"],
            parent_latent_state=state,
            reference=reference,
        )
        order = np.argsort(model_npz["emissions"][:, 1], kind="stable")
        for k in protocol["candidate_k"]:
            proposed = tuple(tuple(int(s) for s in b) for b in np.array_split(order, k))
            for tolerance in protocol["probability_error_budgets"]:
                began = time.monotonic()
                abstract, blocks, evidence = property_guided_latent_refinement(
                    parent, proposed, suite, tolerance
                )
                verify_properties(parent, abstract, evidence["certificate"])
                forecasts = next_positive_probabilities(abstract, blocks)[state]
                paired = paired_user_nll(
                    saved["test_target"],
                    reference,
                    forecasts,
                    saved["test_user_id"],
                    replications=2000,
                    seed=20261015,
                )
                packed = gzip.compress(json.dumps(evidence, sort_keys=True).encode(), mtime=0)
                name = f"{protocol_name}-k{k}-tol{tolerance}.json.gz"
                (out / name).write_bytes(packed)
                record = {
                    "protocol": protocol_name,
                    "candidate_k": k,
                    "property_error_budget_probability": tolerance,
                    "reduced_latent_states": len(blocks),
                    "compression_ratio": 256 / len(blocks),
                    "parent_digest": parent.digest(),
                    "certificate": name,
                    "certificate_sha256": hashlib.sha256(packed).hexdigest(),
                    "paired_nll": paired,
                    "seconds": time.monotonic() - began,
                    "property_count": len(suite),
                }
                records.append(record)
                (out / "progress.json").write_text(json.dumps(records, indent=2))
                volume.commit()
                print(
                    json.dumps(
                        {
                            k: record[k]
                            for k in [
                                "protocol",
                                "candidate_k",
                                "property_error_budget_probability",
                                "reduced_latent_states",
                            ]
                        }
                    ),
                    flush=True,
                )
    report = {
        "status": "complete",
        "phase": "exploratory finite-property human compression pilot",
        "source_run": source_run,
        "records": records,
        "wall_seconds": time.monotonic() - started,
        "limitations": [
            "Specifically enumerated finite-model properties only, not bisimulation.",
            "No guarantee for the unknown human process, unlisted properties or environmental interventions.",
            "Same pilot timing assumptions, one trained seed and observational transition estimates.",
            "No final 27K evaluation or confirmatory parameter selection.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {
        "status": "complete",
        "records": len(records),
        "wall_seconds": report["wall_seconds"],
        "report": str(out / "report.json"),
    }


@app.local_entrypoint()
def main(source_run: str = "human-training-pilot-20261015-b"):
    import json

    call = run.spawn(source_run)
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": run.object_id}), flush=True
    )
    print(json.dumps(call.get(), indent=2), flush=True)
