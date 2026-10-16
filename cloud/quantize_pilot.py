"""Unverified learned-state baseline, evaluated against identical human cases."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-quantization-pilot-v3-cpu")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "polars==1.31.0")
    .pip_install("scikit-learn==1.7.1", "scipy==1.15.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1200,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def quantize(source_run: str) -> dict:
    import json
    import time
    import hashlib
    import numpy as np
    import polars as pl
    import torch
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits
    from behavioral_equivalence.features import bundle_history_tokens
    from behavioral_equivalence.neural import SequencePredictor
    from behavioral_equivalence.statistics import paired_user_nll

    if not source_run.replace("-", "").isalnum():
        raise ValueError("Invalid source run")
    root = Path("/artifacts") / source_run
    out = root / "quantization-pilot-v3-cpu"
    out.mkdir(exist_ok=False)
    source = json.loads((root / "report.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed baseline training required")
    source_out = out / "source"
    source_out.mkdir()
    for name in ["features.py", "neural.py", "statistics.py"]:
        (source_out / name).write_bytes(
            (Path("/research/src/behavioral_equivalence") / name).read_bytes()
        )
    (source_out / "quantize.py").write_bytes(Path(__file__).read_bytes())
    volume.commit()
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Path("/artifacts/datasets/kuairand-1k/prepared-v4")
    if (
        json.loads((data / "report.json").read_text())["output_sha256"]
        != source["prepared_source_sha256"]
    ):
        raise ValueError("Prepared data differs from baseline source")
    bundles = pl.read_parquet(data / "timestamp-bundles.parquet").sort(["user_id", "time_ms"])
    users = bundles["user_id"].to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(users)) + 1, len(users)]
    features = []
    for left, right in zip(bounds, bounds[1:]):
        b = bundles.slice(int(left), int(right - left))
        features.append(
            bundle_history_tokens(
                b["time_ms"].to_numpy(),
                b["available_ms"].to_numpy(),
                b["clicks"].to_numpy(),
                b["exposures"].to_numpy(),
            )
        )
    features = np.concatenate(features)
    indices = bundles.select(["user_id", "time_ms"]).with_row_index("feature_id")
    events = (
        pl.read_parquet(
            data / "interactions.parquet", columns=["user_id", "time_ms", "is_click", "tab"]
        )
        .with_row_index("case_id")
        .join(indices, on=["user_id", "time_ms"], how="left", validate="m:1")
        .sort("case_id")
    )
    feature_id = events["feature_id"].to_numpy()
    tabs = events["tab"].to_numpy()
    records = []
    comparisons = []
    for protocol in ["chronological", "users"]:
        sample = np.load(root / f"{protocol}-case-indices.npz")
        previous = np.load(root / f"{protocol}-predictions.npz")
        model = SequencePredictor("gru")
        model.load_state_dict(
            torch.load(root / f"{protocol}-gru.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        embedded = {}
        teacher = {}
        for split in ["train", "validation", "test"]:
            cases = sample[split]
            parts = []
            predictions = []
            for offset in range(0, len(cases), 1024):
                batch = cases[offset : offset + 1024]
                x = torch.as_tensor(features[feature_id[batch]], device="cpu", dtype=torch.long)
                tab = torch.as_tensor(tabs[batch], device="cpu", dtype=torch.long)
                with torch.no_grad():
                    z = torch.cat([model.encode(x), model.context(tab)], dim=1)
                    parts.append(z.cpu().numpy())
                    predictions.append(torch.sigmoid(model.output(z).squeeze(-1)).cpu().numpy())
            embedded[split] = np.concatenate(parts)
            teacher[split] = np.concatenate(predictions)
            if split == "test":
                expected_ids = events["case_id"].to_numpy()[cases]
                current_targets = events["is_click"].to_numpy()[cases]
                differences = np.abs(teacher[split] - previous["gru_test"])
                diagnostic = {
                    "protocol": protocol,
                    "device": "CPU",
                    "baseline_gpu": source["gpu"],
                    "case_ids_match": bool(np.array_equal(expected_ids, previous["test_case_id"])),
                    "targets_match": bool(np.array_equal(current_targets, previous["test_target"])),
                    "max_probability_difference": float(differences.max()),
                    "mean_probability_difference": float(differences.mean()),
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                }
                (out / f"{protocol}-reproduction.json").write_text(json.dumps(diagnostic, indent=2))
                volume.commit()
                if not diagnostic["case_ids_match"] or not diagnostic["targets_match"]:
                    raise AssertionError("Prediction case alignment changed")
                if not np.allclose(teacher[split], previous["gru_test"], atol=1e-6, rtol=0):
                    raise AssertionError(
                        "Teacher predictions do not reproduce; inspect diagnostics"
                    )
        rng = np.random.default_rng(20261015)
        train_select = rng.choice(
            len(embedded["train"]), min(50000, len(embedded["train"])), replace=False
        )
        mean = embedded["train"][train_select].mean(axis=0)
        scale = embedded["train"][train_select].std(axis=0)
        scale = np.maximum(scale, 1e-6)
        normalized = {s: (z - mean) / scale for s, z in embedded.items()}
        prediction_output = {}
        for k in [16, 32, 64, 128, 256]:
            with threadpool_limits(limits=2):
                clustering = MiniBatchKMeans(
                    n_clusters=k, batch_size=4096, n_init=3, random_state=20261015, max_iter=100
                )
                clustering.fit(normalized["train"][train_select])
                assignment = clustering.predict(normalized["train"])
            count = np.bincount(assignment, minlength=k)
            total = np.bincount(assignment, weights=teacher["train"], minlength=k)
            probabilities = np.divide(
                total, count, out=np.full(k, float(teacher["train"].mean())), where=count > 0
            )
            for split in ["validation", "test"]:
                with threadpool_limits(limits=2):
                    state = clustering.predict(normalized[split])
                p = np.clip(probabilities[state], 1e-6, 1 - 1e-6)
                y = events["is_click"].to_numpy()[sample[split]]
                if split == "test":
                    prediction_output[f"k{k}"] = p
                records.append(
                    {
                        "protocol": protocol,
                        "k": k,
                        "occupied_train_states": int((count > 0).sum()),
                        "split": split,
                        "cases": len(y),
                        "nll_nats": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                        "brier": float(np.mean((p - y) ** 2)),
                    }
                )
            np.savez_compressed(
                out / f"{protocol}-k{k}-state-model.npz",
                centers=clustering.cluster_centers_,
                probabilities=probabilities,
                mean=mean,
                scale=scale,
                counts=count,
            )
        for k in [16, 32, 64, 128]:
            comparison = paired_user_nll(
                previous["test_target"],
                prediction_output["k256"],
                prediction_output[f"k{k}"],
                previous["test_user_id"],
                replications=2000,
                seed=20261015,
            )
            comparisons.append(
                {
                    "protocol": protocol,
                    "reference_k": 256,
                    "candidate_k": k,
                    "nominal_state_reduction": 256 / k,
                    "method": "unverified learned embedding clustering",
                    **comparison,
                }
            )
        np.savez_compressed(
            out / f"{protocol}-test-predictions.npz",
            **prediction_output,
            case_id=previous["test_case_id"],
            user_id=previous["test_user_id"],
            target=previous["test_target"],
        )
        (out / "progress.json").write_text(json.dumps(records, indent=2))
        volume.commit()
    report = {
        "status": "complete",
        "phase": "engineering_pilot",
        "source_run": source_run,
        "metrics": records,
        "paired_comparisons": comparisons,
        "wall_seconds": time.monotonic() - start,
        "device": "CPU",
        "method": "train-only MiniBatchKMeans of frozen GRU/history and tab embeddings; train teacher-mean output",
        "limitations": [
            "unverified abstraction baseline",
            "no transition or label-preservation certificate",
            "one fitted seed",
            "pilot timing assumptions",
            "nominal versus effective states reported",
            "all K values exploratory; confirmatory operating point still requires validation freeze",
        ],
    }
    files = sorted(Path("/research/src/behavioral_equivalence").glob("*.py"))
    report["source_digest"] = hashlib.sha256(
        b"".join(f.name.encode() + f.read_bytes() for f in files)
    ).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {"status": "complete", "wall_seconds": report["wall_seconds"]}
