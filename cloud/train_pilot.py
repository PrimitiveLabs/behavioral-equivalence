"""Bounded GPU pilots on real timestamp-bundle histories, not final evaluation."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-training-pilot-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "polars==1.31.0")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    gpu=["L4", "A10", "T4"],
    cpu=2,
    memory=8192,
    timeout=1200,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def train(run_id: str) -> dict:
    import hashlib
    import json
    import time
    import numpy as np
    import polars as pl
    import torch
    from behavioral_equivalence.features import bundle_history_tokens
    from behavioral_equivalence.neural import SequencePredictor

    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    root = Path("/artifacts/datasets/kuairand-1k/prepared-v4")
    audit = json.loads((root / "independent-audit.json").read_text())
    if audit["status"] != "complete":
        raise ValueError("Independent split audit required")
    out = Path("/artifacts") / run_id
    out.mkdir(exist_ok=False)
    source_out = out / "source"
    source_out.mkdir()
    for source_file in Path("/research/src/behavioral_equivalence").glob("*.py"):
        (source_out / source_file.name).write_bytes(source_file.read_bytes())
    (source_out / "cloud-train.py").write_bytes(Path(__file__).read_bytes())
    (out / "started.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "gpu": torch.cuda.get_device_name(0),
                "function_call_id": modal.current_function_call_id(),
                "phase": "engineering_pilot",
                "state": "running",
            },
            indent=2,
        )
    )
    volume.commit()
    start = time.monotonic()
    torch.manual_seed(20261015)
    np.random.seed(20261015)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(20261015)
    bundles = pl.read_parquet(root / "timestamp-bundles.parquet").sort(["user_id", "time_ms"])
    users = bundles["user_id"].to_numpy()
    boundaries = np.r_[0, np.flatnonzero(np.diff(users)) + 1, len(users)]
    histories = []
    for left, right in zip(boundaries, boundaries[1:]):
        b = bundles.slice(int(left), int(right - left))
        histories.append(
            bundle_history_tokens(
                b["time_ms"].to_numpy(),
                b["available_ms"].to_numpy(),
                b["clicks"].to_numpy(),
                b["exposures"].to_numpy(),
            )
        )
    features = np.concatenate(histories)
    index = bundles.select(["user_id", "time_ms"]).with_row_index("feature_id")
    events = (
        pl.read_parquet(
            root / "interactions.parquet",
            columns=["user_id", "time_ms", "is_click", "tab", "chronological_split", "user_split"],
        )
        .with_row_index("case_id")
        .join(index, on=["user_id", "time_ms"], how="left", validate="m:1")
        .sort("case_id")
    )
    if events["feature_id"].null_count():
        raise AssertionError("Missing history mapping")
    if events.filter(~pl.col("tab").is_between(0, 14)).height:
        raise ValueError("Unknown pre-exposure scenario")
    reports = []
    for protocol, split_column in [
        ("chronological", "chronological_split"),
        ("users", "user_split"),
    ]:
        # Equal fixed random samples for every model. Final 27K data untouched.
        ids = {}
        for split, cap in [
            ("train", 100000),
            ("validation", 20000),
            ("calibration", 20000),
            ("test", 20000),
        ]:
            eligible = np.flatnonzero(events[split_column].to_numpy() == split)
            ids[split] = np.sort(rng.choice(eligible, min(len(eligible), cap), replace=False))
        all_feature_ids = events["feature_id"].to_numpy()
        all_tabs = events["tab"].to_numpy()
        all_targets = events["is_click"].to_numpy()
        np.savez_compressed(out / f"{protocol}-case-indices.npz", **ids)
        # Markov baseline conditions on last completed rate token and current tab.
        train_ids = ids["train"]
        markov_index = (
            features[all_feature_ids[train_ids], -1].astype(int) * 15 + all_tabs[train_ids]
        )
        count = np.bincount(markov_index, minlength=18 * 15)
        successes = np.bincount(markov_index, weights=all_targets[train_ids], minlength=18 * 15)
        probs = (successes + 1) / (count + 2)
        checkpoint = {
            "probabilities": probs.tolist(),
            "smoothing": "Beta(1,1)",
            "state": "last available bundle rate token x current tab",
        }
        (out / f"{protocol}-markov.json").write_text(json.dumps(checkpoint))
        prediction_outputs = {}
        for split in ["validation", "calibration", "test"]:
            prediction_outputs[f"{split}_case_id"] = events["case_id"].to_numpy()[ids[split]]
            prediction_outputs[f"{split}_user_id"] = events["user_id"].to_numpy()[ids[split]]
            prediction_outputs[f"{split}_target"] = all_targets[ids[split]]
        for architecture in ["markov", "gru", "transformer"]:
            model_start = time.monotonic()
            if architecture != "markov":
                torch.manual_seed(20261015)
                model = SequencePredictor(architecture).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                for epoch in range(2):
                    model.train()
                    shuffled = rng.permutation(train_ids)
                    for offset in range(0, len(shuffled), 512):
                        batch = shuffled[offset : offset + 512]
                        x = torch.as_tensor(
                            features[all_feature_ids[batch]], device="cuda", dtype=torch.long
                        )
                        tab = torch.as_tensor(all_tabs[batch], device="cuda", dtype=torch.long)
                        y = torch.as_tensor(all_targets[batch], device="cuda", dtype=torch.float32)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            model(x, tab), y
                        )
                        if not torch.isfinite(loss):
                            raise ArithmeticError("Nonfinite training loss")
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                torch.save(model.state_dict(), out / f"{protocol}-{architecture}.pt")
                model.eval()
            metrics = {}
            for split in ["validation", "calibration", "test"]:
                cases = ids[split]
                predictions = []
                for offset in range(0, len(cases), 1024):
                    batch = cases[offset : offset + 1024]
                    if architecture == "markov":
                        state = (
                            features[all_feature_ids[batch], -1].astype(int) * 15 + all_tabs[batch]
                        )
                        predictions.extend(probs[state].tolist())
                    else:
                        with torch.no_grad():
                            x = torch.as_tensor(
                                features[all_feature_ids[batch]], device="cuda", dtype=torch.long
                            )
                            tab = torch.as_tensor(all_tabs[batch], device="cuda", dtype=torch.long)
                            predictions.extend(torch.sigmoid(model(x, tab)).cpu().tolist())
                y = all_targets[cases]
                p = np.clip(np.array(predictions), 1e-6, 1 - 1e-6)
                metrics[split] = {
                    "cases": len(cases),
                    "nll_nats": float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                    "brier": float(np.mean((p - y) ** 2)),
                    "observed_rate": float(y.mean()),
                    "predicted_rate": float(p.mean()),
                }
                prediction_outputs[f"{architecture}_{split}"] = p
            reports.append(
                {
                    "protocol": protocol,
                    "model": architecture,
                    "metrics": metrics,
                    "wall_seconds": time.monotonic() - model_start,
                }
            )
            (out / "progress.json").write_text(json.dumps(reports, indent=2))
            volume.commit()
        np.savez_compressed(out / f"{protocol}-predictions.npz", **prediction_outputs)
    report = {
        "status": "complete",
        "phase": "engineering_pilot",
        "run_id": run_id,
        "reports": reports,
        "history": "32 available unordered timestamp bundles; rates discretized into 16 bins",
        "models": "Markov, GRU64, 2-layer Transformer64; current tab pre-exposure context",
        "epochs": 2,
        "train_cap_per_protocol": 100000,
        "seed": 20261015,
        "limitations": [
            "availability assumption unverified",
            "single training seed",
            "pilot-only 1K users",
            "selected-field deduplication",
            "not a state-abstraction noninferiority test",
        ],
        "gpu": torch.cuda.get_device_name(0),
        "wall_seconds": time.monotonic() - start,
        "prepared_source_sha256": json.loads((root / "report.json").read_text())["output_sha256"],
    }
    source_files = sorted(Path("/research/src/behavioral_equivalence").glob("*.py"))
    report["source_digest"] = hashlib.sha256(
        b"".join(f.name.encode() + f.read_bytes() for f in source_files)
    ).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report
