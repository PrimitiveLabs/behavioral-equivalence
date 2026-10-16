"""Independent CPU inference diagnosis for saved GPU predictor outputs."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-teacher-cpu-diagnosis-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "polars==1.31.0")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image, cpu=2, memory=8192, timeout=600, max_containers=1, volumes={"/artifacts": volume}
)
def diagnose(source_run: str) -> dict:
    import json
    import time
    import numpy as np
    import polars as pl
    import torch
    from behavioral_equivalence.features import bundle_history_tokens
    from behavioral_equivalence.neural import SequencePredictor

    if not source_run.replace("-", "").isalnum():
        raise ValueError("Invalid source run")
    root = Path("/artifacts") / source_run
    for name in ["features", "neural"]:
        saved = (root / f"source/{name}.py").read_bytes()
        current = Path(f"/research/src/behavioral_equivalence/{name}.py").read_bytes()
        if saved != current:
            raise AssertionError("Saved inference source changed: " + name)
    start = time.monotonic()
    torch.set_num_threads(2)
    data = Path("/artifacts/datasets/kuairand-1k/prepared-v4")
    bundles = pl.read_parquet(data / "timestamp-bundles.parquet").sort(["user_id", "time_ms"])
    user = bundles["user_id"].to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(user)) + 1, len(user)]
    histories = []
    for left, right in zip(bounds, bounds[1:]):
        b = bundles.slice(int(left), int(right - left))
        histories.append(
            bundle_history_tokens(
                b["time_ms"].to_numpy(),
                b["available_ms"].to_numpy(),
                b["clicks"].to_numpy(),
                b["exposures"].to_numpy(),
            )
        )
    histories = np.concatenate(histories)
    indices = bundles.select(["user_id", "time_ms"]).with_row_index("feature_id")
    events = (
        pl.read_parquet(
            data / "interactions.parquet",
            columns=["user_id", "time_ms", "is_click", "tab", "chronological_split", "user_split"],
        )
        .with_row_index("case_id")
        .join(indices, on=["user_id", "time_ms"], how="left", validate="m:1")
        .sort("case_id")
    )
    feature_ids = events["feature_id"].to_numpy()
    tabs = events["tab"].to_numpy()
    reports = []
    for protocol in ["chronological", "users"]:
        case = np.load(root / f"{protocol}-case-indices.npz")["test"]
        saved = np.load(root / f"{protocol}-predictions.npz")
        model = SequencePredictor("gru")
        model.load_state_dict(
            torch.load(root / f"{protocol}-gru.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        predictions = []
        head_differences = []
        for offset in range(0, len(case), 1024):
            batch = case[offset : offset + 1024]
            x = torch.as_tensor(histories[feature_ids[batch]], dtype=torch.long)
            tab = torch.as_tensor(tabs[batch], dtype=torch.long)
            with torch.no_grad():
                direct = torch.sigmoid(model(x, tab))
                manual = torch.sigmoid(
                    model.output(torch.cat([model.encode(x), model.context(tab)], 1)).squeeze(-1)
                )
                predictions.extend(direct.tolist())
                head_differences.append(float((direct - manual).abs().max()))
        p = np.clip(np.array(predictions), 1e-6, 1 - 1e-6)
        difference = np.abs(p - saved["gru_test"])
        report = {
            "protocol": protocol,
            "cases": len(case),
            "case_ids_match": bool(
                np.array_equal(events["case_id"].to_numpy()[case], saved["test_case_id"])
            ),
            "targets_match": bool(
                np.array_equal(events["is_click"].to_numpy()[case], saved["test_target"])
            ),
            "users_match": bool(
                np.array_equal(events["user_id"].to_numpy()[case], saved["test_user_id"])
            ),
            "max_abs_probability_difference": float(difference.max()),
            "mean_abs_probability_difference": float(difference.mean()),
            "manual_head_max_difference": max(head_differences),
            "within_existing_reproduction_tolerance": bool(
                np.allclose(p, saved["gru_test"], atol=1e-6)
            ),
            "source_identical_to_training": True,
        }
        reports.append(report)
    result = {
        "status": "complete",
        "source_run": source_run,
        "device": "CPU",
        "torch": torch.__version__,
        "reports": reports,
        "wall_seconds": time.monotonic() - start,
        "interpretation": "Diagnosis only; does not change saved predictions or relax any tolerance",
    }
    (root / "cpu-reproduction-diagnosis.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result
