"""Paired analysis of fixed fitted pilot models, retaining private case data remotely."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-pilot-analysis-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image, cpu=1, memory=2048, timeout=300, max_containers=1, volumes={"/artifacts": volume}
)
def analyze(run_id: str) -> dict:
    import json
    import numpy as np
    from behavioral_equivalence.statistics import paired_user_nll

    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    root = Path("/artifacts") / run_id
    source = json.loads((root / "report.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Finished training required")
    comparisons = []
    calibration = []
    for protocol in ["chronological", "users"]:
        x = np.load(root / f"{protocol}-predictions.npz")
        y = x["test_target"]
        users = x["test_user_id"]
        ids = x["test_case_id"]
        if len(np.unique(ids)) != len(ids):
            raise AssertionError("Duplicated test cases")
        for candidate in ["gru", "transformer"]:
            result = paired_user_nll(
                y, x["markov_test"], x[f"{candidate}_test"], users, replications=2000, seed=20261015
            )
            result.pop("noninferior_at_margin")
            result.pop("margin_nats")
            comparisons.append(
                {
                    "protocol": protocol,
                    "reference": "markov",
                    "candidate": candidate,
                    "study_role": "fixed-model baseline comparison, not compression",
                    **result,
                }
            )
        for name in ["markov", "gru", "transformer"]:
            predictions = x[f"{name}_test"]
            bins = np.minimum((predictions * 10).astype(int), 9)
            rows = []
            ece = 0.0
            for b in range(10):
                mask = bins == b
                if not mask.any():
                    continue
                observed = float(y[mask].mean())
                predicted = float(predictions[mask].mean())
                ece += float(mask.mean()) * abs(observed - predicted)
                rows.append(
                    {
                        "bin": b,
                        "cases": int(mask.sum()),
                        "observed": observed,
                        "predicted": predicted,
                    }
                )
            calibration.append(
                {
                    "protocol": protocol,
                    "model": name,
                    "ece10": ece,
                    "bins": rows,
                    "scope": "descriptive interaction-weighted calibration",
                }
            )
    report = {
        "status": "complete",
        "phase": "engineering_pilot",
        "source_run": run_id,
        "paired_user_comparisons": comparisons,
        "calibration": calibration,
        "limitations": [
            "conditional on one fitted seed",
            "timing assumptions remain unverified",
            "pilot-only test users",
            "no compression or formal human bound claim",
        ],
    }
    (root / "paired-analysis.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report
