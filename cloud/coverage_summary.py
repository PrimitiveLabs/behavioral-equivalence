"""Monte Carlo uncertainty for the completed source-bound synthetic report."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-coverage-analysis-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": "/research/src"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=120,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def run(campaign: str) -> dict:
    import json
    import hashlib
    from behavioral_equivalence.coverage_analysis import summarize_coverage_uncertainty

    if not campaign.replace("-", "").isalnum():
        raise ValueError("Invalid campaign")
    root = Path("/artifacts") / campaign
    raw = (root / "summary.json").read_bytes()
    report = summarize_coverage_uncertainty(json.loads(raw))
    report["source_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "monte-carlo-coverage.json").write_text(json.dumps(report, indent=2))
    (root / "coverage-analysis-source.py").write_bytes(
        Path("/research/src/behavioral_equivalence/coverage_analysis.py").read_bytes()
    )
    volume.commit()
    return {
        "status": "complete",
        "reported_intervals": report["reported_intervals"],
        "report": str(root / "monte-carlo-coverage.json"),
    }


@app.local_entrypoint()
def main(campaign: str = "uncertainty-main-v1-9550b167f4b9"):
    import json

    call = run.spawn(campaign)
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": run.object_id}), flush=True
    )
    print(json.dumps(call.get(), indent=2), flush=True)
