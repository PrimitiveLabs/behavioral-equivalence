"""Paired summary of the full point-estimate and transition-uncertainty campaigns."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "uncertainty-repeated-v1-d567478458e8"
POINT_SOURCE = "a0450320cc941c72da068fb7873cd294c60d4b1d3bb6c8c0f2ee58e7b006960d"
ORIGINAL_MANIFEST = "38ebd31d33c62c5ea172123273804ba134af0c7e1735314ba31bde90d1cd6905"
app = modal.App("behavioral-equivalence-nominal-analysis-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": "/research/src"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def snapshot():
    import hashlib
    import json

    base = Path("/research/src/behavioral_equivalence")
    if not base.exists():
        base = ROOT / "src/behavioral_equivalence"
    saved = {
        n: (base / n).read_text()
        for n in (
            "__init__.py",
            "nominal_analysis.py",
            "repeated_analysis.py",
            "coverage_analysis.py",
        )
    }
    saved["summarize_nominal.py"] = Path(__file__).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=120,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def summarize():
    import hashlib
    import json
    from behavioral_equivalence.nominal_analysis import summarize_nominal

    volume.reload()
    root = Path("/artifacts") / CAMPAIGN
    nominal_root = root / ("nominal-" + POINT_SOURCE[:16])
    raw_original = (root / "manifest.json").read_bytes()
    raw_nominal = (nominal_root / "manifest.json").read_bytes()
    if hashlib.sha256(raw_original).hexdigest() != ORIGINAL_MANIFEST:
        raise ValueError("Frozen original manifest differs")
    nominal = json.loads(raw_nominal)
    if nominal["original_manifest_sha256"] != ORIGINAL_MANIFEST:
        raise ValueError("Ablation original-manifest binding differs")
    summary = summarize_nominal(
        json.loads(raw_original),
        nominal,
        json.loads((root / "protocol.json").read_text()),
        POINT_SOURCE,
    )
    if summary["independent_datasets"] != 4000 or summary["point_support_evaluations"] != 8000:
        raise ValueError("Incomplete full ablation design")
    saved, digest = snapshot()
    out = nominal_root / ("analysis-" + digest[:16])
    out.mkdir(exist_ok=True)
    summary.update(
        analysis_source_digest=digest,
        original_manifest_sha256=ORIGINAL_MANIFEST,
        nominal_manifest_sha256=hashlib.sha256(raw_nominal).hexdigest(),
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    raw = json.dumps(summary, indent=2).encode()
    (out / "summary.json").write_bytes(raw)
    volume.commit()
    return dict(
        status="complete",
        root=str(out),
        summary_sha256=hashlib.sha256(raw).hexdigest(),
        independent_datasets=4000,
        point_support_evaluations=8000,
        simultaneous_intervals=summary["simultaneous_intervals"],
    )


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = snapshot()
    reservation = BudgetLedger(ROOT / ".campaign/nominal-ablation-budget.sqlite", 2).reserve(
        "analysis-" + digest, 120 * (2 * 1.31e-5 + 4 * 2.22e-6) + 0.002
    )
    state = ROOT / ".campaign" / ("nominal-analysis-" + digest[:16] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = summarize.spawn()
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
