"""Classify full point-ablation containment failures versus numerical ambiguity."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
FOLDER = "/artifacts/uncertainty-repeated-v1-d567478458e8/nominal-a0450320cc941c72"
EXPECTED_SHA = "41683bc731fb206428ebd6e38aa7f34e19fa2509217ee511f88f90c883c68a7e"
app = modal.App("behavioral-equivalence-nominal-failure-counts-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=120,
    retries=0,
    volumes={"/artifacts": volume},
)
def classify():
    import hashlib
    import json

    volume.reload()
    root = Path(FOLDER)
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
        raise ValueError("Frozen ablation manifest changed")
    manifest = json.loads(raw)
    if manifest["status"] != "complete" or manifest["point_evaluations"] != 8000:
        raise ValueError("Complete full ablation required")
    cells = {}
    for chunk in manifest["chunks"]:
        for row in chunk["records"]:
            key = row["support"], chunk["spec"]["regime"]
            cell = cells.setdefault(
                key,
                dict(
                    datasets=0,
                    contained=0,
                    proven_noncoverage=0,
                    unresolved_without_proven_noncoverage=0,
                ),
            )
            contained = all(
                p["enclosure_containment_count"] == chunk["spec"]["size"] for p in row["properties"]
            )
            disproved = any(p["proven_noncoverage_count"] > 0 for p in row["properties"])
            if contained and disproved:
                raise ValueError("Contradictory numeric classification")
            cell["datasets"] += 1
            cell[
                "contained"
                if contained
                else "proven_noncoverage"
                if disproved
                else "unresolved_without_proven_noncoverage"
            ] += 1
    if len(cells) != 4 or any(c["datasets"] != 2000 for c in cells.values()):
        raise ValueError("Missing or duplicate dataset cells")
    source = Path(__file__).read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    result = dict(
        status="complete",
        manifest_sha256=EXPECTED_SHA,
        source_digest=digest,
        cells=[dict(support=s, regime=r, **c) for (s, r), c in sorted(cells.items())],
        scope="Classification of retained directed-enclosure results, not an arithmetic replay or human confidence claim.",
    )
    out = root / ("failure-counts-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.py").write_text(source)
    data = json.dumps(result, indent=2).encode()
    (out / "report.json").write_bytes(data)
    volume.commit()
    return dict(status="complete", root=str(out), report_sha256=hashlib.sha256(data).hexdigest())


@app.local_entrypoint()
def main():
    import hashlib
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reservation = BudgetLedger(ROOT / ".campaign/nominal-ablation-budget.sqlite", 2).reserve(
        "failure-classification-" + digest, 120 * (1.31e-5 + 2 * 2.22e-6) + 0.0001
    )
    state = ROOT / ".campaign" / ("nominal-failure-counts-" + digest[:16] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = classify.spawn()
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
