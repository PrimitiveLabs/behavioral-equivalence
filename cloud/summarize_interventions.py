"""Hash, model-binding and metric audits before seed-cluster intervention analysis."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-intervention-analysis-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def snapshot():
    import hashlib
    import json

    base = (
        Path("/research/src/behavioral_equivalence")
        if Path("/research/src").exists()
        else ROOT / "src/behavioral_equivalence"
    )
    saved = {
        name: (base / name).read_text()
        for name in ["intervention_analysis.py", "catalogue_search.py"]
    }
    saved["summarize_interventions.py"] = Path(__file__).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def load(campaign):
    import hashlib
    import json

    if not campaign.replace("-", "").isalnum():
        raise ValueError("Valid campaign name required")
    root = Path("/artifacts") / campaign
    protocol = json.loads((root / "protocol.json").read_text())
    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    if (
        hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        != protocol["protocol_digest"]
        or protocol["campaign_id"] != campaign
    ):
        raise ValueError("Frozen protocol changed")
    original = json.loads((root / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        != protocol["source_digest"]
    ):
        raise ValueError("Generating source capsule changed")
    for name, text in original.items():
        if (
            name != "intervention_campaign.py"
            and (Path("/research/src/behavioral_equivalence") / name).read_text() != text
        ):
            raise ValueError("Source implementation differs from frozen campaign")
    return root, protocol


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=120,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_world(campaign: str, identity: str):
    import gzip
    import hashlib
    import json
    from itertools import product
    import numpy as np
    from behavioral_equivalence.core import Chain
    from behavioral_equivalence.synthetic import generate
    from behavioral_equivalence.catalogue_search import SEARCH_METHODS
    from behavioral_equivalence.intervention_analysis import CONTROLS

    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        raise ValueError("World SHA256 identity required")
    volume.reload()
    root, protocol = load(campaign)
    saved, digest = snapshot()
    source = root / "worlds" / identity
    summary = json.loads((source / "summary.json").read_text())
    if any(
        hashlib.sha256((source / name).read_bytes()).hexdigest() != sha
        for name, sha in summary["artifact_sha256"].items()
    ):
        raise ValueError("Retained world artifacts changed")
    out = root / ("analysis-" + digest[:16])
    out.mkdir(exist_ok=True)
    destination = out / (identity + ".json")
    if destination.exists():
        cached = json.loads(destination.read_text())
        if (
            cached["artifact_sha256"] != summary["artifact_sha256"]
            or cached["analysis_source_digest"] != digest
        ):
            raise ValueError("Cached structural audit differs")
        return cached
    report = json.loads((source / "report.json").read_text())
    spec = report["spec"]
    if hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() != identity:
        raise ValueError("World spec does not match identity")
    base = generate(spec["family"], spec["size"], spec["seed"])
    if (
        base.chain.digest() != report["world_digest"]
        or list(base.chain.labels) != report["labels"]
        or base.chain.initial.tolist() != report["initial"]
    ):
        raise ValueError("Ground-truth base model does not reproduce")
    if any(
        summary[k] != v
        for k, v in report.items()
        if k not in {"records", "search_catalogue", "heldout_catalogue", "labels", "initial"}
    ):
        raise ValueError("World summary differs from detailed report")
    metrics = {
        "mean_total_prediction_max_error": "total_prediction_max_error",
        "mean_abstraction_max_error": "abstraction_max_error",
        "mean_environment_model_max_error": "environment_model_max_error",
        "mean_initial_completion_error": "initial_completion_error",
    }
    for row in report["means"]:
        records = [
            r
            for r in report["records"]
            if all(r[k] == row[k] for k in ["method", "control", "horizon"])
        ]
        if (
            len(records) != 16
            or {r["heldout_index"] for r in records} != set(range(16))
            or any(r["states"] != row["states"] for r in records)
        ):
            raise ValueError("Mean does not cover all sixteen environments")
        if any(
            abs(row[name] - float(np.mean([r[raw] for r in records]))) > 1e-14
            for name, raw in metrics.items()
        ):
            raise ValueError("Reported error mean does not reproduce")
    expected = set(product(SEARCH_METHODS, CONTROLS, protocol["horizons"], range(16)))
    actual = [
        (r["method"], r["control"], r["horizon"], r["heldout_index"]) for r in report["records"]
    ]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Detailed method/control/horizon matrix is incomplete")
    evidence = json.loads(gzip.decompress((source / "evidence.json.gz").read_bytes()))
    models = {
        (r["method"], r["heldout_index"]): r
        for r in evidence["retained_models"]
        if "heldout_index" in r
    }
    bindings = [(r["method"], r["heldout_index"]) for r in evidence["certificates"]]
    if (
        set(models) != set(product(SEARCH_METHODS, range(16)))
        or len(bindings) != 64
        or set(bindings) != set(models)
    ):
        raise ValueError("Missing or duplicate retained model/certificate pairs")
    searches = {r["method"]: r for r in report["searches"]}
    for entry in evidence["certificates"]:
        key = (entry["method"], entry["heldout_index"])
        cert = entry["certificate"]
        blocks = searches[key[0]]["partition_maps"]["targeted_witness"]
        if cert["partition"] != blocks or cert["horizons"] != protocol["horizons"]:
            raise ValueError("Certificate relation or horizons differ from tested abstraction")
        concrete = Chain(
            np.array(models[key]["estimated_transition"]), base.chain.labels, base.chain.initial
        )
        abstract = Chain(
            np.array(entry["abstract_transition"]),
            tuple(base.chain.labels[b[0]] for b in blocks),
            np.array([base.chain.initial[b].sum() for b in blocks]),
        )
        if (
            cert["concrete_digest"] != concrete.digest()
            or cert["abstract_digest"] != abstract.digest()
        ):
            raise ValueError("Certificate is not bound to retained model inputs")
    result = dict(
        status="complete",
        identity_sha256=identity,
        analysis_source_digest=digest,
        artifact_sha256=summary["artifact_sha256"],
        model_certificate_bindings_checked=64,
        detailed_predictions_checked=len(actual),
        scope="Hashes, detailed mean aggregation and all retained estimated-model/certificate input bindings checked. Integer bounds were verified in generation; this audit does not independently recompute them or certify environment-response error.",
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    destination.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=7200,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def summarize(campaign: str):
    import hashlib
    import json
    from behavioral_equivalence.intervention_analysis import summarize_interventions

    volume.reload()
    root, protocol = load(campaign)
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    result = summarize_interventions(manifest, protocol)
    audits = list(
        audit_world.map(
            [campaign] * len(manifest["results"]),
            [r["world_key"] for r in manifest["results"]],
            order_outputs=True,
            return_exceptions=True,
        )
    )
    errors = [repr(r) for r in audits if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(
            "Structural audit failed; successful worlds remain cached: " + str(errors)
        )
    for record, audit in zip(manifest["results"], audits):
        if (
            record
            != json.loads((root / "worlds" / record["world_key"] / "summary.json").read_text())
            or record["artifact_sha256"] != audit["artifact_sha256"]
        ):
            raise ValueError("Full manifest differs from retained world summary")
    saved, digest = snapshot()
    out = root / ("analysis-" + digest[:16])
    result.update(
        analysis_source_digest=digest,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        audited_worlds=len(audits),
        model_certificate_bindings_checked=sum(
            r["model_certificate_bindings_checked"] for r in audits
        ),
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(campaign: str, smoke_world: str = ""):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = snapshot()
    phase = "smoke" if smoke_world else "full"
    amount = (
        (1 if smoke_world else 400) * 120 * (2 * 1.31e-5 + 4 * 2.22e-6)
        + 7200 * (0.5 * 1.31e-5 + 2.22e-6)
        + 0.1
    )
    reserved = BudgetLedger(ROOT / ".campaign/intervention-analysis-budget.sqlite", 3).reserve(
        digest + "-" + phase, amount
    )
    path = ROOT / ".campaign" / ("intervention-analysis-" + digest[:12] + "-" + phase + ".json")
    if path.exists():
        call = modal.FunctionCall.from_id(json.loads(path.read_text())["function_call_id"])
    else:
        call = (
            audit_world.spawn(campaign, smoke_world) if smoke_world else summarize.spawn(campaign)
        )
        path.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    function_id=(audit_world if smoke_world else summarize).object_id,
                    analysis_source_digest=digest,
                    reservation=reserved,
                ),
                indent=2,
            )
        )
    print(path.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
