"""Source-bound multidimensional intervention and abstraction-control campaign."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-intervention-main-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)
SOURCE_FILES = [
    "__init__.py",
    "core.py",
    "synthetic.py",
    "refinement.py",
    "integer_bounds.py",
    "intervention_space.py",
    "catalogue_search.py",
    "intervention_benchmark.py",
]


def snapshot():
    import hashlib
    import json

    base = (
        Path("/research/src/behavioral_equivalence")
        if Path("/research/src").exists()
        else ROOT / "src/behavioral_equivalence"
    )
    saved = {name: (base / name).read_text() for name in SOURCE_FILES}
    saved["intervention_campaign.py"] = Path(__file__).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def validate(protocol):
    import hashlib
    import json

    _, source = snapshot()
    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    if (
        protocol["source_digest"] != source
        or protocol["protocol_digest"] != digest
        or protocol["campaign_id"] != "intervention-main-v1-" + digest[:12]
    ):
        raise ValueError("Intervention source/protocol differs from freeze")


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=900,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def world(spec: dict, protocol: dict):
    import gzip
    import hashlib
    import json
    from behavioral_equivalence.intervention_benchmark import run_intervention_world

    validate(protocol)
    volume.reload()
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    out = Path("/artifacts") / protocol["campaign_id"] / "worlds" / identity
    if (out / "summary.json").exists():
        old = json.loads((out / "summary.json").read_text())
        if (
            old["source_digest"] != protocol["source_digest"]
            or old["protocol_digest"] != protocol["protocol_digest"]
            or old["spec"] != spec
        ):
            raise ValueError("Cached intervention source/spec changed")
        if any(
            hashlib.sha256((out / name).read_bytes()).hexdigest() != sha
            for name, sha in old["artifact_sha256"].items()
        ):
            raise ValueError("Cached intervention artifacts changed")
        return old
    report, evidence = run_intervention_world(
        **spec, budget=protocol["environment_budget"], horizons=tuple(protocol["horizons"])
    )
    if spec["surface"] == "negative_control" and any(
        r["total_prediction_max_error"] > 1e-12 for r in report["records"]
    ):
        raise AssertionError("Structural negative control changed observable predictions")
    out.mkdir(parents=True, exist_ok=True)
    report.update(
        source_digest=protocol["source_digest"], protocol_digest=protocol["protocol_digest"]
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "evidence.json.gz").write_bytes(
        gzip.compress(json.dumps(evidence, sort_keys=True).encode(), mtime=0)
    )
    summary = {
        k: v
        for k, v in report.items()
        if k not in {"records", "search_catalogue", "heldout_catalogue", "labels", "initial"}
    }
    summary["world_key"] = identity
    summary["artifact_sha256"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ["report.json", "evidence.json.gz"]
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=86400,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run(protocol: dict, smoke: bool):
    import itertools
    import json

    validate(protocol)
    root = Path("/artifacts") / protocol["campaign_id"]
    root.mkdir(exist_ok=True)
    saved, _ = snapshot()
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    volume.commit()
    specs = (
        [
            dict(family="trust", size=32, seed=7, surface=surface)
            for surface in ["localized", "negative_control"]
        ]
        if smoke
        else [
            dict(family=f, size=n, seed=s, surface=surface)
            for f, n, s, surface in itertools.product(
                protocol["families"], protocol["sizes"], protocol["seeds"], protocol["surfaces"]
            )
        ]
    )
    results, failures = [], []
    for result in world.map(
        specs, [protocol] * len(specs), order_outputs=True, return_exceptions=True
    ):
        if isinstance(result, Exception):
            failures.append(repr(result))
        else:
            results.append(result)
        if len(results) % 10 == 0:
            (root / "progress.json").write_text(
                json.dumps(
                    dict(
                        worlds_complete=len(results), worlds_expected=len(specs), failures=failures
                    )
                )
            )
            volume.commit()
    report = dict(
        status="incomplete" if failures else "complete",
        phase="smoke" if smoke else "frozen multidimensional synthetic intervention",
        results=results,
        failures=failures,
        source_digest=protocol["source_digest"],
        protocol_digest=protocol["protocol_digest"],
        worlds=len(results),
        expected_worlds=len(specs),
        verified_estimated_model_certificates=sum(
            r["verified_estimated_model_certificates"] for r in results
        ),
    )
    (root / ("smoke-manifest.json" if smoke else "manifest.json")).write_text(
        json.dumps(report, indent=2)
    )
    volume.commit()
    return {k: v for k, v in report.items() if k != "results"}


@app.local_entrypoint()
def main(protocol_path: str, smoke: bool = True):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    protocol = json.loads(Path(protocol_path).read_text())
    validate(protocol)
    suffix = "smoke" if smoke else "full"
    state = ROOT / ".campaign" / (protocol["campaign_id"] + "-" + suffix + ".json")
    if state.exists():
        previous = json.loads(state.read_text())
        print(json.dumps(previous), flush=True)
        try:
            print(
                json.dumps(modal.FunctionCall.from_id(previous["function_call_id"]).get(timeout=1)),
                flush=True,
            )
        except TimeoutError:
            print("Existing invocation remains live", flush=True)
        return
    worlds = 2 if smoke else protocol["expected_worlds"]
    reservation = worlds * 900 * (2 * 1.31e-5 + 4 * 2.22e-6) + 86400 * (0.5 * 1.31e-5 + 2.22e-6) + 1
    reserved = BudgetLedger(ROOT / ".campaign/intervention-budget.sqlite", 20).reserve(
        protocol["campaign_id"] + "-" + suffix, reservation
    )
    call = run.spawn(protocol, smoke)
    record = dict(function_call_id=call.object_id, function_id=run.object_id, reservation=reserved)
    state.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    print(json.dumps(call.get()), flush=True)
