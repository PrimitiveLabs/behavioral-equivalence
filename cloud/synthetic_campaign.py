"""Source-bound, checkpointed full synthetic campaign with independent PRISM checks."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-synthetic-main-v3")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("default-jre-headless", "curl")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pydantic==2.11.7")
    .run_commands(
        "curl -fL https://github.com/prismmodelchecker/prism/releases/download/v4.11/"
        "prism-4.11-linux64-x86.tar.gz -o /tmp/prism.tar.gz",
        "echo '6d4677b6ab64daa16a8bb5ad05f79c46313ad67835cb8445ee770e7f57f611c8"
        "  /tmp/prism.tar.gz' | sha256sum -c -",
        "mkdir /opt/prism && tar -xzf /tmp/prism.tar.gz -C /opt/prism --strip-components=1",
        "cd /opt/prism && ./install.sh",
    )
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def source_digest(source: Path, protocol: dict) -> str:
    import hashlib

    parts = []
    for name in protocol["source_files"]:
        parts.append(name.encode() + (source / name).read_bytes())
    parts.append(b"cloud/synthetic_campaign.py" + Path(__file__).read_bytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def validate_protocol(protocol: dict) -> None:
    import hashlib
    import json

    semantic = {k: v for k, v in protocol.items() if k not in ("campaign_id", "protocol_digest")}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    if (
        protocol["protocol_digest"] != digest
        or protocol["campaign_id"] != "synthetic-main-v3-" + digest[:12]
    ):
        raise ValueError("Protocol fingerprint or campaign ID mismatch")
    if not protocol["source_files"] or any(
        Path(n).name != n or not n.endswith(".py") for n in protocol["source_files"]
    ):
        raise ValueError("Source manifest must contain plain Python filenames")
    budget = protocol["resource_budget"]
    if budget["worker"] != {"timeout_seconds": 180, "cpu_cores": 2, "memory_gib": 2}:
        raise ValueError("Worker resource declaration differs from executable limits")
    if budget["coordinator"] != {"timeout_seconds": 86400, "cpu_cores": 0.5, "memory_gib": 1}:
        raise ValueError("Coordinator resource declaration differs from executable limits")
    if budget["startup_buffer_usd"] < 2:
        raise ValueError("Startup and idle reserve is too small")
    if budget["compute_allowance_usd"] <= 0 or any(
        v <= 0 for v in budget["rates_per_second"].values()
    ):
        raise ValueError("Positive resource prices and allowance required")


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(2048, 2048),
    timeout=180,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def world_worker(spec: dict, protocol: dict) -> dict:
    import hashlib
    import json
    import time
    import traceback
    import numpy as np
    from behavioral_equivalence.benchmark import evaluate_world, probability_tables
    from behavioral_equivalence.prism import check, write_model
    from behavioral_equivalence.synthetic import generate

    validate_protocol(protocol)
    if (
        source_digest(Path("/research/src/behavioral_equivalence"), protocol)
        != protocol["source_digest"]
    ):
        raise ValueError("Frozen source differs from executing source")
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    root = Path("/artifacts") / protocol["campaign_id"] / "worlds" / key
    volume.reload()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (
            report["source_digest"] != protocol["source_digest"]
            or report["spec"] != spec
            or report.get("protocol_digest") != protocol["protocol_digest"]
        ):
            raise ValueError("Cached world provenance mismatch")
        return {
            "key": key,
            "status": report["status"],
            "bound_violations": report.get("bound_violations", 0),
            "cached": True,
        }
    start = time.monotonic()
    try:

        def verify(model, index, method):
            model_file = root / f"model-{index}.prism"
            write_model(model, model_file)
            h = max(protocol["horizons"])
            expected = probability_tables(model, [h])[h][0]
            began = time.monotonic()
            checks = []
            for column, target in enumerate(["complete", "disengage"]):
                value = check(model_file, target, h, root / f"model-{index}-{target}.log")
                difference = abs(value - float(expected[column]))
                if difference > protocol["numerical_tolerance"]:
                    raise AssertionError(f"PRISM disagrees for {method}/{target}: {difference}")
                checks.append(
                    {
                        "target": target,
                        "horizon": h,
                        "state": 0,
                        "prism": value,
                        "dynamic_programming": float(expected[column]),
                        "difference": difference,
                    }
                )
            return {
                "status": "passed",
                "seconds": time.monotonic() - began,
                "backend": "PRISM 4.11 explicit",
                "checks": checks,
            }

        report = evaluate_world(spec, protocol, verify=verify)
        world = generate(spec["family"], spec["size"], spec["seed"], spec["perturbation"])
        np.savez_compressed(
            root / "world.npz",
            p=world.chain.transitions,
            initial=world.chain.initial,
            labels=np.array(world.chain.labels),
        )
        report.update(
            status="complete",
            source_digest=protocol["source_digest"],
            protocol_digest=protocol["protocol_digest"],
            wall_seconds=time.monotonic() - start,
        )
    except Exception as error:
        # A remote process boundary preserves actual failure evidence for resume.
        report = {
            "status": "failed",
            "spec": spec,
            "source_digest": protocol["source_digest"],
            "protocol_digest": protocol["protocol_digest"],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - start,
        }
    temporary = root / "report.partial"
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(report_path)
    volume.commit()
    return {
        "key": key,
        "status": report["status"],
        "bound_violations": report.get("bound_violations", 0),
        "wall_seconds": report["wall_seconds"],
        "cached": False,
    }


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=86400,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def campaign(protocol: dict) -> dict:
    import itertools
    import json
    import time

    validate_protocol(protocol)
    source = Path("/research/src/behavioral_equivalence")
    if source_digest(source, protocol) != protocol["source_digest"]:
        raise ValueError("Source changed after protocol freeze")
    specs = [
        dict(family=f, size=n, perturbation=e, seed=s)
        for f, n, e, s in itertools.product(
            protocol["families"], protocol["sizes"], protocol["perturbations"], protocol["seeds"]
        )
    ]
    if len(specs) != protocol["expected_worlds"]:
        raise ValueError("World count differs from frozen protocol")
    from behavioral_equivalence.budget import resource_reservation

    projected = resource_reservation(protocol, len(specs))
    if projected > protocol["resource_budget"]["compute_allowance_usd"]:
        raise ValueError("Bounded resource reservation exceeds this campaign's allowance")
    root = Path("/artifacts") / protocol["campaign_id"]
    root.mkdir(exist_ok=True)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    archive = root / "source"
    archive.mkdir(exist_ok=True)
    for name in protocol["source_files"]:
        (archive / name).write_bytes((source / name).read_bytes())
    (archive / "synthetic_campaign.py").write_bytes(Path(__file__).read_bytes())
    volume.commit()
    start = time.monotonic()
    results = []
    for index, result in enumerate(
        world_worker.map(specs, [protocol] * len(specs), return_exceptions=True, order_outputs=True)
    ):
        spec = specs[index]
        if isinstance(result, Exception):
            result = {"status": "failed", "spec": spec, "error": repr(result)}
        results.append(result)
        if len(results) % 25 == 0:
            (root / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_inputs": len(results),
                        "total_inputs": len(specs),
                        "failed": sum(r["status"] != "complete" for r in results),
                    }
                )
            )
            volume.commit()
            print(f"Synthetic campaign: {len(results)}/{len(specs)} inputs reconciled")
    report = {
        "status": "complete" if all(r["status"] == "complete" for r in results) else "incomplete",
        "worlds": len(specs),
        "failed_worlds": sum(r["status"] != "complete" for r in results),
        "bound_violations": sum(r.get("bound_violations", 0) for r in results),
        "results": results,
        "source_digest": protocol["source_digest"],
        "protocol_digest": protocol["protocol_digest"],
        "resource_bound_estimate_usd": projected,
        "actual_billing_usd": None,
        "wall_seconds": time.monotonic() - start,
        "phase": protocol["phase"],
        "scope": "numerical synthetic checks; no human-fidelity or universal numerical soundness claim",
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k != "results"}


@app.local_entrypoint()
def main(protocol_path: str):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger, resource_reservation

    protocol = json.loads(Path(protocol_path).read_text())
    validate_protocol(protocol)
    if source_digest(ROOT / "src/behavioral_equivalence", protocol) != protocol["source_digest"]:
        raise ValueError("Local source changed after freeze")
    control = ROOT / ".campaign"
    control.mkdir(exist_ok=True)
    state_file = control / (protocol["campaign_id"] + ".json")
    if state_file.exists():
        recorded = json.loads(state_file.read_text())
        call = modal.FunctionCall.from_id(recorded["function_call_id"])
        try:
            print(call.get(timeout=0))
        except TimeoutError:
            print("Existing cloud call remains live: " + recorded["function_call_id"])
        return
    ledger = BudgetLedger(control / "synthetic-budget.sqlite", 25)
    reservation = ledger.reserve(
        protocol["campaign_id"], resource_reservation(protocol, protocol["expected_worlds"])
    )
    call = campaign.spawn(protocol)
    recorded = {
        "campaign_id": protocol["campaign_id"],
        "protocol_digest": protocol["protocol_digest"],
        "function_call_id": call.object_id,
        "function_id": campaign.object_id,
        "reservation": reservation,
    }
    temporary = state_file.with_suffix(".partial")
    temporary.write_text(json.dumps(recorded, indent=2))
    temporary.replace(state_file)
    print(json.dumps(recorded))
    print(call.get())
