"""Frozen full uncertainty benchmark with exact numeric and uniform abstractions."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-uncertainty-main-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def source_digest(protocol: dict) -> str:
    import hashlib

    source = (
        Path("/research/src/behavioral_equivalence")
        if Path("/research/src").exists()
        else ROOT / "src/behavioral_equivalence"
    )
    payload = b"".join(
        name.encode() + (source / name).read_bytes() for name in protocol["source_files"]
    )
    return hashlib.sha256(
        payload + b"cloud/uncertainty_campaign.py" + Path(__file__).read_bytes()
    ).hexdigest()


def validate(protocol: dict) -> None:
    import hashlib
    import json
    from behavioral_equivalence.budget import resource_reservation

    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    if (
        protocol["protocol_digest"] != digest
        or protocol["campaign_id"] != "uncertainty-main-v1-" + digest[:12]
        or source_digest(protocol) != protocol["source_digest"]
    ):
        raise ValueError("Frozen protocol or source fingerprint mismatch")
    if protocol["resource_budget"]["worker"] != {
        "timeout_seconds": 300,
        "cpu_cores": 2,
        "memory_gib": 4,
    } or protocol["resource_budget"]["coordinator"] != {
        "timeout_seconds": 7200,
        "cpu_cores": 0.5,
        "memory_gib": 1,
    }:
        raise ValueError("Declared resources differ from hard function limits")
    if (
        resource_reservation(protocol, protocol["expected_worlds"])
        > protocol["resource_budget"]["compute_allowance_usd"]
    ):
        raise ValueError("Conservative resource bound exceeds allowance")


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=300,
    retries=0,
    max_containers=4,
    volumes={"/artifacts": volume},
)
def worker(spec: dict, protocol: dict) -> dict:
    import gzip
    import hashlib
    import json
    import time
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.synthetic import generate
    from behavioral_equivalence.confidence import certified_hoeffding, verify_hoeffding
    from behavioral_equivalence.uncertainty import clopper_pearson_intervals
    from behavioral_equivalence.robust_benchmark import assess_intervals

    validate(protocol)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    root = Path("/artifacts") / protocol["campaign_id"] / "worlds" / key
    if (root / "report.json").exists():
        old = json.loads((root / "report.json").read_text())
        if old["protocol_digest"] != protocol["protocol_digest"] or old["spec"] != spec:
            raise ValueError("Cached uncertainty result provenance mismatch")
        return {
            "key": key,
            "status": "complete",
            "method_datasets": len(old["records"]),
            "cached": True,
        }
    root.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    world = generate(spec["family"], spec["size"], spec["seed"], protocol["perturbation"])
    np.savez_compressed(
        root / "world.npz",
        p=world.chain.transitions,
        labels=world.chain.labels,
        initial=world.chain.initial,
    )
    records = []
    alpha = Fraction(*protocol["alpha"])
    for m in protocol["samples_per_row"]:
        rng = np.random.default_rng(
            [20261015, protocol["families"].index(spec["family"]), spec["size"], spec["seed"], m]
        )
        counts = np.array([rng.multinomial(m, row) for row in world.chain.transitions])
        np.savez_compressed(root / f"counts-{m}.npz", counts=counts)
        for method in ["certified_hoeffding", "clopper_pearson_float_quantiles"]:
            began = time.monotonic()
            if method == "certified_hoeffding":
                uncertain, confidence = certified_hoeffding(counts, world.chain.labels, alpha)
                verify_hoeffding(counts, world.chain.labels, confidence)
            else:
                uncertain = clopper_pearson_intervals(counts, world.chain.labels, float(alpha))
                confidence = {
                    "statistical_numeric_scope": "floating beta quantiles lack directed-rounding proof"
                }
            metrics, evidence = assess_intervals(
                world.chain, uncertain, world.partition, protocol["horizons"], protocol["bits"]
            )
            artifact = {"confidence": confidence, "evidence": evidence}
            packed = gzip.compress(json.dumps(artifact, sort_keys=True).encode(), mtime=0)
            name = f"{m}-{method}.json.gz"
            (root / name).write_bytes(packed)
            records.append(
                {
                    "method": method,
                    "samples_per_row": m,
                    **metrics,
                    "certificate": name,
                    "certificate_sha256": hashlib.sha256(packed).hexdigest(),
                    "seconds": time.monotonic() - began,
                }
            )
    report = {
        "status": "complete",
        "spec": spec,
        "protocol_digest": protocol["protocol_digest"],
        "source_digest": protocol["source_digest"],
        "world_digest": world.chain.digest(),
        "records": records,
        "wall_seconds": time.monotonic() - start,
        "sampling_scope": "NumPy multinomial simulation under fixed row design; theoretical guarantees conditional on the stated iid sampling experiment",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {
        "key": key,
        "status": "complete",
        "method_datasets": len(records),
        "wall_seconds": report["wall_seconds"],
        "cached": False,
    }


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=7200,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def campaign(protocol: dict) -> dict:
    import itertools
    import json
    import time

    validate(protocol)
    specs = [
        dict(family=f, size=n, seed=s)
        for f, n, s in itertools.product(protocol["families"], protocol["sizes"], protocol["seeds"])
    ]
    if len(specs) != protocol["expected_worlds"]:
        raise ValueError("World matrix differs from frozen count")
    root = Path("/artifacts") / protocol["campaign_id"]
    root.mkdir(exist_ok=True)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    src = root / "source"
    src.mkdir(exist_ok=True)
    for name in protocol["source_files"]:
        (src / name).write_bytes((Path("/research/src/behavioral_equivalence") / name).read_bytes())
    (src / "uncertainty_campaign.py").write_bytes(Path(__file__).read_bytes())
    volume.commit()
    began = time.monotonic()
    results = []
    for index, result in enumerate(
        worker.map(specs, [protocol] * len(specs), return_exceptions=True, order_outputs=True)
    ):
        if isinstance(result, Exception):
            result = {"status": "failed", "spec": specs[index], "error": repr(result)}
        results.append(result)
        if len(results) % 10 == 0:
            progress = {
                "reconciled_worlds": len(results),
                "total_worlds": len(specs),
                "failed": sum(r["status"] != "complete" for r in results),
            }
            (root / "progress.json").write_text(json.dumps(progress))
            volume.commit()
            print(json.dumps(progress), flush=True)
    report = {
        "status": "complete"
        if all(r["status"] == "complete" for r in results)
        else "completed_with_failures",
        "campaign_id": protocol["campaign_id"],
        "protocol_digest": protocol["protocol_digest"],
        "worlds": len(results),
        "method_datasets": sum(r.get("method_datasets", 0) for r in results),
        "failed": sum(r["status"] != "complete" for r in results),
        "results": results,
        "wall_seconds": time.monotonic() - began,
    }
    (root / "manifest.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main(config: str):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger, resource_reservation

    protocol = json.loads(Path(config).read_text())
    validate(protocol)
    folder = ROOT / ".campaign"
    folder.mkdir(exist_ok=True)
    handle = folder / (protocol["campaign_id"] + ".json")
    if handle.exists():
        saved = json.loads(handle.read_text())
        if saved["protocol_digest"] != protocol["protocol_digest"]:
            raise ValueError("Saved uncertainty handle differs from frozen protocol")
        print(json.dumps(saved), flush=True)
        result = modal.FunctionCall.from_id(saved["function_call_id"]).get()
    else:
        reservation = BudgetLedger(folder / "uncertainty-budget.sqlite", 8).reserve(
            protocol["campaign_id"], resource_reservation(protocol, protocol["expected_worlds"])
        )
        call = campaign.spawn(protocol)
        saved = {
            "campaign_id": protocol["campaign_id"],
            "protocol_digest": protocol["protocol_digest"],
            "function_call_id": call.object_id,
            "function_id": campaign.object_id,
            "reservation": reservation,
        }
        temporary = handle.with_suffix(".partial")
        temporary.write_text(json.dumps(saved, indent=2))
        temporary.replace(handle)
        print(json.dumps(saved), flush=True)
        result = call.get()
    print(json.dumps(result, indent=2), flush=True)
