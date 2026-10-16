"""Post-audit finished synthetic worlds without changing their frozen experiment."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-integer-certificate-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=1200,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_ready(campaign: str, max_worlds: int = 1) -> dict:
    import hashlib
    import json
    import time
    from behavioral_equivalence.certificate_audit import audit_world

    if not campaign.replace("-", "").isalnum() or not 1 <= max_worlds <= 25:
        raise ValueError("Valid campaign and bounded audit wave required")
    root = Path("/artifacts") / campaign
    protocol = json.loads((root / "protocol.json").read_text())
    # Reconstructed quotients must use the same source as the experiment.
    source = Path("/research/src/behavioral_equivalence")
    for name in ["core.py", "prism.py"]:
        if (source / name).read_bytes() != (root / "source" / name).read_bytes():
            raise ValueError("Audit quotient/serializer differs from frozen source")
    files = ["integer_bounds.py", "certificate_audit.py", "uncertainty.py"]
    snapshot = {name: (source / name).read_text() for name in files}
    snapshot["certify_campaign.py"] = Path(__file__).read_text()
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    out = root / f"integer-audit-{digest[:16]}"
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(snapshot, indent=2))
    start = time.monotonic()
    records = []
    for path in sorted((root / "worlds").glob("*/report.json")):
        saved = out / path.parent.name / "report.json"
        if saved.exists() or json.loads(path.read_text())["status"] != "complete":
            continue
        result = audit_world(path.parent, saved.parent, protocol["horizons"])
        records.append(
            {"world": path.parent.name, "verified_certificates": result["verified_certificates"]}
        )
        volume.commit()
        if len(records) == max_worlds:
            break
    result = {
        "status": "complete",
        "campaign": campaign,
        "audit_source_digest": digest,
        "audit_root": str(out),
        "worlds_this_wave": len(records),
        "certificates_this_wave": sum(r["verified_certificates"] for r in records),
        "records": records,
        "wall_seconds": time.monotonic() - start,
        "scope": "completed source worlds only; campaign-wide audit not implied",
    }
    (out / "latest-wave.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(campaign: str, max_worlds: int = 1):
    import json

    call = audit_ready.spawn(campaign, max_worlds)
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": audit_ready.object_id}),
        flush=True,
    )
    print(json.dumps(call.get(), indent=2), flush=True)
