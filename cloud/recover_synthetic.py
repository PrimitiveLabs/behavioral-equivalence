"""Preserve a terminal timeout attempt and rerun the unchanged frozen worker."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-synthetic-timeout-recovery-v1")
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
    .add_local_file(
        ROOT / "cloud/synthetic_campaign.py", remote_path="/research/frozen-synthetic.py"
    )
)


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(2048, 2048),
    timeout=180,
    retries=0,
    volumes={"/artifacts": volume},
)
def recover(original_call_id: str, manifest_sha256: str, campaign: str):
    import hashlib
    import json
    import shutil
    from datetime import datetime, timezone
    import modal
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "frozen_synthetic", "/research/frozen-synthetic.py"
    )
    worker = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(worker)

    original = modal.FunctionCall.from_id(original_call_id).get(timeout=0)
    if original["status"] != "incomplete" or original["failed_worlds"] != 1:
        raise ValueError("Recovery requires a terminal single-failure original campaign")
    if not campaign.replace("-", "").isalnum():
        raise ValueError("Campaign name required")
    volume.reload()
    root = Path("/artifacts") / campaign
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("Original terminal manifest changed")
    manifest = json.loads(raw)
    protocol = json.loads((root / "protocol.json").read_text())
    worker.validate_protocol(protocol)
    if (
        worker.source_digest(Path("/research/src/behavioral_equivalence"), protocol)
        != protocol["source_digest"]
    ):
        raise ValueError("Frozen scientific worker changed")
    failed = [r for r in manifest["results"] if r["status"] != "complete"]
    if len(failed) != 1 or not failed[0].get("error", "").startswith("FunctionTimeoutError("):
        raise ValueError("Only the recorded operational timeout is eligible")
    spec = failed[0]["spec"]
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    recovery = root / "timeout-recovery-v1"
    recovery.mkdir(exist_ok=True)
    backup = recovery / "original-attempt" / key
    if not backup.exists():
        shutil.copytree(root / "worlds" / key, backup)
        (recovery / "original-manifest.json").write_bytes(raw)
        (recovery / "source.py").write_bytes(Path(__file__).read_bytes())
        retained = {
            str(p.relative_to(backup)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(backup.rglob("*"))
            if p.is_file()
        }
        (recovery / "attempt-preservation.json").write_text(
            json.dumps(
                dict(
                    original_call_id=original_call_id,
                    original_manifest_sha256=manifest_sha256,
                    failure=failed[0],
                    retained_artifact_sha256=retained,
                    source_digest=protocol["source_digest"],
                    protocol_digest=protocol["protocol_digest"],
                    unchanged_worker_timeout_seconds=180,
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                ),
                indent=2,
            )
        )
        volume.commit()
    # The exact frozen worker body executes inside identical CPU/memory/time
    # limits. The explicit source mount avoids imported-module packaging.
    result = worker.world_worker.local(spec, protocol)
    volume.reload()
    if result["status"] != "complete":
        raise RuntimeError("Unchanged timeout retry did not complete: " + str(result))
    replacement_raw = (root / "worlds" / key / "report.json").read_bytes()
    replacement = json.loads(replacement_raw)
    if (
        replacement["spec"] != spec
        or replacement["source_digest"] != protocol["source_digest"]
        or replacement["protocol_digest"] != protocol["protocol_digest"]
    ):
        raise ValueError("Recovered worker provenance differs from frozen campaign")
    results = [result if r == failed[0] else r for r in manifest["results"]]
    reconciled = dict(
        manifest,
        status="complete",
        failed_worlds=0,
        results=results,
        bound_violations=sum(r.get("bound_violations", 0) for r in results),
        recovery=dict(
            original_manifest_sha256=manifest_sha256,
            spec=spec,
            recovered_report_sha256=hashlib.sha256(replacement_raw).hexdigest(),
            original_manifest_preserved=True,
            source_and_scientific_parameters="unchanged",
            actual_billing_usd=None,
        ),
    )
    (recovery / "reconciled-manifest.json").write_text(json.dumps(reconciled, indent=2))
    volume.commit()
    return {k: v for k, v in reconciled.items() if k != "results"} | {
        "recovery_root": str(recovery)
    }


@app.local_entrypoint()
def recover_timeout(
    original_call_id: str = "fc-01M5078YA8CXQM8V3PSAM2K7YF",
    campaign: str = "synthetic-main-v3-d9f1fdcc7c35",
):
    import hashlib
    import json
    import sys
    import modal

    local = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(local / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    manifest_sha = hashlib.sha256(
        (local / "reports/preflight/synthetic-main-original-manifest.json").read_bytes()
    ).hexdigest()
    controller_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = "timeout-recovery-" + manifest_sha[:16] + "-" + controller_sha[:12]
    reservation = BudgetLedger(local / ".campaign/synthetic-budget.sqlite", 25).reserve(
        identity,
        180 * (2 * 1.31e-5 + 2 * 2.22e-6) + 600 * (0.5 * 1.31e-5 + 2.22e-6) + 0.1,
    )
    state = local / ".campaign" / (identity + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = recover.spawn(original_call_id, manifest_sha, campaign)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    function_id=recover.object_id,
                    original_manifest_sha256=manifest_sha,
                    reservation=reservation,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
