"""Follow a frozen synthetic campaign and recompute every integer certificate."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
AUDIT_SOURCE_DIGEST = "6d9dbce18e2635f04651415ab849d5081c5007bffc0ca931d246603cde3036eb"
app = modal.App("behavioral-equivalence-integer-certificate-campaign-v2")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def sources() -> tuple[dict, str]:
    import hashlib
    import json

    saved = {"certify_campaign_parallel.py": Path(__file__).read_text()}
    for name in [
        "core.py",
        "prism.py",
        "integer_bounds.py",
        "certificate_audit.py",
        "uncertainty.py",
    ]:
        saved[name] = (Path("/research/src/behavioral_equivalence") / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_root(campaign: str) -> Path:
    if not campaign.replace("-", "").isalnum():
        raise ValueError("Plain campaign identifier required")
    root = Path("/artifacts") / campaign
    for name in ["core.py", "prism.py"]:
        if (root / "source" / name).read_bytes() != (
            Path("/research/src/behavioral_equivalence") / name
        ).read_bytes():
            raise ValueError("Quotient or PRISM serializer changed after the experiment freeze")
    return root


def checked_audit_root(root: Path) -> Path:
    """Reuse existing certificates only with identical certificate implementation."""
    import json

    out = root / ("integer-campaign-audit-" + AUDIT_SOURCE_DIGEST[:16])
    original = json.loads((out / "source.json").read_text())
    current, _ = sources()
    for name in [
        "core.py",
        "prism.py",
        "integer_bounds.py",
        "certificate_audit.py",
        "uncertainty.py",
    ]:
        if current[name] != original[name]:
            raise ValueError(
                "Certificate implementation changed; existing proof artifacts cannot be reused"
            )
    return out


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=600,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_one(campaign: str, key: str) -> dict:
    import json
    import time
    from behavioral_equivalence.certificate_audit import audit_world

    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("SHA256 world key required")
    start = time.monotonic()
    # Containers are reused across waves while the source campaign commits new
    # worlds. Refresh the mount before consulting those newly visible inputs.
    volume.reload()
    root = checked_root(campaign)
    protocol = json.loads((root / "protocol.json").read_text())
    source = root / "worlds" / key
    original = json.loads((source / "report.json").read_text())
    if (
        original["protocol_digest"] != protocol["protocol_digest"]
        or original["source_digest"] != protocol["source_digest"]
    ):
        raise ValueError("World source or protocol differs from the campaign")
    out = checked_audit_root(root) / "worlds" / key
    if (out / "report.json").exists():
        result = json.loads((out / "report.json").read_text())
        if result["source_report_sha256"] != sha256(source / "report.json"):
            raise ValueError("Cached audit source report changed")
        if any(
            sha256(out / r["certificate"]) != r["certificate_sha256"] for r in result["methods"]
        ):
            raise ValueError("Cached certificate changed")
    else:
        result = audit_world(source, out, protocol["horizons"])
        volume.commit()
    return {
        "key": key,
        "verified_certificates": result["verified_certificates"],
        "audit_report_sha256": sha256(out / "report.json"),
        "source_report_sha256": result["source_report_sha256"],
        "strict_requested_epsilon_exceedances": sum(
            m["budget_strictly_satisfied"] is False for m in result["methods"]
        ),
        "serialized_exact_quotients_with_nonzero_radius": sum(
            m["method"] == "exact_rational" and not m["epsilon_is_exactly_zero"]
            for m in result["methods"]
        ),
        "seconds": time.monotonic() - start,
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
def follow(campaign: str, source_call_id: str) -> dict:
    import hashlib
    import itertools
    import json
    import time

    root = checked_root(campaign)
    protocol = json.loads((root / "protocol.json").read_text())
    expected = {
        hashlib.sha256(
            json.dumps(dict(family=f, size=n, perturbation=e, seed=s), sort_keys=True).encode()
        ).hexdigest()
        for f, n, e, s in itertools.product(
            protocol["families"], protocol["sizes"], protocol["perturbations"], protocol["seeds"]
        )
    }
    if len(expected) != protocol["expected_worlds"]:
        raise ValueError("Frozen matrix size mismatch")
    saved, digest = sources()
    out = checked_audit_root(root)
    revisions = out / "controller-revisions"
    revisions.mkdir(exist_ok=True)
    (revisions / (digest + ".json")).write_text(json.dumps(saved, indent=2))
    volume.commit()
    completed = {}
    start = time.monotonic()
    while len(completed) < len(expected):
        volume.reload()
        ready = [
            key
            for key in sorted(expected - completed.keys())
            if (root / "worlds" / key / "report.json").exists()
        ]
        failed_sources = [
            key
            for key in ready
            if json.loads((root / "worlds" / key / "report.json").read_text())["status"]
            != "complete"
        ]
        if failed_sources:
            raise RuntimeError(
                "Source worlds are terminal failures, not valid audit inputs: "
                + str(failed_sources)
            )
        if ready:
            wave = ready[:64]
            results = list(
                audit_one.map(
                    [campaign] * len(wave), wave, return_exceptions=True, order_outputs=True
                )
            )
            failures = [repr(r) for r in results if isinstance(r, Exception)]
            if failures:
                raise RuntimeError(
                    "Integer certificate audit failed; successful world artifacts remain cached: "
                    + str(failures)
                )
            completed.update({r["key"]: r for r in results})
            progress = {
                "status": "running",
                "audit_source_digest": AUDIT_SOURCE_DIGEST,
                "controller_source_digest": digest,
                "campaign": campaign,
                "worlds_audited": len(completed),
                "expected_worlds": len(expected),
                "verified_certificates": sum(
                    r["verified_certificates"] for r in completed.values()
                ),
                "records": list(completed.values()),
            }
            (out / "progress.json").write_text(json.dumps(progress, indent=2))
            volume.commit()
            print(json.dumps({k: v for k, v in progress.items() if k != "records"}), flush=True)
        else:
            try:
                source_status = modal.FunctionCall.from_id(source_call_id).get(timeout=0)
            except TimeoutError:
                time.sleep(30)
                continue
            # A terminal source must expose a completed report for every input.
            volume.reload()
            if source_status["status"] != "complete" or any(
                not (root / "worlds" / key / "report.json").exists() for key in expected
            ):
                raise RuntimeError("Source campaign is terminal without complete source evidence")
    result = {
        "status": "complete",
        "campaign": campaign,
        "audit_source_digest": AUDIT_SOURCE_DIGEST,
        "controller_source_digest": digest,
        "source_protocol_digest": protocol["protocol_digest"],
        "worlds_audited": len(completed),
        "verified_certificates": sum(r["verified_certificates"] for r in completed.values()),
        "strict_requested_epsilon_exceedances": sum(
            r["strict_requested_epsilon_exceedances"] for r in completed.values()
        ),
        "serialized_exact_quotients_with_nonzero_radius": sum(
            r["serialized_exact_quotients_with_nonzero_radius"] for r in completed.values()
        ),
        "records": list(completed.values()),
        "wall_seconds": time.monotonic() - start,
        "scope": "All frozen-matrix quotient inputs bound to exact normalized-kernel certificate recomputation. Not independent proof review or human validity.",
    }
    (out / "manifest.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "records"} | {"audit_root": str(out)}


@app.local_entrypoint()
def main(
    campaign: str = "synthetic-main-v3-d9f1fdcc7c35",
    source_call_id: str = "fc-01M5078YA8CXQM8V3PSAM2K7YF",
):
    import hashlib
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    run_id = "integer-audit-v2-" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    amount = 1200 * 600 * (2 * 1.31e-5 + 4 * 2.22e-6) + 86400 * (0.5 * 1.31e-5 + 2.22e-6) + 1
    print(
        json.dumps(
            BudgetLedger(ROOT / ".campaign/integer-audit-budget.sqlite", 30).reserve(
                "integer-audit-v2-8992160b280dd3f0", amount
            )
        ),
        flush=True,
    )
    print(
        json.dumps(
            BudgetLedger(ROOT / ".campaign/integer-audit-budget.sqlite", 30).reserve(
                "mount-refresh-recovery-overhead-v1", 1.5
            )
        ),
        flush=True,
    )
    handle = ROOT / ".campaign" / (run_id + ".json")
    if handle.exists():
        call = modal.FunctionCall.from_id(json.loads(handle.read_text())["function_call_id"])
        try:
            print(json.dumps(call.get(timeout=0)), flush=True)
        except TimeoutError:
            print("Existing audit call has not returned; no duplicate submission.", flush=True)
        return
    call = follow.spawn(campaign, source_call_id)
    record = {"function_call_id": call.object_id, "function_id": follow.object_id}
    handle.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
