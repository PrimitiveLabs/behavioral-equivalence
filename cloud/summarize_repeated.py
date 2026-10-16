"""Checkpointed replay and fixed-world summary of repeated uncertainty evidence."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-repeated-evidence-replay-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def replay_source():
    import hashlib
    import json

    base = (
        Path("/research/src/behavioral_equivalence")
        if Path("/research/src").exists()
        else ROOT / "src/behavioral_equivalence"
    )
    saved = {
        name: (base / name).read_text()
        for name in ["repeated_evidence.py", "repeated_analysis.py", "coverage_analysis.py"]
    }
    saved["summarize_repeated.py"] = Path(__file__).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def load_protocol(campaign):
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
        raise ValueError("Protocol differs from frozen campaign")
    source = json.loads((root / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        != protocol["source_digest"]
    ):
        raise ValueError("Original source snapshot changed")
    for name, text in source.items():
        if (
            name != "uncertainty_repeated_campaign.py"
            and (Path("/research/src/behavioral_equivalence") / name).read_text() != text
        ):
            raise ValueError("Replay arithmetic differs from generating source")
    return root, protocol


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=600,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_one(campaign: str, chunk: str):
    import hashlib
    import json
    from behavioral_equivalence.repeated_evidence import audit_repetition_chunk

    if len(chunk) != 64 or any(c not in "0123456789abcdef" for c in chunk):
        raise ValueError("SHA256 chunk identity required")
    volume.reload()
    root, protocol = load_protocol(campaign)
    saved, digest = replay_source()
    out = root / ("replay-" + digest[:16])
    out.mkdir(exist_ok=True)
    path = root / "chunks" / chunk
    checksum = hashlib.sha256((path / "report.json").read_bytes()).hexdigest()
    destination = out / (chunk + ".json")
    if destination.exists():
        result = json.loads(destination.read_text())
        if result["report_sha256"] != checksum or result["replay_source_digest"] != digest:
            raise ValueError("Cached replay evidence changed")
        return result
    result = audit_repetition_chunk(path, protocol)
    result["replay_source_digest"] = digest
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    destination.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=14400,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def summarize(campaign: str):
    import hashlib
    import json
    from behavioral_equivalence.repeated_analysis import summarize_repetitions

    volume.reload()
    root, protocol = load_protocol(campaign)
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    summary = summarize_repetitions(manifest, protocol)
    # Structural validation precedes expensive work; every chunk then replays.
    chunks = [r["chunk_digest"] for r in manifest["chunks"]]
    if len(chunks) != len(set(chunks)):
        raise ValueError("Duplicate chunk in full manifest")
    replayed = list(
        audit_one.map([campaign] * len(chunks), chunks, order_outputs=True, return_exceptions=True)
    )
    failures = [repr(r) for r in replayed if isinstance(r, Exception)]
    if failures:
        raise RuntimeError(
            "Evidence replay failed; successful chunks remain cached: " + str(failures)
        )
    for original, audit in zip(manifest["chunks"], replayed):
        if original != json.loads(
            (root / "chunks" / original["chunk_digest"] / "report.json").read_text()
        ):
            raise ValueError("Manifest chunk differs from retained report")
        if audit["verified_method_support_evaluations"] != len(original["records"]):
            raise ValueError("Incomplete chunk replay")
    _, digest = replay_source()
    out = root / ("replay-" + digest[:16])
    summary.update(
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        replay_source_digest=digest,
        replayed_chunks=len(replayed),
        replayed_datasets=sum(r["verified_datasets"] for r in replayed),
        replayed_method_support_evaluations=sum(
            r["verified_method_support_evaluations"] for r in replayed
        ),
        evidence_replay_scope="All retained draws, confidence sets, integer certificates and report metrics replayed using the original arithmetic implementation.",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return {k: v for k, v in summary.items() if k != "cells"}


@app.local_entrypoint()
def main(campaign: str, smoke_chunk: str = ""):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = replay_source()
    if not smoke_chunk:
        maximum = 400 * 600 * (2 * 1.31e-5 + 4 * 2.22e-6) + 14400 * (0.5 * 1.31e-5 + 2.22e-6) + 1
        print(
            json.dumps(
                BudgetLedger(ROOT / ".campaign/repeated-replay-budget.sqlite", 12).reserve(
                    campaign, maximum
                )
            )
        )
    call = audit_one.spawn(campaign, smoke_chunk) if smoke_chunk else summarize.spawn(campaign)
    record = dict(
        function_call_id=call.object_id,
        function_id=(audit_one if smoke_chunk else summarize).object_id,
        replay_source_digest=digest,
        phase="smoke" if smoke_chunk else "full",
    )
    (
        ROOT
        / ".campaign"
        / ("repeated-replay-" + digest[:12] + ("-smoke.json" if smoke_chunk else "-full.json"))
    ).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    print(json.dumps(call.get()), flush=True)
