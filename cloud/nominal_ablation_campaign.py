"""Source-bound, cached secondary ablation on all retained synthetic counts."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-nominal-ablation-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(ROOT / "configs/nominal-ablation-v1.json", "/research/config.json")
)


def snapshot():
    import hashlib
    import json

    base = Path("/research/src/behavioral_equivalence")
    if not base.exists():
        base = ROOT / "src/behavioral_equivalence"
    saved = {
        p.name: p.read_text()
        for p in [
            base / n
            for n in (
                "__init__.py",
                "core.py",
                "synthetic.py",
                "uncertainty.py",
                "integer_bounds.py",
                "uncertainty_abstraction.py",
                "robust_benchmark.py",
                "nominal_ablation.py",
                "nominal_chunk.py",
            )
        ]
    }
    saved["nominal_ablation_campaign.py"] = Path(__file__).read_text()
    config = Path("/research/config.json")
    if not config.exists():
        config = ROOT / "configs/nominal-ablation-v1.json"
    saved["config.json"] = config.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def inputs():
    import hashlib
    import json

    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    root = Path("/artifacts") / config["campaign"]
    raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["original_manifest_sha256"]:
        raise ValueError("Original campaign manifest changed")
    manifest = json.loads(raw)
    protocol = json.loads((root / "protocol.json").read_text())
    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    if (
        hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        != protocol["protocol_digest"]
    ):
        raise ValueError("Frozen original protocol changed")
    replay = json.loads(
        (root / ("replay-" + config["original_replay_source"][:16]) / "summary.json").read_text()
    )
    if (
        manifest["status"] != "complete"
        or len(manifest["chunks"]) != config["chunks"]
        or replay["manifest_sha256"] != config["original_manifest_sha256"]
        or replay["replayed_datasets"] != config["datasets"]
    ):
        raise ValueError("Complete independently executed original evidence replay required")
    original = json.loads((root / "source.json").read_text())
    for name in original:
        if name in saved and original[name] != saved[name]:
            raise ValueError("Original numerical implementation changed")
    return root, protocol, manifest, saved, digest


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=120,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def worker(chunk: str, expected_sha256: str):
    import gzip
    import hashlib
    import json
    from behavioral_equivalence.nominal_chunk import evaluate_nominal_chunk

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    if len(chunk) != 64 or any(c not in "0123456789abcdef" for c in chunk):
        raise ValueError("Chunk digest required")
    root = Path("/artifacts") / config["campaign"]
    protocol = json.loads((root / "protocol.json").read_text())
    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    if (
        hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        != protocol["protocol_digest"]
    ):
        raise ValueError("Frozen original protocol changed")
    destination = root / ("nominal-" + digest[:16]) / "chunks" / chunk
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "report.json"
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if (
            old["source_digest"] != digest
            or old["calibration_report_sha256"] != expected_sha256
            or hashlib.sha256((destination / "evidence.json.gz").read_bytes()).hexdigest()
            != old["evidence_sha256"]
        ):
            raise ValueError("Cached ablation differs")
        return old
    result = evaluate_nominal_chunk(root / "chunks" / chunk, protocol, expected_sha256)
    packed = gzip.compress(json.dumps(result, sort_keys=True).encode(), mtime=0)
    (destination / "evidence.json.gz").write_bytes(packed)
    report = {k: v for k, v in result.items() if k != "certificates"}
    report.update(source_digest=digest, evidence_sha256=hashlib.sha256(packed).hexdigest())
    report_path.write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=3600,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run(smoke: bool):
    import hashlib
    import json
    import subprocess

    volume.reload()
    root, protocol, manifest, saved, digest = inputs()
    out = root / ("nominal-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    if smoke:
        test = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-W",
                "error",
                "/research/tests/test_nominal_ablation.py",
                "/research/tests/test_nominal_chunk.py",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if test.returncode:
            raise RuntimeError(test.stdout + test.stderr)
        chunks = [
            next(
                c
                for c in manifest["chunks"]
                if c["spec"]["size"] == 128 and c["spec"]["regime"] == "imbalanced_active"
            )
        ]
    else:
        if json.loads((out / "smoke.json").read_text())["status"] != "complete":
            raise ValueError("Passing source-bound 128-state smoke required")
        chunks = manifest["chunks"]
    jobs = []
    for chunk in chunks:
        raw = (root / "chunks" / chunk["chunk_digest"] / "report.json").read_bytes()
        if json.loads(raw) != chunk:
            raise ValueError("Retained chunk differs from frozen manifest")
        jobs.append((chunk["chunk_digest"], hashlib.sha256(raw).hexdigest()))
    records, failures = [], []
    for item in worker.starmap(jobs, order_outputs=True, return_exceptions=True):
        if isinstance(item, Exception):
            failures.append(repr(item))
        else:
            records.append(item)
        (out / "progress.json").write_text(
            json.dumps(dict(chunks_complete=len(records), expected=len(jobs), failures=failures))
        )
        volume.commit()
    result = dict(
        status="incomplete" if failures else "complete",
        source_digest=digest,
        phase="smoke" if smoke else "secondary paired nominal ablation",
        original_manifest_sha256=json.loads(saved["config.json"])["original_manifest_sha256"],
        chunks=records,
        failures=failures,
        point_evaluations=sum(len(x["records"]) for x in records),
        root=str(out),
        statistical_coverage_claim="none",
    )
    (out / ("smoke.json" if smoke else "manifest.json")).write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "chunks"}


@app.local_entrypoint()
def main(smoke: bool = True):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = snapshot()
    # All 400 worker timeouts, one extra smoke worker, two coordinators, startup allowance.
    amount = 401 * 120 * (2 * 1.31e-5 + 4 * 2.22e-6) + 2 * 3600 * (0.5 * 1.31e-5 + 2.22e-6) + 0.24
    reservation = BudgetLedger(ROOT / ".campaign/nominal-ablation-budget.sqlite", 2).reserve(
        digest, amount
    )
    state = (
        ROOT / ".campaign" / ("nominal-" + digest[:16] + ("-smoke.json" if smoke else "-full.json"))
    )
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = run.spawn(smoke)
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
