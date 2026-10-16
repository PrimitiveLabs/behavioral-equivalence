"""Checkpointed arithmetic replay of all retained point-ablation evidence."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-nominal-replay-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_nominal_replay.py", remote_path="/research/tests/test_nominal_replay.py"
    )
    .add_local_file(ROOT / "configs/nominal-replay-v1.json", remote_path="/research/config.json")
)


def checksum(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    import json
    import hashlib

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/nominal-replay-v1.json"
    )
    names = [
        "__init__.py",
        "core.py",
        "synthetic.py",
        "uncertainty.py",
        "integer_bounds.py",
        "uncertainty_abstraction.py",
        "robust_benchmark.py",
        "nominal_ablation.py",
        "nominal_chunk.py",
        "nominal_replay.py",
    ]
    saved = {name: (base / "src/behavioral_equivalence" / name).read_text() for name in names}
    saved.update(
        {
            "cloud.py": Path(__file__).read_text(),
            "config.json": config.read_text(),
            "test_nominal_replay.py": (base / "tests/test_nominal_replay.py").read_text(),
        }
    )
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def context():
    import json
    import hashlib

    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    root = Path("/artifacts") / config["campaign"]
    nominal = root / config["nominal_relative"]
    original = json.loads((nominal / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        != config["nominal_source_digest"]
    ):
        raise ValueError("Original nominal source changed")
    for name in original:
        if name in saved and name != "config.json" and original[name] != saved[name]:
            raise ValueError("Replay arithmetic differs from saved nominal implementation")
    protocol = json.loads((root / "protocol.json").read_text())
    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    if (
        hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        != config["original_protocol_digest"]
        or protocol["protocol_digest"] != config["original_protocol_digest"]
    ):
        raise ValueError("Frozen synthetic protocol changed")
    return root, nominal, protocol, saved, digest, config


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=120,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_chunk(chunk, calibration_sha, ablation_sha):
    import json
    from behavioral_equivalence.nominal_replay import replay_nominal_chunk

    volume.reload()
    root, nominal, protocol, saved, digest, config = context()
    if len(chunk) != 64 or any(c not in "0123456789abcdef" for c in chunk):
        raise ValueError("Frozen chunk identity required")
    out = nominal / ("replay-" + digest[:16])
    out.mkdir(exist_ok=True)
    target = out / (chunk + ".json")
    if target.exists():
        old = json.loads(target.read_text())
        if (
            old["replay_source_digest"] != digest
            or old["calibration_report_sha256"] != calibration_sha
            or old["ablation_report_sha256"] != ablation_sha
        ):
            raise ValueError("Cached replay differs from its original inputs")
        return old
    result = replay_nominal_chunk(
        root / "chunks" / chunk, nominal / "chunks" / chunk, protocol, calibration_sha, ablation_sha
    )
    result["replay_source_digest"] = digest
    target.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=3600,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run(smoke):
    import json
    import subprocess

    volume.reload()
    root, nominal, protocol, saved, digest, config = context()
    if (
        checksum(root / "manifest.json") != config["original_manifest_sha256"]
        or checksum(nominal / "manifest.json") != config["nominal_manifest_sha256"]
    ):
        raise ValueError("Frozen original or nominal manifest changed")
    original = json.loads((root / "manifest.json").read_text())
    manifest = json.loads((nominal / "manifest.json").read_text())
    if (
        original["status"] != "complete"
        or manifest["status"] != "complete"
        or len(manifest["chunks"]) != config["chunks"]
    ):
        raise ValueError("Complete frozen campaigns required")
    originals = {c["chunk_digest"]: c for c in original["chunks"]}
    chunks = manifest["chunks"]
    if {c["original_chunk_digest"] for c in chunks} != set(originals) or len(originals) != len(
        chunks
    ):
        raise ValueError("Nominal and retained-count chunk families differ")
    out = nominal / ("replay-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    test_output = None
    if smoke:
        tests = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-W",
                "error",
                "/research/tests/test_nominal_replay.py",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        test_output = tests.stdout
        chunks = [
            next(
                c
                for c in chunks
                if c["spec"]["size"] == 128 and c["spec"]["regime"] == "imbalanced_active"
            )
        ]
    elif json.loads((out / "smoke.json").read_text())["status"] != "complete":
        raise ValueError("Source-bound smoke required before full replay")
    jobs = []
    for chunk in chunks:
        identity = chunk["original_chunk_digest"]
        original_path = root / "chunks" / identity / "report.json"
        ablation_path = nominal / "chunks" / identity / "report.json"
        if (
            json.loads(original_path.read_text()) != originals[identity]
            or json.loads(ablation_path.read_text()) != chunk
        ):
            raise ValueError("Retained chunk report differs from frozen manifest")
        jobs.append((identity, checksum(original_path), checksum(ablation_path)))
    results, failures = [], []
    for result in audit_chunk.starmap(jobs, order_outputs=True, return_exceptions=True):
        if isinstance(result, Exception):
            failures.append(repr(result))
        else:
            results.append(result)
        (out / "progress.json").write_text(
            json.dumps(dict(complete=len(results), expected=len(jobs), failures=failures))
        )
        volume.commit()
    datasets = sum(r["verified_datasets"] for r in results)
    certificates = sum(r["verified_point_certificates"] for r in results)
    complete = not failures and (
        smoke or (datasets == config["datasets"] and certificates == config["point_certificates"])
    )
    report = dict(
        status="complete" if complete else "incomplete",
        source_digest=digest,
        phase="smoke" if smoke else "full",
        chunks=len(results),
        verified_datasets=datasets,
        verified_point_certificates=certificates,
        failures=failures,
        records=results,
        tests=test_output,
        nominal_manifest_sha256=config["nominal_manifest_sha256"],
        original_manifest_sha256=config["original_manifest_sha256"],
        scope=config["scope"],
        root=str(out),
    )
    (out / ("smoke.json" if smoke else "manifest.json")).write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k != "records"}


@app.local_entrypoint()
def main(smoke: bool = True):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    digest = snapshot()[1]
    # 401 worker timeouts, two coordinators, and startup allowance fit within2.10USD.
    reservation = BudgetLedger(ROOT / ".campaign/repeated-replay-budget.sqlite", 12).reserve(
        "nominal-replay-" + digest, 2.10
    )
    state = (
        ROOT
        / ".campaign"
        / ("nominal-replay-" + digest[:20] + ("-smoke.json" if smoke else "-full.json"))
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
