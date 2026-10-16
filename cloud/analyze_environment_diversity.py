"""Replay persisted diversity certificates and aggregate every paired world."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-diversity-analysis-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_diversity_analysis.py",
        remote_path="/research/tests/test_diversity_analysis.py",
    )
    .add_local_file(
        ROOT / "configs/environment-diversity-analysis-v1.json", remote_path="/research/config.json"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def snapshot():
    import json
    import hashlib

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": (
            base / "config.json"
            if base == Path("/research")
            else base / "configs/environment-diversity-analysis-v1.json"
        ).read_text(),
        "test_diversity_analysis.py": (base / "tests/test_diversity_analysis.py").read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=180, retries=0)
def smoke():
    import subprocess

    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_diversity_analysis.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(status="complete", source_digest=snapshot()[1], tests=result.stdout)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(4096, 4096),
    timeout=180,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def audit_world(completion):
    import json
    import gzip
    import numpy as np
    from behavioral_equivalence.diversity_analysis import (
        diversity_world_means,
        replay_diversity_certificates,
    )

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    relative = Path(completion["root"])
    if relative.is_absolute():
        if not relative.is_relative_to("/artifacts/" + config["campaign_root"]):
            raise ValueError("World escaped frozen campaign")
        root = relative
    else:
        raise ValueError("Absolute frozen world artifact path required")
    if (
        ".." in relative.parts
        or completion["status"] != "complete"
        or completion["source_digest"] != config["campaign_source_digest"]
        or any(checksum(root / name) != sha for name, sha in completion["output_sha256"].items())
    ):
        raise ValueError("Checksum-bound complete world required")
    out = Path("/artifacts") / config["campaign_root"] / ("analysis-" + digest[:16]) / root.name
    if (out / "audit.json").exists():
        cached = json.loads((out / "audit.json").read_text())
        if (
            cached["analysis_source_digest"] != digest
            or cached["world_report_sha256"] != completion["output_sha256"]["report.json.gz"]
            or cached["world_models_sha256"] != completion["output_sha256"]["models.npz"]
            or cached["certificate_replay_count"] != 48
        ):
            raise ValueError("Cached world audit changed")
        return cached
    with gzip.open(root / "report.json.gz", "rt") as f:
        report = json.load(f)
    if any(report[key] != value for key, value in completion["spec"].items()):
        raise ValueError("World report spec differs from manifest")
    means = diversity_world_means(report)
    with np.load(root / "models.npz", allow_pickle=False) as arrays:
        replayed = replay_diversity_certificates(report, arrays)
    if replayed != 48:
        raise ValueError("All span/context certificates require replay")
    result = dict(
        status="complete",
        source_digest=config["campaign_source_digest"],
        analysis_source_digest=digest,
        spec=completion["spec"],
        world_report_sha256=completion["output_sha256"]["report.json.gz"],
        world_models_sha256=completion["output_sha256"]["models.npz"],
        certificate_replay_count=replayed,
        means=means,
        searches=report["searches"],
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=3600,
    retries=0,
    volumes={"/artifacts": volume},
)
def full(manifest_sha256: str):
    import json
    import hashlib
    from behavioral_equivalence.diversity_analysis import summarize_diversity

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    root = Path("/artifacts") / config["campaign_root"]
    if checksum(root / "manifest.json") != manifest_sha256:
        raise ValueError("Pinned terminal campaign manifest required")
    manifest = json.loads((root / "manifest.json").read_text())
    parent_source = json.loads((root / "source.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["failures"]
        or manifest["source_digest"] != config["campaign_source_digest"]
        or hashlib.sha256(json.dumps(parent_source, sort_keys=True).encode()).hexdigest()
        != config["campaign_source_digest"]
        or len(manifest["worlds"]) != config["expected_worlds"]
    ):
        raise ValueError("Full frozen source-bound campaign required")
    for name in [
        "environment_diversity.py",
        "synthetic.py",
        "core.py",
        "refinement.py",
        "intervention_space.py",
        "catalogue_search.py",
        "integer_bounds.py",
        "uncertainty.py",
    ]:
        key = "src/behavioral_equivalence/" + name
        if saved[key] != parent_source[key]:
            raise ValueError("Frozen model/certificate implementation changed")
    out = root / ("analysis-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    volume.commit()
    rows = []
    failures = []
    for result in audit_world.map(manifest["worlds"], order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failures.append(dict(error_type=type(result).__name__, message=str(result)[:1000]))
        else:
            rows.append(result)
        (out / "progress.json").write_text(
            json.dumps(
                dict(completed=len(rows), failed=len(failures), expected=config["expected_worlds"]),
                indent=2,
            )
        )
        volume.commit()
    audit = dict(
        status="complete" if len(rows) == config["expected_worlds"] and not failures else "failed",
        analysis_source_digest=digest,
        campaign_manifest_sha256=manifest_sha256,
        worlds=rows,
        failures=failures,
    )
    (out / "audit-manifest.json").write_text(json.dumps(audit, indent=2))
    volume.commit()
    if audit["status"] != "complete":
        raise RuntimeError("Diversity audit incomplete; preserve all original handles")
    summary = summarize_diversity(
        rows, json.loads(parent_source["config.json"]), config["campaign_source_digest"]
    )
    if summary["certificates_replayed"] != config["expected_certificates"]:
        raise ValueError("Certificate matrix incomplete")
    summary.update(
        analysis_source_digest=digest,
        campaign_manifest_sha256=manifest_sha256,
        audit_manifest_sha256=checksum(out / "audit-manifest.json"),
        root=str(out),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return {k: v for k, v in summary.items() if k != "comparisons"}


@app.local_entrypoint()
def main(stage: str = "smoke", manifest_sha256: str = ""):
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    if stage not in {"smoke", "full"}:
        raise ValueError("Smoke or full stage required")
    if stage == "full" and (
        len(manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in manifest_sha256)
    ):
        raise ValueError("Pinned complete manifest SHA256 required")
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    identity = hashlib.sha256(json.dumps([stage, digest, manifest_sha256]).encode()).hexdigest()
    amount = 0.03 if stage == "smoke" else config["analysis_reservation_usd"]
    reservation = BudgetLedger(ROOT / ".campaign/environment-diversity-budget.sqlite", 8).reserve(
        identity, amount
    )
    state = ROOT / ".campaign" / ("diversity-analysis-" + identity[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = smoke.spawn() if stage == "smoke" else full.spawn(manifest_sha256)
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
