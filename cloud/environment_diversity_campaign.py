"""Source-bound exploratory controlled diversity matrix,400matched synthetic worlds."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-environment-diversity-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/environment-diversity-v1.json", remote_path="/research/config.json"
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
    import hashlib
    import json

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": (
            base / "config.json"
            if base == Path("/research")
            else base / "configs/environment-diversity-v1.json"
        ).read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(4096, 4096),
    timeout=600,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def world(spec):
    import json
    import gzip
    import numpy as np
    from behavioral_equivalence.environment_diversity import run_diversity_world

    volume.reload()
    saved, digest = snapshot()
    root = Path("/artifacts") / ("environment-diversity-" + digest[:16])
    name = f"{spec['family']}-{spec['size']}-{spec['seed']}-{spec['surface']}"
    out = root / name
    if (out / "completion.json").exists():
        cached = json.loads((out / "completion.json").read_text())
        if (
            cached["source_digest"] != digest
            or cached["spec"] != spec
            or any(checksum(out / n) != sha for n, sha in cached["output_sha256"].items())
        ):
            raise ValueError("Cached diversity world changed")
        return cached
    report, arrays = run_diversity_world(**spec)
    if (
        len(report["records"]) != 240
        or len(report["certificates"]) != 48
        or any(s["query_count"] != 8 for s in report["searches"])
    ):
        raise ValueError("Complete fixed-budget diversity output required")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "report.json.gz").open("wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as archive:
            archive.write(json.dumps(report, sort_keys=True).encode())
    np.savez_compressed(out / "models.npz", **arrays)
    result = dict(
        status="complete",
        source_digest=digest,
        spec=spec,
        metric_rows=len(report["records"]),
        certificates=len(report["certificates"]),
        wall_seconds=report["wall_seconds"],
        output_sha256={n: checksum(out / n) for n in ["report.json.gz", "models.npz"]},
        root=str(out),
    )
    (out / "completion.json").write_text(json.dumps(result, indent=2))
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
def full():
    import json

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    root = Path("/artifacts") / ("environment-diversity-" + digest[:16])
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    volume.commit()
    specs = [
        dict(family=f, size=n, seed=s, surface=surface)
        for f in config["families"]
        for n in config["sizes"]
        for s in config["seeds"]
        for surface in config["surfaces"]
    ]
    if len(specs) != config["expected_worlds"]:
        raise ValueError("Frozen complete diversity matrix required")
    rows = []
    failures = []
    for result in world.map(specs, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failures.append(dict(error_type=type(result).__name__, message=str(result)[:1000]))
        else:
            rows.append(result)
        (root / "progress.json").write_text(
            json.dumps(
                dict(completed=len(rows), failed=len(failures), expected=len(specs)), indent=2
            )
        )
        volume.commit()
    result = dict(
        status="complete" if len(rows) == len(specs) and not failures else "failed",
        source_digest=digest,
        worlds=rows,
        failures=failures,
        metric_rows=sum(r["metric_rows"] for r in rows),
        certificates=sum(r["certificates"] for r in rows),
        root=str(root),
        scope=config["scope"],
        phase=config["phase"],
    )
    (root / "manifest.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    if result["status"] != "complete":
        raise RuntimeError(
            "Diversity matrix incomplete; retain original manifest and source-bound handles"
        )
    if (
        result["metric_rows"] != config["expected_metric_rows"]
        or result["certificates"] != config["expected_targeted_certificates"]
    ):
        raise ValueError("Diversity output totals disagree")
    return {k: v for k, v in result.items() if k != "worlds"}


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    reservation = BudgetLedger(ROOT / ".campaign/environment-diversity-budget.sqlite", 8).reserve(
        "full-" + digest, config["resource_budget"]["reservation_full_usd"]
    )
    state = ROOT / ".campaign" / ("environment-diversity-" + digest[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = full.spawn()
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
