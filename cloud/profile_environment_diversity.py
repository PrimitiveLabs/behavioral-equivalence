"""Cloud-only runtime profile before the fixed-budget diversity ablation matrix."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-diversity-profile-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_environment_diversity.py",
        remote_path="/research/tests/test_environment_diversity.py",
    )
)


def snapshot():
    import json
    import hashlib

    base = Path("/research") if Path("/research/src").exists() else ROOT
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "test_environment_diversity.py": (base / "tests/test_environment_diversity.py").read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(4096, 4096),
    timeout=600,
    retries=0,
    volumes={"/artifacts": volume},
)
def profile():
    import subprocess
    import json
    import numpy as np
    from behavioral_equivalence.environment_diversity import run_diversity_world

    saved, digest = snapshot()
    tests = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_environment_diversity.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    root = Path("/artifacts") / ("environment-diversity-profile-" + digest[:16])
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    rows = []
    for family, surface in [("trust", "localized"), ("memory", "oscillatory")]:
        report, arrays = run_diversity_world(family, 128, 44999, surface)
        name = family + "-" + surface
        (root / (name + ".json")).write_text(json.dumps(report, indent=2))
        np.savez_compressed(root / (name + ".npz"), **arrays)
        rows.append(
            dict(
                family=family,
                surface=surface,
                size=128,
                seed=44999,
                wall_seconds=report["wall_seconds"],
                certificates=len(report["certificates"]),
                metric_rows=len(report["records"]),
            )
        )
    result = dict(
        status="complete",
        source_digest=digest,
        tests=tests.stdout,
        rows=rows,
        root=str(root),
        scope="Excluded pilot seed44999runtime and implementation profile; no confirmatory performance claim. Main diversity extension remains unexecuted.",
    )
    (root / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    digest = snapshot()[1]
    reservation = BudgetLedger(ROOT / ".campaign/environment-diversity-budget.sqlite", 8).reserve(
        "profile-" + digest, 0.1
    )
    state = ROOT / ".campaign" / ("diversity-profile-" + digest[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = profile.spawn()
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
