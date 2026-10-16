"""Validation-only user-variance planning; never opens final prediction scores."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-validation-precision-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_precision_planning.py",
        remote_path="/research/tests/test_precision_planning.py",
    )
    .add_local_file(
        ROOT / "configs/human-precision-inputs-v1.json", remote_path="/research/inputs.json"
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

    base = Path("/research") if Path("/research/inputs.json").exists() else ROOT
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "inputs.json": (
            base / "inputs.json"
            if base == Path("/research")
            else base / "configs/human-precision-inputs-v1.json"
        ).read_text(),
        "precision_planning.py": (
            base / "src/behavioral_equivalence/precision_planning.py"
        ).read_text(),
        "test_precision_planning.py": (base / "tests/test_precision_planning.py").read_text(),
    }
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=180,
    retries=0,
    volumes={"/artifacts": volume},
)
def run():
    import json
    import subprocess
    import numpy as np
    from behavioral_equivalence.precision_planning import (
        user_nll_differences,
        normal_precision_plan,
    )

    saved, digest = snapshot()
    tests = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_precision_planning.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    volume.reload()
    out = Path("/artifacts") / ("human-validation-precision-" + digest[:16])
    rows = []
    for spec in json.loads(saved["inputs.json"]):
        relative = Path(spec["root"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not str(relative).startswith("human-calendar-training-")
        ):
            raise ValueError("Private quantizer path required")
        root = Path("/artifacts") / relative
        if checksum(root / "report.json") != spec["report_sha256"]:
            raise ValueError("Frozen quantizer report changed")
        report = json.loads((root / "report.json").read_text())
        if (
            report["status"] != "complete"
            or report["protocol"] != spec["protocol"]
            or report["seed"] != spec["seed"]
            or checksum(root / "validation-predictions.npz")
            != report["output_sha256"]["validation-predictions.npz"]
        ):
            raise ValueError("Checksum-bound validation forecasts required")
        with np.load(root / "validation-predictions.npz", allow_pickle=False) as a:
            differences = user_nll_differences(
                a["target"], a["k256_probability"], a["k128_probability"], a["user_id"]
            )
            cases = len(a["target"])
        plan = normal_precision_plan(differences)
        rows.append(
            dict(
                protocol=spec["protocol"],
                seed=spec["seed"],
                validation_cases=cases,
                quantizer_report_sha256=spec["report_sha256"],
                **plan,
            )
        )
    result = dict(
        status="complete",
        source_digest=digest,
        tests=tests.stdout,
        rows=rows,
        comparison="K128 versus K256 separately fitted training-only quantizers; validation variance proxy, not nested-primary performance",
        final_test_scoring="none",
        scope="Approximate precision planning conditional on validation variance transporting to independent future user clusters. No finite-sample power promise, no final model decision and no across-seed independence assumption.",
        root=str(out),
    )
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    digest = snapshot()[1]
    reservation = BudgetLedger(ROOT / ".campaign/validation-precision-budget.sqlite", 0.5).reserve(
        digest, 180 * (1.31e-5 + 2 * 2.22e-6) + 0.05
    )
    state = ROOT / ".campaign" / ("validation-precision-" + digest[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = run.spawn()
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
