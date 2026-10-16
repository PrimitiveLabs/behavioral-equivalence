"""Validate supplementary outcome/head code in a pinned optional-dependency image."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-supplementary-validation-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1", "pytest==8.4.1")
    .env({"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
    .add_local_file(
        ROOT / "tools/supplementary_outcomes.py",
        remote_path="/research/tools/supplementary_outcomes.py",
    )
    .add_local_file(
        ROOT / "tools/supplementary_heads.py", remote_path="/research/tools/supplementary_heads.py"
    )
    .add_local_file(
        ROOT / "tests/test_supplementary_outcomes.py",
        remote_path="/research/tests/test_supplementary_outcomes.py",
    )
    .add_local_file(
        ROOT / "tests/test_supplementary_heads.py",
        remote_path="/research/tests/test_supplementary_heads.py",
    )
)


def source_identity():
    import hashlib
    import json

    base = Path("/research") if Path("/research/tools/supplementary_heads.py").exists() else ROOT
    source = {"cloud.py": Path(__file__).read_text()}
    for name in [
        "tools/supplementary_outcomes.py",
        "tools/supplementary_heads.py",
        "tests/test_supplementary_outcomes.py",
        "tests/test_supplementary_heads.py",
    ]:
        source[name] = (base / name).read_text()
    return source, hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=300,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def validate():
    import json
    import subprocess
    import time

    source, digest = source_identity()
    start = time.monotonic()
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "-r",
            "s",
            "/research/tests/test_supplementary_outcomes.py",
            "/research/tests/test_supplementary_heads.py",
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    report = dict(
        status="complete"
        if result.returncode == 0 and "skipped" not in result.stdout
        else "failed",
        source_digest=digest,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        wall_seconds=time.monotonic() - start,
        scope="Pinned implementation tests, including fitting absent categories and missing labels. No human data, model inference API or human model training.",
    )
    out = Path("/artifacts") / ("supplementary-validation-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(source, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    source, digest = source_identity()
    state = ROOT / ".campaign" / ("supplementary-validation-" + digest[:16] + ".json")
    if state.exists():
        saved = json.loads(state.read_text())
        call = modal.FunctionCall.from_id(saved["function_call_id"])
    else:
        print(
            json.dumps(
                BudgetLedger(ROOT / ".campaign/human-case-audit-budget.sqlite", 4).reserve(
                    "supplementary-validation-" + digest, 0.03
                )
            ),
            flush=True,
        )
        with state.with_suffix(".submission-attempt").open("x") as marker:
            marker.write(digest + "\n")
        call = validate.spawn()
        saved = dict(source_digest=digest, function_call_id=call.object_id, reservation_usd=0.03)
        state.write_text(json.dumps(saved, indent=2))
    print(json.dumps(saved), flush=True)
    (
        ROOT / "reports/source-snapshots" / ("supplementary-validation-" + digest[:16] + ".json")
    ).write_text(json.dumps(source, indent=2) + "\n")
    report = call.get()
    (ROOT / "reports/preflight" / ("supplementary-validation-" + digest[:16] + ".json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
    if report["status"] != "complete":
        raise RuntimeError("Supplementary validation failed; retain this attempt")
