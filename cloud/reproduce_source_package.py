"""Build a source-only release wheel and test it in a fresh pinned cloud image."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-source-package-reproduction-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==2.2.6",
        "scipy==1.15.3",
        "pydantic==2.11.7",
        "pytest==8.4.1",
        "ruff==0.12.0",
        "setuptools==80.9.0",
        "wheel==0.45.1",
        "torch==2.7.1",
        "scikit-learn==1.7.1",
        "polars==1.31.0",
        "duckdb==1.3.2",
        "pyarrow==20.0.0",
    )
    .apt_install("default-jre-headless", "curl")
    .run_commands(
        "curl -fL https://github.com/prismmodelchecker/prism/releases/download/v4.11/prism-4.11-linux64-x86.tar.gz -o /tmp/prism.tar.gz",
        "echo '6d4677b6ab64daa16a8bb5ad05f79c46313ad67835cb8445ee770e7f57f611c8  /tmp/prism.tar.gz' | sha256sum -c -",
        "mkdir /opt/prism && tar -xzf /tmp/prism.tar.gz -C /opt/prism --strip-components=1",
        "cd /opt/prism && ./install.sh",
    )
    .env(
        {
            "OPENBLAS_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "POLARS_MAX_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
)


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def reproduce(archive, expected_digest):
    import io
    import os
    import json
    import re
    import hashlib
    import subprocess
    import tempfile
    import tarfile
    import importlib.metadata
    import time

    if hashlib.sha256(archive).hexdigest() != expected_digest:
        raise ValueError("Source archive identity changed")
    started = time.monotonic()
    checks = []
    with tempfile.TemporaryDirectory(prefix="source-reproduction-") as temporary:
        base = Path(temporary)
        source = base / "source"
        source.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            for member in tar.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or not member.isfile():
                    raise ValueError("Unsafe non-file archive member")
            tar.extractall(source, filter="data")
        manifest = json.loads((source / "reports/source-manifest.json").read_text())
        for name, sha in manifest["files"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Extracted source hash changed")
        env = dict(os.environ, PYTHONPATH="")

        def run(command, where=source):
            result = subprocess.run(command, cwd=where, env=env, capture_output=True, text=True)
            checks.append(
                dict(
                    command=command,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
            return result

        import sys

        wheel_dir = base / "wheels"
        wheel_dir.mkdir()
        build = run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                str(source),
                "-w",
                str(wheel_dir),
            ]
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if build.returncode or len(wheels) != 1:
            result = dict(status="failed", stage="wheel_build", checks=checks)
        else:
            venv = base / "test-env"
            create = run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)])
            python = str(venv / "bin/python")
            install = (
                run([python, "-m", "pip", "install", "--no-deps", str(wheels[0])])
                if not create.returncode
                else create
            )
            probe = (
                run(
                    [
                        python,
                        "-c",
                        "import behavioral_equivalence; print(behavioral_equivalence.__file__)",
                    ]
                )
                if not install.returncode
                else install
            )
            imported_from_wheel = (
                str(venv) in probe.stdout and str(source / "src") not in probe.stdout
            )
            tests = (
                run([python, "-m", "pytest", "-q", "-W", "error", "-o", "pythonpath=", "tests"])
                if imported_from_wheel
                else probe
            )
            skipped = re.search(r"\b(\d+) skipped\b", tests.stdout)
            passed = re.search(r"\b(\d+) passed\b", tests.stdout)
            result = dict(
                status="complete"
                if imported_from_wheel and tests.returncode == 0 and not skipped
                else "failed",
                stage="installed_wheel_tests",
                checks=checks,
                imported_from_wheel=imported_from_wheel,
                tests_passed=int(passed.group(1)) if passed else 0,
                tests_skipped=int(skipped.group(1)) if skipped else 0,
                wheel_sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
            )
        scientific = base / "scientific-rerun"
        if result["status"] == "complete":
            replay = run([python, "tools/scientific_reproduction.py", "--output", str(scientific)])
            if replay.returncode:
                result.update(status="failed", stage="scientific_reproduction")
            else:
                measured = json.loads((scientific / "report.json").read_text())
                result["scientific_reproduction"] = {
                    k: v
                    for k, v in measured.items()
                    if k not in {"world_results", "external", "artifacts"}
                }
        dependencies = {
            name: importlib.metadata.version(name)
            for name in [
                "numpy",
                "scipy",
                "pydantic",
                "pytest",
                "ruff",
                "setuptools",
                "wheel",
                "torch",
                "scikit-learn",
                "polars",
                "duckdb",
                "pyarrow",
            ]
        }
        result.update(
            source_bundle_sha256=expected_digest,
            source_files=len(manifest["files"]),
            source_hashes_verified=True,
            checker_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            dependencies=dependencies,
            wall_seconds=time.monotonic() - started,
            scope="Fresh pinned cloud dependencies, independently extracted source, built wheel and original tests against installed wheel with source-path injection disabled, plus a small all-five-mechanism/six-method scientific rerun with pinned PRISM. No human rows, weights, full 1200-world rerun or independently authored proof review.",
        )
        root = Path("/artifacts") / ("source-package-reproduction-" + expected_digest[:16])
        root.mkdir(exist_ok=True)
        if scientific.exists():
            import shutil

            shutil.copytree(scientific, root / "scientific-rerun", dirs_exist_ok=True)
        if len(wheels) == 1:
            (root / wheels[0].name).write_bytes(wheels[0].read_bytes())
        (root / "checker-source.py").write_bytes(Path(__file__).read_bytes())
        result["root"] = str(root)
        (root / "report.json").write_text(json.dumps(result, indent=2))
        volume.commit()
        return result


@app.local_entrypoint()
def main():
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    from behavioral_equivalence.budget import BudgetLedger
    from run_final_human_jobs import launcher_lock, claim_submission

    with launcher_lock(ROOT / ".campaign/source-package-launcher.lock"):
        package = json.loads((ROOT / "reports/current-source-package.json").read_text())
        raw = (ROOT / package["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != package["sha256"]:
            raise ValueError("Source pointer changed")
        checker = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        identity = "source-package-" + package["sha256"] + "-" + checker
        reservation = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40).reserve(
            identity, 0.14
        )
        state = ROOT / ".campaign" / (identity + ".json")
        if state.exists():
            call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
        else:
            claim_submission(state.with_suffix(".submission-attempt"))
            call = reproduce.spawn(raw, package["sha256"])
            state.write_text(
                json.dumps(
                    dict(
                        function_call_id=call.object_id,
                        reservation=reservation,
                        source_bundle_sha256=package["sha256"],
                        checker_source_sha256=checker,
                    ),
                    indent=2,
                )
            )
        print(state.read_text(), flush=True)
        result = call.get()
        (
            ROOT
            / "reports/preflight"
            / (
                "source-package-reproduction-"
                + package["sha256"][:16]
                + "-"
                + checker[:12]
                + ".json"
            )
        ).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
