"""Bounded detached CPU pilot. Results persist without the local client."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-pilot-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("default-jre-headless", "curl")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pydantic==2.11.7")
    .run_commands(
        "curl -fL https://github.com/prismmodelchecker/prism/releases/download/v4.11/"
        "prism-4.11-linux64-x86.tar.gz -o /tmp/prism.tar.gz",
        "echo '6d4677b6ab64daa16a8bb5ad05f79c46313ad67835cb8445ee770e7f57f611c8"
        "  /tmp/prism.tar.gz' | sha256sum -c -",
        "mkdir /opt/prism && tar -xzf /tmp/prism.tar.gz -C /opt/prism --strip-components=1",
        "cd /opt/prism && ./install.sh",
    )
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image, cpu=2, memory=2048, timeout=900, max_containers=1, volumes={"/artifacts": volume}
)
def run_pilot(run_id: str) -> dict:
    import hashlib
    import json
    import platform
    import time
    import numpy as np
    from behavioral_equivalence.core import (
        approximate_partition,
        coupling_bound,
        exact_partition,
        quotient,
        rational_reachability,
        reachability,
    )
    from behavioral_equivalence.synthetic import FAMILIES, generate, intervene
    from behavioral_equivalence.prism import check, write_model
    from behavioral_equivalence.uncertainty import IntervalChain, robust_reachability

    # The caller supplies a filesystem-safe immutable run ID.
    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    destination = Path("/artifacts") / run_id
    destination.mkdir(exist_ok=False)
    start = time.monotonic()
    records = []
    for family in FAMILIES:
        for perturbation in (0.0, 0.01):
            world = generate(family, 32, 20261015, perturbation)
            blocks = (
                exact_partition([world.chain])
                if perturbation == 0
                else approximate_partition([world.chain], 0.02)
            )
            reduced, epsilon = quotient(world.chain, blocks)
            stem = f"{family}-{perturbation}"
            model_file = destination / f"{stem}.prism"
            # Cross-check all concrete states against their abstract block.
            write_model(reduced, model_file)
            for horizon in (5, 10, 25, 50):
                for target in ("complete", "disengage"):
                    full = reachability(world.chain, target, horizon)
                    abstract = reachability(reduced, target, horizon)
                    error = max(
                        abs(float(full[s] - abstract[i])) for i, b in enumerate(blocks) for s in b
                    )
                    bound = coupling_bound(epsilon, horizon)
                    prism = check(
                        model_file, target, horizon, destination / f"{stem}-{target}-{horizon}.log"
                    )
                    if abs(prism - float(abstract[0])) > 1e-10:
                        raise AssertionError("PRISM disagrees with dynamic programming")
                    if error > bound + 1e-10:
                        raise AssertionError("Observed discrepancy exceeds numerical bound")
                    records.append(
                        {
                            "family": family,
                            "perturbation": perturbation,
                            "horizon": horizon,
                            "target": target,
                            "original_states": world.chain.size,
                            "abstract_states": reduced.size,
                            "epsilon": epsilon,
                            "actual_max_error": error,
                            "bound": bound,
                            "prism": prism,
                            "dp": float(abstract[0]),
                            "world_digest": world.chain.digest(),
                        }
                    )
            if perturbation == 0:
                rational = rational_reachability(reduced, "complete", 5)
                if (
                    max(
                        abs(float(v) - x)
                        for v, x in zip(rational, reachability(reduced, "complete", 5))
                    )
                    > 1e-12
                ):
                    raise AssertionError("Exact arithmetic oracle disagrees")
                changed = intervene(world, 0.3)
                refined = approximate_partition([world.chain, changed], 0.02)
                witness = {
                    "before_epsilon": quotient(changed, blocks)[1],
                    "after_epsilon": quotient(changed, refined)[1],
                    "before_states": len(blocks),
                    "after_states": len(refined),
                }
                (destination / f"{family}-intervention.json").write_text(json.dumps(witness))
    (destination / "nominal-checkpoint.json").write_text(json.dumps(records, indent=2))
    volume.commit()
    uncertain = IntervalChain(
        np.array([[0.1, 0.2], [0.0, 1.0]]),
        np.array([[0.8, 0.9], [0.0, 1.0]]),
        ("active", "complete"),
    )
    uncertainty_file = destination / "uncertainty.prism"
    write_model(uncertain, uncertainty_file)
    lower, upper = robust_reachability(uncertain, "complete", 5)
    for mode, expected in (("min", lower[0]), ("max", upper[0])):
        value = check(
            uncertainty_file, "complete", 5, destination / f"robust-{mode}.log", extremum=mode
        )
        if abs(value - expected) > 1e-10:
            raise AssertionError("Robust verification disagrees with PRISM")
    report = {
        "status": "complete",
        "scope": "engineering pilot, not confirmatory evidence",
        "records": records,
        "wall_seconds": time.monotonic() - start,
        "backend": "PRISM 4.11",
        "python": platform.python_version(),
        "numerical_tolerance": 1e-10,
        "formal_numeric_soundness": "not established",
    }
    encoded = json.dumps(report, indent=2).encode()
    (destination / "report.json").write_bytes(encoded)
    manifest = {
        "run_id": run_id,
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "records": len(records),
        "wall_seconds": report["wall_seconds"],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return manifest
