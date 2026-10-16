"""Repeated fixed-design transition sampling in known synthetic worlds."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-uncertainty-comparison-pilot-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .pip_install("scipy==1.15.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image, cpu=1, memory=2048, timeout=600, max_containers=1, volumes={"/artifacts": volume}
)
def run(run_id: str) -> dict:
    import json
    import time
    import hashlib
    import numpy as np
    from behavioral_equivalence.synthetic import FAMILIES, generate
    from behavioral_equivalence.core import Chain, quotient, reachability
    from behavioral_equivalence.uncertainty import (
        multinomial_intervals,
        robust_reachability,
        clopper_pearson_intervals,
    )

    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    out = Path("/artifacts") / run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    rng = np.random.default_rng(20261015)
    records = []
    for family in FAMILIES:
        world = generate(family, 32, 123)
        q, epsilon = quotient(world.chain, world.partition)
        if epsilon != 0:
            raise AssertionError("Pilot requires an exact ground-truth quotient")
        for trials in [100, 1000, 10000]:
            for repetition in range(20):
                counts = np.array([rng.multinomial(trials, row) for row in q.transitions])
                estimated = Chain(counts / trials, q.labels, q.initial)
                for method, builder in [
                    ("hoeffding", multinomial_intervals),
                    ("clopper_pearson", clopper_pearson_intervals),
                ]:
                    interval = builder(counts, q.labels, alpha=0.05)
                    kernel_included = bool(
                        np.all(interval.lower <= q.transitions)
                        and np.all(q.transitions <= interval.upper)
                    )
                    for horizon in [5, 25]:
                        for target in ["complete", "disengage"]:
                            nominal = reachability(q, target, horizon)
                            lower, upper = robust_reachability(interval, target, horizon)
                            truth = float(q.initial @ nominal)
                            lo, hi = float(q.initial @ lower), float(q.initial @ upper)
                            records.append(
                                {
                                    "family": family,
                                    "method": method,
                                    "trials_per_row": trials,
                                    "replicate": repetition,
                                    "horizon": horizon,
                                    "target": target,
                                    "true_probability": truth,
                                    "nominal_estimate": float(
                                        q.initial @ reachability(estimated, target, horizon)
                                    ),
                                    "lower": lo,
                                    "upper": hi,
                                    "width": hi - lo,
                                    "kernel_included": kernel_included,
                                    "outcome_covered": lo - 1e-12 <= truth <= hi + 1e-12,
                                }
                            )
        (out / "progress.json").write_text(json.dumps(records, indent=2))
        volume.commit()
    report = {
        "status": "complete",
        "phase": "engineering_pilot",
        "run_id": run_id,
        "records": records,
        "worlds": 5,
        "replications_per_world_sample_size": 20,
        "alpha": 0.05,
        "semantics": "row-rectangular per-step adversary",
        "confidence_sets": ["simultaneous coordinate Hoeffding", "simultaneous Clopper-Pearson"],
        "design": "paired identical fixed iid transition samples for both methods",
        "scope": "known synthetic kernels only; floating-point engineering checks",
        "limitations": [
            "small Monte Carlo replication count",
            "five worlds only",
            "no human confidence guarantee",
            "zero lower bound intervals not checked by PRISM",
        ],
        "wall_seconds": time.monotonic() - start,
    }
    files = sorted(Path("/research/src/behavioral_equivalence").glob("*.py"))
    report["source_digest"] = hashlib.sha256(
        b"".join(f.name.encode() + f.read_bytes() for f in files)
    ).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {"run_id": run_id, "records": len(records), "wall_seconds": report["wall_seconds"]}
