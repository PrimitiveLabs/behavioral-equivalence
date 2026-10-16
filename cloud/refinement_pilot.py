"""Equal-budget synthetic intervention search and withheld-environment checks."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-refinement-pilot-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
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
    from behavioral_equivalence.synthetic import FAMILIES, generate, intervene
    from behavioral_equivalence.core import exact_partition, quotient, reachability
    from behavioral_equivalence.refinement import budgeted_witness, refine

    if not run_id.replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    out = Path("/artifacts") / run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    records = []
    for family in FAMILIES:
        for seed in [0, 1, 2]:
            world = generate(family, 32, seed)
            original = exact_partition([world.chain])
            for method in ["random", "grid", "adaptive"]:
                witness = budgeted_witness(world, original, method, 8, 10, seed=seed)
                refined = refine(world, witness, 0.01)
                for heldout in [0.073, 0.173, 0.273]:
                    changed = intervene(world, heldout)
                    before, before_epsilon = quotient(changed, original)
                    after, after_epsilon = quotient(changed, refined)
                    truth = reachability(changed, "complete", 10)

                    def error(model, blocks):
                        predicted = reachability(model, "complete", 10)
                        return max(
                            abs(float(truth[s] - predicted[i]))
                            for i, b in enumerate(blocks)
                            for s in b
                        )

                    records.append(
                        {
                            "family": family,
                            "seed": seed,
                            "method": method,
                            "search_budget": 8,
                            "actual_environment_evaluations": witness.environment_evaluations,
                            "witness_strength": witness.strength,
                            "js_nats": witness.js_nats,
                            "heldout_strength": heldout,
                            "states_before": len(original),
                            "states_after": len(refined),
                            "error_before": error(before, original),
                            "error_after": error(after, refined),
                            "epsilon_before": before_epsilon,
                            "epsilon_after": after_epsilon,
                        }
                    )
    report = {
        "status": "complete",
        "phase": "engineering_pilot",
        "records": records,
        "worlds": 15,
        "scope": "known synthetic environments only",
        "limitations": [
            "one-dimensional intervention family",
            "endpoint grid is a strong baseline in this monotone family",
            "only environment-evaluation budgets compared; no per-method runtime claim",
            "no matched-capacity extra-data controls yet",
            "no human intervention evidence",
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
