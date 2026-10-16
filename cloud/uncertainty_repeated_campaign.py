"""Repeated fixed-world calibration, row imbalance and terminal-support controls."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-uncertainty-repeated-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)
SOURCE_FILES = [
    "__init__.py",
    "core.py",
    "synthetic.py",
    "uncertainty.py",
    "integer_bounds.py",
    "uncertainty_abstraction.py",
    "confidence.py",
    "robust_benchmark.py",
    "uncertainty_repetition.py",
]


def source_snapshot():
    import hashlib
    import json

    base = (
        Path("/research/src/behavioral_equivalence")
        if Path("/research/src").exists()
        else ROOT / "src/behavioral_equivalence"
    )
    saved = {name: (base / name).read_text() for name in SOURCE_FILES}
    saved["uncertainty_repeated_campaign.py"] = Path(__file__).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def validate(protocol):
    import hashlib
    import json

    _, source = source_snapshot()
    semantic = {k: v for k, v in protocol.items() if k not in {"campaign_id", "protocol_digest"}}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    if (
        protocol["source_digest"] != source
        or protocol["protocol_digest"] != digest
        or protocol["campaign_id"] != "uncertainty-repeated-v1-" + digest[:12]
    ):
        raise ValueError("Repeated-calibration protocol or source changed after freeze")


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=600,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def worker(spec: dict, protocol: dict):
    import gzip
    import hashlib
    import json
    import time
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.synthetic import generate
    from behavioral_equivalence.confidence import certified_hoeffding, verify_hoeffding
    from behavioral_equivalence.uncertainty import clopper_pearson_intervals
    from behavioral_equivalence.robust_benchmark import assess_intervals
    from behavioral_equivalence.uncertainty_repetition import (
        row_sample_schedule,
        terminal_label_support,
    )

    validate(protocol)
    volume.reload()
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    out = Path("/artifacts") / protocol["campaign_id"] / "chunks" / identity
    if (out / "report.json").exists():
        old = json.loads((out / "report.json").read_text())
        if (
            old["source_digest"] != protocol["source_digest"]
            or old["protocol_digest"] != protocol["protocol_digest"]
            or old["spec"] != spec
        ):
            raise ValueError("Cached repeated-calibration source changed")
        if any(
            hashlib.sha256((out / name).read_bytes()).hexdigest() != checksum
            for name, checksum in old["artifact_sha256"].items()
        ):
            raise ValueError("Cached repeated-calibration evidence changed")
        return old
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    world = generate(spec["family"], spec["size"], spec["seed"], protocol["perturbation"])
    schedule = row_sample_schedule(world.chain, spec["regime"], protocol["samples_per_active"])
    active = np.array(world.chain.labels) == "active"
    mapping = np.array(
        [next(i for i, b in enumerate(world.partition) if s in b) for s in range(world.chain.size)]
    )
    records, artifacts = [], {}
    for repetition in range(spec["first_repetition"], spec["last_repetition"]):
        rng = np.random.default_rng(
            [
                20261015,
                protocol["families"].index(spec["family"]),
                spec["size"],
                spec["seed"],
                protocol["regimes"].index(spec["regime"]),
                repetition,
            ]
        )
        counts = np.array(
            [rng.multinomial(int(m), row) for m, row in zip(schedule, world.chain.transitions)]
        )
        evidence = {
            "counts": counts.tolist(),
            "row_sample_schedule": schedule.tolist(),
            "methods": [],
        }
        for method in ["certified_hoeffding", "clopper_pearson_float_quantiles"]:
            if method == "certified_hoeffding":
                uncertain, confidence = certified_hoeffding(
                    counts, world.chain.labels, Fraction(1, 20)
                )
                verify_hoeffding(counts, world.chain.labels, confidence)
            else:
                uncertain = clopper_pearson_intervals(counts, world.chain.labels, 0.05)
                confidence = {
                    "statistical_numeric_scope": "floating beta quantiles; no outward statistical-tail certificate"
                }
            for support in ["unknown_terminal_rows", "known_closed_terminal_labels"]:
                allowed = (
                    uncertain
                    if support == "unknown_terminal_rows"
                    else terminal_label_support(uncertain, world.chain)
                )
                metrics, certificate = assess_intervals(
                    world.chain, allowed, world.partition, protocol["horizons"], protocol["bits"]
                )
                widths = []
                for h in protocol["horizons"]:
                    values = certificate["robust_values"]["complete"]["values"][str(h)]
                    delta = np.array(values["max_upper"], dtype=float) - np.array(
                        values["min_lower"], dtype=float
                    )
                    width = delta[mapping] / (1 << protocol["bits"])
                    widths.append(
                        {
                            "horizon": h,
                            "mean_active_width": float(width[active].mean()),
                            "initial_weighted_width": float(np.sum(world.chain.initial * width)),
                        }
                    )
                record = {
                    "repetition": repetition,
                    "method": method,
                    "support": support,
                    **metrics,
                    "completion_widths": widths,
                }
                records.append(record)
                evidence["methods"].append(
                    {
                        "method": method,
                        "support": support,
                        "confidence": confidence,
                        "certificate": certificate,
                    }
                )
        name = f"repetition-{repetition:03d}.json.gz"
        packed = gzip.compress(json.dumps(evidence, sort_keys=True).encode(), mtime=0)
        (out / name).write_bytes(packed)
        artifacts[name] = hashlib.sha256(packed).hexdigest()
    result = {
        "status": "complete",
        "chunk_digest": identity,
        "source_digest": protocol["source_digest"],
        "protocol_digest": protocol["protocol_digest"],
        "spec": spec,
        "world_digest": world.chain.digest(),
        "row_sample_schedule": schedule.tolist(),
        "records": records,
        "artifact_sha256": artifacts,
        "wall_seconds": time.monotonic() - started,
        "scope": "Independent row samples with schedules fixed from the synthetic world before draws. Terminal support is extra known specification information, not learned from observed zeros.",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=86400,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run(protocol: dict, smoke: bool):
    import itertools
    import json

    validate(protocol)
    out = Path("/artifacts") / protocol["campaign_id"]
    out.mkdir(exist_ok=True)
    saved, _ = source_snapshot()
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    volume.commit()
    if smoke:
        specs = [
            dict(
                family="trust",
                size=128,
                seed=30000,
                regime="imbalanced_active",
                first_repetition=0,
                last_repetition=2,
            )
        ]
    else:
        specs = [
            dict(
                family=f,
                size=n,
                seed=s,
                regime=r,
                first_repetition=i,
                last_repetition=i + protocol["chunk_size"],
            )
            for f, n, s, r, i in itertools.product(
                protocol["families"],
                protocol["sizes"],
                protocol["seeds"],
                protocol["regimes"],
                range(0, protocol["repetitions"], protocol["chunk_size"]),
            )
        ]
    records, failures = [], []
    for result in worker.map(
        specs, [protocol] * len(specs), return_exceptions=True, order_outputs=True
    ):
        if isinstance(result, Exception):
            failures.append(repr(result))
        else:
            records.append(result)
        if len(records) % 10 == 0:
            (out / "progress.json").write_text(
                json.dumps(
                    {
                        "chunks_complete": len(records),
                        "chunks_expected": len(specs),
                        "failures": failures,
                    }
                )
            )
            volume.commit()
    result = {
        "status": "incomplete" if failures else "complete",
        "phase": "smoke" if smoke else "frozen repeated synthetic calibration",
        "source_digest": protocol["source_digest"],
        "protocol_digest": protocol["protocol_digest"],
        "chunks": records,
        "failures": failures,
        "datasets": sum(len(r["artifact_sha256"]) for r in records),
        "method_dataset_evaluations": sum(len(r["records"]) for r in records),
    }
    (out / ("smoke-manifest.json" if smoke else "manifest.json")).write_text(
        json.dumps(result, indent=2)
    )
    volume.commit()
    return {k: v for k, v in result.items() if k != "chunks"}


@app.local_entrypoint()
def main(protocol_path: str, smoke: bool = True):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    protocol = json.loads(Path(protocol_path).read_text())
    validate(protocol)
    if not smoke:
        amount = 400 * 600 * (2 * 1.31e-5 + 4 * 2.22e-6) + 86400 * (0.5 * 1.31e-5 + 2.22e-6) + 2
        print(
            json.dumps(
                BudgetLedger(ROOT / ".campaign/uncertainty-repeated-budget.sqlite", 15).reserve(
                    protocol["campaign_id"], amount
                )
            ),
            flush=True,
        )
    handle = (
        ROOT / ".campaign" / (protocol["campaign_id"] + ("-smoke" if smoke else "-full") + ".json")
    )
    if handle.exists():
        call = modal.FunctionCall.from_id(json.loads(handle.read_text())["function_call_id"])
        try:
            print(json.dumps(call.get(timeout=0)), flush=True)
        except TimeoutError:
            print(
                "Existing repeated-calibration call remains pending; no duplicate submission.",
                flush=True,
            )
        return
    call = run.spawn(protocol, smoke)
    record = {"function_call_id": call.object_id, "function_id": run.object_id}
    handle.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
