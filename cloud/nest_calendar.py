"""Source-bound nested abstractions of completed human point-model parents."""

from pathlib import Path, PurePosixPath
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-human-nested-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-nested-abstraction-v1.json",
        remote_path="/research/nested-config.json",
    )
    .add_local_file(
        ROOT / "tests/test_nested_design.py", remote_path="/research/test_nested_design.py"
    )
)


def checksum(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    import hashlib
    import json

    remote = Path("/research/nested-config.json").exists()
    base = Path("/research") if remote else ROOT
    saved = {
        "nest_calendar.py": Path(__file__).read_text(),
        "config.json": (
            base / ("nested-config.json" if remote else "configs/human-nested-abstraction-v1.json")
        ).read_text(),
        "test_nested_design.py": (
            base / ("test_nested_design.py" if remote else "tests/test_nested_design.py")
        ).read_text(),
    }
    for name in [
        "core.py",
        "clustering.py",
        "decision_model.py",
        "integer_bounds.py",
        "uncertainty.py",
        "rational.py",
        "learned_abstraction.py",
        "property_bounds.py",
        "property_refinement.py",
        "nested_design.py",
    ]:
        saved[name] = (base / "src/behavioral_equivalence" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(image=image, cpu=(2, 2), memory=(2048, 2048), timeout=300)
def smoke():
    import subprocess
    import numpy as np
    from behavioral_equivalence.decision_model import decision_chain
    from behavioral_equivalence.learned_abstraction import certified_latent_refinement
    from behavioral_equivalence.property_refinement import property_guided_latent_refinement
    from behavioral_equivalence.integer_bounds import verify_abstraction
    from behavioral_equivalence.property_bounds import verify_properties
    from behavioral_equivalence.nested_design import human_event_suite

    test = subprocess.run(
        ["python", "-m", "pytest", "-q", "/research/test_nested_design.py", "-W", "error"],
        capture_output=True,
        text=True,
        check=True,
    )
    parent = decision_chain(
        np.full((4, 2, 4), 0.25),
        np.array([[0.75, 0.25]] * 2 + [[0.25, 0.75]] * 2),
        np.full(4, 0.25),
    )
    blocks = ((0, 1), (2, 3))
    q, b, evidence = certified_latent_refinement(parent, blocks, 0.01)
    verify_abstraction(parent, q, evidence["certificate"])
    events = human_event_suite((1, 5))
    p, pb, pe = property_guided_latent_refinement(parent, blocks, events, 0.01)
    verify_properties(parent, p, pe["certificate"])
    assert b == pb == blocks and q.size == p.size == 6
    return dict(status="passed", source_digest=snapshot()[1], tests=test.stdout, latent_states=2)


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(8192, 8192),
    timeout=21600,
    max_containers=2,
    retries=0,
    volumes={"/artifacts": volume},
)
def construct(parent_root: str, parent_report_sha256: str):
    import gzip
    import hashlib
    import json
    import time
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.core import quotient
    from behavioral_equivalence.decision_model import decision_chain, decision_partition
    from behavioral_equivalence.learned_abstraction import (
        certified_latent_refinement,
        next_positive_probabilities,
    )
    from behavioral_equivalence.property_refinement import property_guided_latent_refinement
    from behavioral_equivalence.property_bounds import (
        event_predictions,
        certify_properties,
        verify_properties,
    )
    from behavioral_equivalence.integer_bounds import verify_abstraction
    from behavioral_equivalence.nested_design import (
        latent_partition,
        human_event_suite,
        nested_run_identity,
    )

    parts = PurePosixPath(parent_root)
    if (
        parts.is_absolute()
        or any(not part.replace("-", "").replace("_", "").isalnum() for part in parts.parts)
        or len(parent_report_sha256) != 64
    ):
        raise ValueError("Relative immutable parent root and SHA256 required")
    volume.reload()
    started = time.monotonic()
    saved, source_digest = snapshot()
    config = json.loads(saved["config.json"])
    source = Path("/artifacts") / parent_root
    raw = (source / "report.json").read_bytes()
    report = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != parent_report_sha256 or report["status"] != "complete":
        raise ValueError("Completed source-bound parent report required")
    parent_source = json.loads((source / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(parent_source, sort_keys=True).encode()).hexdigest()
        != report["source_digest"]
    ):
        raise ValueError("Parent estimator source changed")
    for name in ["core.py", "decision_model.py"]:
        if parent_source[name] != saved[name]:
            raise ValueError("Parent kernel or lifting semantics changed")
    model_name = f"k{config['parent_latent_states']}-parent.npz"
    if checksum(source / model_name) != report["output_sha256"][model_name]:
        raise ValueError("Parent point-model arrays changed")
    quant = source.parent
    if checksum(quant / "report.json") != report["quantization_report_sha256"]:
        raise ValueError("Parent training-only quantization changed")
    qr = json.loads((quant / "report.json").read_text())
    quant_name = f"k{config['parent_latent_states']}-model.npz"
    if checksum(quant / quant_name) != qr["output_sha256"][quant_name]:
        raise ValueError("Parent centroid artifact changed")
    with np.load(source / model_name, allow_pickle=False) as a:
        parent = decision_chain(a["transitions"], a["emissions"], a["initial"])
    if parent.digest() != next(
        m["chain_digest"]
        for m in report["models"]
        if m["latent_states"] == config["parent_latent_states"]
    ):
        raise ValueError("Parent chain differs from recorded digest")
    with np.load(quant / quant_name, allow_pickle=False) as a:
        centers = a["centers"]
    identity = nested_run_identity(source_digest, parent_root, parent_report_sha256)
    out = source / ("nested-" + identity[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    events = human_event_suite(tuple(config["decision_horizons"]))
    truth = event_predictions(parent, events)[:, ::3]
    seed = config["partition_seed"]
    records = []

    def completed(name):
        path = out / (name + ".json")
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        if item["run_identity"] != identity or any(
            checksum(out / n) != sha for n, sha in item["output_sha256"].items()
        ):
            raise ValueError("Cached nested artifact changed")
        return item

    def save(name, kind, abstract, blocks, evidence, seconds, **metadata):
        certificate = evidence.pop("certificate")
        cert_name = name + "-certificate.json.gz"
        packed = gzip.compress(
            json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode(), mtime=0
        )
        (out / cert_name).write_bytes(packed)
        model_name = name + "-model.npz"
        mapping = np.empty(config["parent_latent_states"], dtype=np.int64)
        for i, block in enumerate(blocks):
            mapping[list(block)] = i
        np.savez_compressed(
            out / model_name,
            transitions=abstract.transitions,
            initial=abstract.initial,
            latent_mapping=mapping,
            next_positive=next_positive_probabilities(abstract, blocks),
        )
        prediction = event_predictions(abstract, events)[:, 3 * mapping]
        radius = certificate.get("one_step_radius", certificate.get("epsilon"))
        if radius is None:
            radius_display = None
        elif "numerator_hex" in radius:
            radius_display = float(
                Fraction(int(radius["numerator_hex"], 16), int(radius["denominator_hex"], 16))
            )
        else:
            radius_display = None
        item = dict(
            name=name,
            kind=kind,
            run_identity=identity,
            parent_digest=parent.digest(),
            abstract_digest=abstract.digest(),
            latent_blocks=[list(b) for b in blocks],
            parent_latent_states=config["parent_latent_states"],
            reduced_latent_states=len(blocks),
            compression_ratio=config["parent_latent_states"] / len(blocks),
            one_step_radius_for_display=radius_display,
            floating_max_enumerated_property_difference=float(np.abs(truth - prediction).max()),
            evidence=evidence,
            wall_seconds=seconds,
            output_sha256={n: checksum(out / n) for n in [cert_name, model_name]},
            **metadata,
        )
        (out / (name + ".json")).write_text(json.dumps(item, indent=2))
        volume.commit()
        return item

    for count in config["initial_state_counts"]:
        initial = latent_partition(centers, count, seed)
        for method, budgets in [
            ("strict", config["strict_one_step_budgets"]),
            ("finite", config["finite_property_budgets"]),
        ]:
            for budget in budgets:
                name = f"{method}-k{count}-b{str(budget).replace('.', 'p')}"
                item = completed(name)
                if item is None:
                    begin = time.monotonic()
                    if method == "strict":
                        abstract, blocks, evidence = certified_latent_refinement(
                            parent, initial, budget, tuple(config["decision_horizons"])
                        )
                        verify_abstraction(parent, abstract, evidence["certificate"])
                    else:
                        abstract, blocks, evidence = property_guided_latent_refinement(
                            parent, initial, events, budget, config["certificate_bits"]
                        )
                        verify_properties(parent, abstract, evidence["certificate"])
                    item = save(
                        name,
                        method,
                        abstract,
                        blocks,
                        evidence,
                        time.monotonic() - begin,
                        initial_latent_states=count,
                        budget=budget,
                    )
                records.append(item)
                (out / "progress.json").write_text(
                    json.dumps(dict(completed=len(records), last=name))
                )
                volume.commit()
                print(
                    json.dumps(
                        dict(name=name, reduced_latent_states=item["reduced_latent_states"])
                    ),
                    flush=True,
                )
    capacities = sorted({r["reduced_latent_states"] for r in records})
    for count in capacities:
        for control in config["controls"]:
            name = f"control-{control}-k{count}"
            item = completed(name)
            if item is None:
                begin = time.monotonic()
                features = truth.T if control == "predictive" else centers
                blocks = latent_partition(features, count, seed, control == "random")
                abstract, _ = quotient(parent, decision_partition(blocks))
                cert = certify_properties(
                    parent,
                    abstract,
                    decision_partition(blocks),
                    events,
                    config["certificate_bits"],
                    tuple(range(0, parent.size, 3)),
                )
                verify_properties(parent, abstract, cert)
                item = save(
                    name,
                    "control",
                    abstract,
                    blocks,
                    {"certificate": cert},
                    time.monotonic() - begin,
                    control=control,
                )
            records.append(item)
    primary = [
        r
        for r in records
        if r["kind"] == config["primary"]["method"]
        and r.get("initial_latent_states") == config["primary"]["initial_latent_states"]
        and r.get("budget") == config["primary"]["budget"]
    ]
    if len(primary) != 1:
        raise AssertionError("Exactly one fixed primary candidate required")
    result = dict(
        status="complete",
        source_digest=source_digest,
        run_identity=identity,
        parent_root=parent_root,
        parent_report_sha256=parent_report_sha256,
        protocol=report["protocol"],
        architecture=report["architecture"],
        seed=report["seed"],
        records=records,
        primary=primary[0]["name"],
        primary_compression_met=primary[0]["reduced_latent_states"] <= 128,
        wall_seconds=time.monotonic() - started,
        root=str(out),
        final_test_predictions="not evaluated",
        provider_inference_calls=0,
        scope="Point-model compression only. Strict certificates cover label-path events under the coupling assumptions; finite certificates cover only the enumerated events. Neither proves human predictive noninferiority or human uncertainty coverage.",
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {k: v for k, v in result.items() if k != "records"}


@app.local_entrypoint()
def main(stage: str = "smoke", parent_root: str = "", parent_report_sha256: str = ""):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger
    from behavioral_equivalence.nested_design import nested_run_identity

    if stage == "smoke":
        print(json.dumps(smoke.remote()), flush=True)
        return
    if stage != "construct":
        raise ValueError("Stage must be smoke or construct")
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    b = config["resource_budget"]
    identity = nested_run_identity(digest, parent_root, parent_report_sha256)
    reserved = BudgetLedger(
        ROOT / ".campaign/nested-human-budget.sqlite", b["ceiling_usd"]
    ).reserve(
        identity,
        b["timeout_seconds"]
        * (
            b["worker_cpu"] * b["rate_cpu_second"]
            + b["worker_memory_gib"] * b["rate_memory_gib_second"]
        )
        + b["startup_buffer_usd"],
    )
    state = ROOT / ".campaign" / ("nested-" + identity[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = construct.spawn(parent_root, parent_report_sha256)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    function_id=construct.object_id,
                    source_digest=digest,
                    parent_root=parent_root,
                    parent_report_sha256=parent_report_sha256,
                    reservation=reserved,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
