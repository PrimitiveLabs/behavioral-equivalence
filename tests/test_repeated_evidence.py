"""Integrity tests for retained evidence, including self-consistent hash tampering."""

from fractions import Fraction
import gzip
import hashlib
import json
import numpy as np
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.confidence import certified_hoeffding
from behavioral_equivalence.uncertainty import clopper_pearson_intervals
from behavioral_equivalence.uncertainty_repetition import (
    row_sample_schedule,
    terminal_label_support,
)
from behavioral_equivalence.robust_benchmark import assess_intervals
from behavioral_equivalence.repeated_analysis import METHODS, SUPPORTS
from behavioral_equivalence.repeated_evidence import audit_repetition_chunk


def write_artifacts(path, report, evidence):
    packed = gzip.compress(json.dumps(evidence, sort_keys=True).encode(), mtime=0)
    (path / "repetition-000.json.gz").write_bytes(packed)
    report["artifact_sha256"] = {"repetition-000.json.gz": hashlib.sha256(packed).hexdigest()}
    (path / "report.json").write_text(json.dumps(report))


@pytest.fixture
def retained(tmp_path):
    protocol = dict(
        source_digest="s",
        protocol_digest="p",
        families=["trust"],
        sizes=[16],
        seeds=[7],
        regimes=["uniform_active"],
        repetitions=1,
        samples_per_active=1000,
        perturbation=0.01,
        horizons=[5],
        bits=32,
    )
    spec = dict(
        family="trust",
        size=16,
        seed=7,
        regime="uniform_active",
        first_repetition=0,
        last_repetition=1,
    )
    world = generate("trust", 16, 7, 0.01)
    schedule = row_sample_schedule(world.chain, "uniform_active")
    rng = np.random.default_rng([20261015, 0, 16, 7, 0, 0])
    counts = np.array(
        [rng.multinomial(int(m), row) for m, row in zip(schedule, world.chain.transitions)]
    )
    evidence = dict(counts=counts.tolist(), row_sample_schedule=schedule.tolist(), methods=[])
    report = dict(
        status="complete",
        spec=spec,
        source_digest="s",
        protocol_digest="p",
        chunk_digest=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        world_digest=world.chain.digest(),
        row_sample_schedule=schedule.tolist(),
        records=[],
    )
    mapping = np.array(
        [next(i for i, b in enumerate(world.partition) if s in b) for s in range(16)]
    )
    active = np.array(world.chain.labels) == "active"
    for method in METHODS:
        if method == METHODS[0]:
            uncertain, confidence = certified_hoeffding(counts, world.chain.labels, Fraction(1, 20))
        else:
            uncertain = clopper_pearson_intervals(counts, world.chain.labels, 0.05)
            confidence = {
                "statistical_numeric_scope": "floating beta quantiles; no outward statistical-tail certificate"
            }
        for support in SUPPORTS:
            allowed = (
                uncertain
                if support == SUPPORTS[0]
                else terminal_label_support(uncertain, world.chain)
            )
            metrics, certificate = assess_intervals(world.chain, allowed, world.partition, [5], 32)
            values = certificate["robust_values"]["complete"]["values"]["5"]
            width = (
                np.array(values["max_upper"], dtype=float)
                - np.array(values["min_lower"], dtype=float)
            )[mapping] / (1 << 32)
            report["records"].append(
                dict(
                    repetition=0,
                    method=method,
                    support=support,
                    **metrics,
                    completion_widths=[
                        dict(
                            horizon=5,
                            mean_active_width=float(width[active].mean()),
                            initial_weighted_width=float(np.sum(world.chain.initial * width)),
                        )
                    ],
                )
            )
            evidence["methods"].append(
                dict(method=method, support=support, confidence=confidence, certificate=certificate)
            )
    write_artifacts(tmp_path, report, evidence)
    return tmp_path, protocol, report, evidence


def test_replay_checks_all_four_paired_evaluations(retained):
    path, protocol, _, _ = retained
    audit = audit_repetition_chunk(path, protocol)
    assert audit["verified_datasets"] == 1
    assert audit["verified_method_support_evaluations"] == 4


@pytest.mark.parametrize("damage", ["counts", "metrics", "certificate", "schedule", "missing"])
def test_tamper_is_rejected_even_with_fresh_file_hash(retained, damage):
    path, protocol, report, evidence = retained
    if damage == "counts":
        evidence["counts"][0][0] += 1
    elif damage == "metrics":
        report["records"][0]["kernel_membership"] = not report["records"][0]["kernel_membership"]
    elif damage == "certificate":
        evidence["methods"][0]["certificate"]["robust_values"]["complete"]["bits"] = 1
    elif damage == "schedule":
        report["row_sample_schedule"][0] += 1
    else:
        evidence["methods"].pop()
    write_artifacts(path, report, evidence)
    with pytest.raises(ValueError):
        audit_repetition_chunk(path, protocol)
