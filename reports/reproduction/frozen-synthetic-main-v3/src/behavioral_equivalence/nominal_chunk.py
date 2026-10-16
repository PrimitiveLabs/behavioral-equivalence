"""Paired point-kernel ablation over immutable retained calibration counts."""

import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .synthetic import generate
from .nominal_ablation import empirical_point_intervals, verify_empirical_point
from .robust_benchmark import assess_intervals


def evaluate_nominal_chunk(path: Path, protocol: dict, report_sha256: str) -> dict:
    raw = (path / "report.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != report_sha256:
        raise ValueError("Frozen calibration report digest differs")
    report = json.loads(raw)
    spec = report["spec"]
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (
        report["status"] != "complete"
        or report["chunk_digest"] != identity
        or report["source_digest"] != protocol["source_digest"]
        or report["protocol_digest"] != protocol["protocol_digest"]
        or spec["family"] not in protocol["families"]
        or spec["size"] not in protocol["sizes"]
        or spec["seed"] not in protocol["seeds"]
        or spec["regime"] not in protocol["regimes"]
        or not 0 <= spec["first_repetition"] < spec["last_repetition"] <= protocol["repetitions"]
    ):
        raise ValueError("Completed source-bound calibration chunk required")
    expected_files = {
        f"repetition-{repetition:03d}.json.gz"
        for repetition in range(spec["first_repetition"], spec["last_repetition"])
    }
    if set(report["artifact_sha256"]) != expected_files:
        raise ValueError("Missing or extra retained count files")
    world = generate(spec["family"], spec["size"], spec["seed"], protocol["perturbation"])
    if world.chain.digest() != report["world_digest"]:
        raise ValueError("Generating world differs")
    mapping = np.array(
        [
            next(i for i, block in enumerate(world.partition) if s in block)
            for s in range(world.chain.size)
        ]
    )
    active = np.array(world.chain.labels) == "active"
    rows = []
    certificates = []
    for repetition in range(spec["first_repetition"], spec["last_repetition"]):
        name = f"repetition-{repetition:03d}.json.gz"
        packed = (path / name).read_bytes()
        if hashlib.sha256(packed).hexdigest() != report["artifact_sha256"][name]:
            raise ValueError("Retained count file digest differs")
        evidence = json.loads(gzip.decompress(packed))
        counts = np.asarray(evidence["counts"])
        if counts.sum(axis=1).tolist() != report["row_sample_schedule"]:
            raise ValueError("Count totals differ from frozen row schedule")
        for known, support in (
            (False, "unknown_terminal_rows"),
            (True, "known_closed_terminal_labels"),
        ):
            intervals, numeric_certificate = empirical_point_intervals(
                counts, world.chain.labels, known
            )
            verify_empirical_point(counts, world.chain.labels, numeric_certificate)
            metrics, certificate = assess_intervals(
                world.chain, intervals, world.partition, protocol["horizons"], protocol["bits"]
            )
            widths = []
            for horizon in protocol["horizons"]:
                values = certificate["robust_values"]["complete"]["values"][str(horizon)]
                width = (
                    np.array(values["max_upper"], dtype=float)
                    - np.array(values["min_lower"], dtype=float)
                )[mapping] / (1 << protocol["bits"])
                widths.append(
                    dict(
                        horizon=horizon,
                        mean_active_width=float(width[active].mean()),
                        initial_weighted_width=float(world.chain.initial @ width),
                    )
                )
            rows.append(
                dict(
                    repetition=repetition,
                    support=support,
                    method="empirical_point",
                    **metrics,
                    completion_widths=widths,
                )
            )
            certificates.append(
                dict(
                    repetition=repetition,
                    support=support,
                    point_numeric=numeric_certificate,
                    envelope=certificate,
                )
            )
    return dict(
        status="complete",
        calibration_report_sha256=report_sha256,
        original_chunk_digest=identity,
        spec=spec,
        records=rows,
        certificates=certificates,
        statistical_coverage_claim="none",
        scope="Secondary paired ablation on retained counts; generating probabilities are known synthetic quantities, not human confidence coverage.",
    )
