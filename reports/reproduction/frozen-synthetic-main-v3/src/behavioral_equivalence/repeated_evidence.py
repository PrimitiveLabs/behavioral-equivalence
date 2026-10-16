"""Recompute retained repeated-calibration evidence from fixed synthetic inputs."""

from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .synthetic import generate
from .confidence import certified_hoeffding, verify_hoeffding
from .uncertainty import clopper_pearson_intervals
from .uncertainty_repetition import row_sample_schedule, terminal_label_support
from .robust_benchmark import assess_intervals
from .repeated_analysis import METHODS, SUPPORTS


def audit_repetition_chunk(path: Path, protocol: dict) -> dict:
    """Replay draws and integer calculations, then compare saved certificates.

    This is a separately executed evidence replay using the same certified
    arithmetic implementation, not a second independent proof or algorithm.
    """
    started = time.monotonic()
    raw = (path / "report.json").read_bytes()
    report = json.loads(raw)
    spec = report["spec"]
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (
        report["status"] != "complete"
        or report["chunk_digest"] != identity
        or report["source_digest"] != protocol["source_digest"]
        or report["protocol_digest"] != protocol["protocol_digest"]
    ):
        raise ValueError("Completed source-bound chunk required")
    if (
        spec["family"] not in protocol["families"]
        or spec["size"] not in protocol["sizes"]
        or spec["seed"] not in protocol["seeds"]
        or spec["regime"] not in protocol["regimes"]
        or not 0 <= spec["first_repetition"] < spec["last_repetition"] <= protocol["repetitions"]
    ):
        raise ValueError("Chunk lies outside frozen design")
    world = generate(spec["family"], spec["size"], spec["seed"], protocol["perturbation"])
    schedule = row_sample_schedule(world.chain, spec["regime"], protocol["samples_per_active"])
    if (
        world.chain.digest() != report["world_digest"]
        or schedule.tolist() != report["row_sample_schedule"]
    ):
        raise ValueError("Ground-truth world or row schedule differs")
    expected_files = {
        f"repetition-{r:03d}.json.gz"
        for r in range(spec["first_repetition"], spec["last_repetition"])
    }
    if set(report["artifact_sha256"]) != expected_files:
        raise ValueError("Missing or extraneous repetition artifacts")
    saved_records = {(r["repetition"], r["method"], r["support"]): r for r in report["records"]}
    if (
        len(saved_records) != len(report["records"])
        or len(saved_records) != len(expected_files) * 4
    ):
        raise ValueError("Duplicate or missing method/support records")
    mapping = np.array(
        [next(i for i, b in enumerate(world.partition) if s in b) for s in range(world.chain.size)]
    )
    active = np.array(world.chain.labels) == "active"
    verified = 0
    for repetition in range(spec["first_repetition"], spec["last_repetition"]):
        name = f"repetition-{repetition:03d}.json.gz"
        packed = (path / name).read_bytes()
        if hashlib.sha256(packed).hexdigest() != report["artifact_sha256"][name]:
            raise ValueError("Retained calibration artifact changed")
        evidence = json.loads(gzip.decompress(packed))
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
        if (
            counts.tolist() != evidence["counts"]
            or schedule.tolist() != evidence["row_sample_schedule"]
        ):
            raise ValueError("Calibration draws do not reproduce")
        certificates = {(r["method"], r["support"]): r for r in evidence["methods"]}
        if (
            set(certificates) != {(m, s) for m in METHODS for s in SUPPORTS}
            or len(evidence["methods"]) != 4
        ):
            raise ValueError("Missing or duplicate retained certificates")
        for method in METHODS:
            if method == METHODS[0]:
                uncertain, confidence = certified_hoeffding(
                    counts, world.chain.labels, Fraction(1, 20)
                )
                verify_hoeffding(counts, world.chain.labels, confidence)
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
                metrics, certificate = assess_intervals(
                    world.chain, allowed, world.partition, protocol["horizons"], protocol["bits"]
                )
                retained = certificates[(method, support)]
                if retained["confidence"] != confidence or retained["certificate"] != certificate:
                    raise ValueError("Confidence or integer model certificate does not reproduce")
                widths = []
                for h in protocol["horizons"]:
                    values = certificate["robust_values"]["complete"]["values"][str(h)]
                    width = (
                        np.array(values["max_upper"], dtype=float)
                        - np.array(values["min_lower"], dtype=float)
                    )[mapping] / (1 << protocol["bits"])
                    widths.append(
                        dict(
                            horizon=h,
                            mean_active_width=float(width[active].mean()),
                            initial_weighted_width=float(np.sum(world.chain.initial * width)),
                        )
                    )
                reconstructed = dict(
                    repetition=repetition,
                    method=method,
                    support=support,
                    **metrics,
                    completion_widths=widths,
                )
                if saved_records[(repetition, method, support)] != reconstructed:
                    raise ValueError("Reported coverage or widths differ from certificates")
                verified += 1
    return dict(
        status="complete",
        chunk_digest=identity,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        verified_datasets=len(expected_files),
        verified_method_support_evaluations=verified,
        wall_seconds=time.monotonic() - started,
        scope="Source-bound replay with the original certified arithmetic, not an independent mathematical implementation",
    )
