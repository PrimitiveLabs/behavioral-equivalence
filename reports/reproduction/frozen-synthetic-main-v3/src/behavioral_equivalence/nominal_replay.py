"""Separately executed replay of retained empirical-point ablation certificates."""

import gzip
import hashlib
import json
from pathlib import Path
from .nominal_chunk import evaluate_nominal_chunk


def replay_nominal_chunk(
    calibration: Path,
    ablation: Path,
    protocol: dict,
    calibration_report_sha256: str,
    ablation_report_sha256: str,
) -> dict:
    raw = (ablation / "report.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != ablation_report_sha256:
        raise ValueError("Frozen ablation report digest differs")
    report = json.loads(raw)
    packed = (ablation / "evidence.json.gz").read_bytes()
    if (
        report["status"] != "complete"
        or report["calibration_report_sha256"] != calibration_report_sha256
        or hashlib.sha256(packed).hexdigest() != report["evidence_sha256"]
    ):
        raise ValueError("Stored ablation evidence differs from its source-bound report")
    stored = json.loads(gzip.decompress(packed))
    recomputed = evaluate_nominal_chunk(calibration, protocol, calibration_report_sha256)
    if json.dumps(stored, sort_keys=True) != json.dumps(recomputed, sort_keys=True):
        raise ValueError("Saved empirical-point numerical evidence differs from arithmetic replay")
    if any(report[key] != stored[key] for key in stored if key != "certificates"):
        raise ValueError("Ablation report and saved evidence disagree")
    spec = recomputed["spec"]
    count = spec["last_repetition"] - spec["first_repetition"]
    if len(recomputed["certificates"]) != 2 * count:
        raise ValueError("Both support interpretations required for every retained count dataset")
    return dict(
        status="complete",
        original_chunk_digest=recomputed["original_chunk_digest"],
        calibration_report_sha256=calibration_report_sha256,
        ablation_report_sha256=ablation_report_sha256,
        evidence_sha256=report["evidence_sha256"],
        verified_datasets=count,
        verified_point_certificates=2 * count,
        scope="Retained-count and canonical numerical-evidence replay using the existing arithmetic implementation; not a second independent theorem proof or new dataset",
    )
