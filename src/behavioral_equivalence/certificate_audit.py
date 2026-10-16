"""Exact arithmetic audit of a completed, frozen synthetic world artifact."""

from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from .core import Chain, quotient
from .integer_bounds import certify_abstraction, verify_abstraction
from .prism import write_model


def audit_world(source: Path, destination: Path, horizons: list[int], bits: int = 48) -> dict:
    report_bytes = (source / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if report["status"] != "complete":
        raise ValueError("Only completed synthetic worlds can be audited")
    with np.load(source / "world.npz", allow_pickle=False) as saved:
        concrete = Chain(saved["p"], tuple(saved["labels"].tolist()), saved["initial"])
    if concrete.digest() != report["world_digest"]:
        raise ValueError("World kernel differs from recorded digest")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, method in enumerate(report["methods"]):
        blocks = tuple(tuple(b) for b in method["partition"])
        abstract, _ = quotient(concrete, blocks)
        # Bind the certified kernel to the actual model supplied to PRISM.
        # PRISM's initial point is state zero; certificates cover all related pairs.
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "model.prism"
            write_model(abstract, generated)
            if generated.read_bytes() != (source / f"model-{index}.prism").read_bytes():
                raise ValueError("Reconstructed quotient differs from PRISM input")
        cert = certify_abstraction(concrete, abstract, blocks, horizons, bits)
        verify_abstraction(concrete, abstract, cert)
        packed = gzip.compress(json.dumps(cert, sort_keys=True).encode(), mtime=0)
        name = f"certificate-{index}.json.gz"
        (destination / name).write_bytes(packed)
        value = cert["epsilon"]
        epsilon = Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16))
        budget = method["parameters"].get("epsilon_budget")
        excess = (
            max(Fraction(), epsilon - Fraction.from_float(budget)) if budget is not None else None
        )
        records.append(
            {
                "index": index,
                "method": method["method"],
                "states": abstract.size,
                "epsilon_approximation_for_display": float(epsilon),
                "epsilon_is_exactly_zero": epsilon == 0,
                "exact_budget_excess_for_display": float(excess) if excess is not None else None,
                "budget_strictly_satisfied": excess == 0 if excess is not None else None,
                "certificate": name,
                "certificate_sha256": hashlib.sha256(packed).hexdigest(),
                "verified": True,
            }
        )
    result = {
        "status": "complete",
        "spec": report["spec"],
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "source_protocol_digest": report["protocol_digest"],
        "source_digest": report["source_digest"],
        "world_digest": concrete.digest(),
        "bits": bits,
        "horizons": horizons,
        "methods": records,
        "verified_certificates": len(records),
        "numerical_semantics": "exact normalized binary-rational kernels and directed integer probability enclosures",
        "certificate_scope": "all related deterministic initial state pairs; bounded label events; no statistical human claim",
        "validation_scope": "certificate recomputation and exact binding to PRISM model text, not an independent proof-assistant check",
    }
    (destination / "report.json").write_text(json.dumps(result, indent=2))
    return result
