"""Bind complete existing data and forecast matrices before supplementary scoring."""

from pathlib import Path
import hashlib
import json


def build_config(root):
    root = Path(root)
    design = json.loads((root / "configs/human-trajectory-supplement-v1.json").read_text())
    parent = json.loads((root / "configs/human-final-horizon-v1.json").read_text())
    seq_path = root / "reports/preflight/trajectory-sequences-f0ea31ab14ee9c5e.json"
    raw = seq_path.read_bytes()
    sequence = json.loads(raw)
    if (
        sequence["status"] != "complete"
        or len(sequence["buckets"]) != 32
        or {r["bucket"] for r in sequence["buckets"]} != set(range(32))
        or sequence["source_digest"]
        != "f0ea31ab14ee9c5e170bbe5c49fbe7b9ee1c22bd7c8d95b3cc53008b0b27bb89"
    ):
        raise ValueError("Complete source-bound sequence reconstruction required")
    cfg = dict(
        design,
        status="execution_frozen",
        sequence_root=sequence["artifact_root"],
        sequence_report_sha256=hashlib.sha256(raw).hexdigest(),
        probes=[],
    )
    expected = {(p, s) for p in design["protocols"] for s in design["seeds"]}
    if (
        len(parent["probes"]) != len(expected)
        or {(p["protocol"], p["seed"]) for p in parent["probes"]} != expected
    ):
        raise ValueError("Complete fixed parent matrix required")
    for probe in parent["probes"]:
        path = (
            root
            / "reports/final-human"
            / f"direct_horizon-{probe['protocol']}-{probe['seed']}-24.json"
        )
        raw = path.read_bytes()
        report = json.loads(raw)
        if (
            report["status"] != "complete"
            or report["protocol"] != probe["protocol"]
            or report["seed"] != probe["seed"]
            or report["source_digest"]
            != "d4984c93acbe0b14f2a81e9477738e49dec01bd53115a14ae8f0534ac66ad3be"
            or any(
                f"bucket-{b:02d}-test-forecasts.npz" not in report["output_sha256"]
                for b in range(32)
            )
        ):
            raise ValueError("Complete original source-bound horizon forecasts required")
        cfg["probes"].append(
            dict(
                protocol=probe["protocol"],
                seed=probe["seed"],
                nested_root=probe["nested_root"],
                nested_report_sha256=probe["nested_report_sha256"],
                forecast_report_sha256=hashlib.sha256(raw).hexdigest(),
                forecast_report={"root": report["root"], "output_sha256": report["output_sha256"]},
            )
        )
    return cfg


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    target = root / "configs/human-trajectory-final-v1.json"
    raw = json.dumps(build_config(root), indent=2) + "\n"
    if target.exists() and target.read_text() != raw:
        raise ValueError("Existing trajectory freeze cannot change")
    target.write_text(raw)
    print(hashlib.sha256(raw.encode()).hexdigest())
