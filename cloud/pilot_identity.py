"""Freeze pilot-user identity fingerprints without using outcomes or item IDs."""

import modal

app = modal.App("behavioral-equivalence-pilot-identity-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("polars==1.31.0", "numpy==2.2.6")
    .env({"OPENBLAS_NUM_THREADS": "1"})
)


@app.function(
    image=image, cpu=2, memory=4096, timeout=600, max_containers=1, volumes={"/artifacts": volume}
)
def fingerprint() -> dict:
    from pathlib import Path
    import hashlib
    import json
    import time
    import numpy as np
    import polars as pl

    start = time.monotonic()
    root = Path("/artifacts/datasets/kuairand-1k/prepared-v4")
    # Raw converted logs retain duplicates. Prepared/deduplicated sequences
    # cannot establish exact identity equivalence with a raw final release.
    paths = sorted(root.glob("log_*.parquet"))
    if len(paths) != 3:
        raise ValueError("Expected all three unfiltered pilot source logs")
    records = (
        pl.scan_parquet(paths)
        .select(["user_id", "time_ms", "tab"])
        .collect()
        .sort(["user_id", "time_ms", "tab"])
    )
    fingerprints = []
    for group in records.partition_by("user_id", maintain_order=True):
        payload = (
            np.column_stack([group["time_ms"].to_numpy(), group["tab"].to_numpy()])
            .astype("<i8")
            .tobytes(order="C")
        )
        fingerprints.append(
            {
                "pilot_user_id": int(group["user_id"][0]),
                "rows": group.height,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if len({r["sha256"] for r in fingerprints}) != len(fingerprints):
        raise ValueError("Pilot identity fingerprints are not unique")
    artifact = {
        "schema_version": "0.1",
        "dataset": "KuaiRand-1K",
        "records": fingerprints,
        "algorithm": "SHA256 of sorted raw (time_ms,tab) pairs, each signed little-endian int64",
        "not_inputs": ["outcome labels", "video identifiers", "release-specific user identifiers"],
        "usage": "pilot exclusion only; never a predictor or outcome stratification feature",
    }
    (root / "pilot-user-fingerprints.json").write_text(json.dumps(artifact, indent=2))
    report = {
        "status": "complete",
        "users": len(fingerprints),
        "raw_rows": records.height,
        "unique_fingerprints": len({r["sha256"] for r in fingerprints}),
        "artifact_sha256": hashlib.sha256(json.dumps(artifact, indent=2).encode()).hexdigest(),
        "wall_seconds": time.monotonic() - start,
        "final_release_mapping": "pending 27K source conversion",
    }
    (root / "pilot-identity-report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report
