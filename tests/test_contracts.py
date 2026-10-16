import pytest
from pydantic import ValidationError
from behavioral_equivalence.contracts import ArtifactRef, RunManifest


def test_run_cannot_complete_without_evidence_or_run_without_job():
    base = dict(
        run_id="r",
        task_id="E1",
        phase="engineering_pilot",
        source_digest="a" * 64,
        config_digest="b" * 64,
        scientific_outcome="not_evaluated",
    )
    for state in ("running", "complete", "failed"):
        with pytest.raises(ValidationError):
            RunManifest(**base, state=state)
    artifact = ArtifactRef(
        sha256="c" * 64, uri="r2://research/digest", media_type="application/json", bytes=2
    )
    manifest = RunManifest(**base, state="complete", outputs=(artifact,))
    assert manifest.scientific_outcome == "not_evaluated"
    with pytest.raises(ValidationError):
        ArtifactRef(sha256="bad", uri="x", media_type="x", bytes=1)
