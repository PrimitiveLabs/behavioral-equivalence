import hashlib
import pytest
from pydantic import ValidationError
from behavioral_equivalence.evidence_registry import (
    EvidenceEntry,
    EvidenceRegistry,
    verify_registry,
)


def entry(path="report.json", **changes):
    values = dict(
        evidence_id="parent",
        stage="human_calibrated_parent",
        domain="human_data",
        artifact=dict(
            uri=path,
            sha256=hashlib.sha256(b"{}").hexdigest(),
            bytes=2,
            media_type="application/json",
        ),
        scope="Estimated model only; no human fidelity conclusion",
    )
    values.update(changes)
    return EvidenceEntry(**values)


def test_explicit_stages_and_content_identity(tmp_path):
    (tmp_path / "report.json").write_bytes(b"{}")
    parent = entry()
    nested = entry(evidence_id="nested", stage="human_nested_abstraction", parents=("parent",))
    registry = EvidenceRegistry(entries=(parent, nested))
    assert verify_registry(registry, tmp_path) == dict(
        status="verified",
        artifacts=2,
        scope="artifact_identity_only",
        scientific_completion="not_assessed",
    )
    (tmp_path / "report.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="digest"):
        verify_registry(registry, tmp_path)
    (tmp_path / "report.json").write_bytes(b"longer")
    with pytest.raises(ValueError, match="size"):
        verify_registry(registry, tmp_path)


def test_domain_paths_and_lineage_rejected(tmp_path):
    with pytest.raises(ValidationError, match="domain"):
        entry(domain="specified_model")
    for path in ("/outside", "../outside", "a/../../outside", "a\\outside"):
        with pytest.raises(ValidationError, match="relative"):
            entry(path)
    with pytest.raises(ValidationError, match="Duplicate"):
        EvidenceRegistry(entries=(entry(), entry()))
    with pytest.raises(ValidationError, match="Missing"):
        EvidenceRegistry(entries=(entry(parents=("unknown",)),))
    with pytest.raises(ValidationError, match="Cyclic"):
        EvidenceRegistry(entries=(entry(parents=("parent",)),))
    outside = tmp_path.parent / (tmp_path.name + "-outside.json")
    outside.write_bytes(b"{}")
    (tmp_path / "report.json").symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        verify_registry(EvidenceRegistry(entries=(entry(),)), tmp_path)


def test_model_and_release_domains(tmp_path):
    for stage, domain in (
        ("synthetic_summary", "specified_model"),
        ("human_uncertainty", "human_data"),
        ("release_test_reproduction", "release"),
    ):
        assert entry(stage=stage, domain=domain).domain == domain
    registry = EvidenceRegistry(entries=(entry(parents=()),))
    with pytest.raises(FileNotFoundError):
        verify_registry(registry, tmp_path)
