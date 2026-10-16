"""Typed, content-verified inventory across separately frozen experiment formats."""

import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .contracts import ArtifactRef


class EvidenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: str = Field(min_length=1)
    stage: Literal[
        "synthetic_summary",
        "certificate_audit",
        "uncertainty_summary",
        "intervention_summary",
        "human_training_checkpoint",
        "human_quantization",
        "human_calibrated_parent",
        "human_nested_abstraction",
        "human_data_audit",
        "human_multistep_inputs",
        "human_uncertainty",
        "release_test_reproduction",
    ]
    domain: Literal["specified_model", "human_data", "release"]
    artifact: ArtifactRef
    scope: str = Field(min_length=1)
    parents: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_stage_domain(self) -> "EvidenceEntry":
        expected = (
            "human_data"
            if self.stage.startswith("human_")
            else "release"
            if self.stage == "release_test_reproduction"
            else "specified_model"
        )
        if self.domain != expected:
            raise ValueError("Stage and scientific domain disagree")
        path = PurePosixPath(self.artifact.uri)
        if path.is_absolute() or ".." in path.parts or "\\" in self.artifact.uri:
            raise ValueError("Registry artifacts require repository-relative paths")
        return self


class EvidenceRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["0.1"] = "0.1"
    verification_scope: Literal["artifact_identity_only"] = "artifact_identity_only"
    entries: tuple[EvidenceEntry, ...]

    @model_validator(mode="after")
    def validate_lineage(self) -> "EvidenceRegistry":
        entries = {entry.evidence_id: entry for entry in self.entries}
        if len(entries) != len(self.entries):
            raise ValueError("Duplicate evidence identities")
        finished: set[str] = set()

        def visit(identity: str, ancestors: frozenset[str]) -> None:
            if identity not in entries:
                raise ValueError("Missing parent evidence")
            if identity in ancestors:
                raise ValueError("Cyclic evidence lineage")
            if identity in finished:
                return
            for parent in entries[identity].parents:
                visit(parent, ancestors | {identity})
            finished.add(identity)

        for identity in entries:
            visit(identity, frozenset())
        return self


def verify_registry(registry: EvidenceRegistry, root: Path) -> dict:
    """Verify identity only. Never infer experiment stage from report fields."""
    root = root.resolve()
    for entry in registry.entries:
        path = (root / entry.artifact.uri).resolve()
        if root not in path.parents:
            raise ValueError("Artifact resolves outside repository")
        raw = path.read_bytes()
        if len(raw) != entry.artifact.bytes:
            raise ValueError(f"Artifact size differs: {entry.evidence_id}")
        if hashlib.sha256(raw).hexdigest() != entry.artifact.sha256:
            raise ValueError(f"Artifact digest differs: {entry.evidence_id}")
    return {
        "status": "verified",
        "artifacts": len(registry.entries),
        "scope": registry.verification_scope,
        "scientific_completion": "not_assessed",
    }
