"""Strict versioned artifact envelopes for cross-workstream integration."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = str


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    bytes: int = Field(ge=0)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["0.1"] = "0.1"
    run_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    phase: Literal["engineering_pilot", "exploratory", "confirmatory", "reproduction"]
    source_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    config_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    inputs: tuple[ArtifactRef, ...] = ()
    outputs: tuple[ArtifactRef, ...] = ()
    cloud_job_id: str | None = None
    state: Literal["planned", "submitted", "running", "complete", "failed", "blocked"]
    scientific_outcome: Literal["not_evaluated", "supported", "unsupported", "inconclusive"]
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_terminal_evidence(self) -> "RunManifest":
        if self.state == "complete" and not self.outputs:
            raise ValueError("Completed run requires evidence artifacts")
        if self.state in ("submitted", "running") and not self.cloud_job_id:
            raise ValueError("Submitted cloud work requires a live job reference")
        if self.state in ("failed", "blocked") and not self.failure_reason:
            raise ValueError("Failed or blocked work requires a recorded reason")
        return self


class ModelCertificate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["0.1"] = "0.1"
    concrete_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    abstract_digest: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    partition: tuple[tuple[int, ...], ...]
    target_label: str = Field(min_length=1)
    horizon: int = Field(ge=0)
    epsilon_upper: float = Field(ge=0, le=1, allow_inf_nan=False)
    initial_mismatch_upper: float = Field(ge=0, le=1, allow_inf_nan=False)
    uncertainty_semantics: Literal["none", "row_rectangular_per_step", "fixed_kernel"]
    nominal_lower: float = Field(ge=0, le=1, allow_inf_nan=False)
    nominal_upper: float = Field(ge=0, le=1, allow_inf_nan=False)
    numeric_status: Literal["exact_rational", "outward_rounded", "engineering_tolerance_only"]
    numeric_allowance: float = Field(ge=0, le=1, allow_inf_nan=False)
    proof_revision: str = Field(min_length=1)
    scope: Literal["specified_model_only"] = "specified_model_only"

    @model_validator(mode="after")
    def validate_interval(self) -> "ModelCertificate":
        if self.nominal_lower > self.nominal_upper:
            raise ValueError("Lower bound exceeds upper bound")
        if not self.partition or any(not b for b in self.partition):
            raise ValueError("Nonempty partition required")
        flat = [s for block in self.partition for s in block]
        if sorted(flat) != list(range(len(flat))):
            raise ValueError("Partition must cover contiguous concrete indices once")
        return self
