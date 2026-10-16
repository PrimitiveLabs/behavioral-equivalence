"""The shared exported contract files must match their executable models."""

import json
from pathlib import Path
import pytest
from behavioral_equivalence.contracts import ArtifactRef, RunManifest, ModelCertificate
from behavioral_equivalence.evidence_registry import EvidenceRegistry
from behavioral_equivalence.human_comparison import HumanPrimaryResult


@pytest.mark.parametrize(
    "model", [ArtifactRef, RunManifest, ModelCertificate, EvidenceRegistry, HumanPrimaryResult]
)
def test_exported_schema_matches_executable_contract(model):
    path = Path(__file__).parents[1] / "spec/schemas" / (model.__name__ + ".schema.json")
    assert json.loads(path.read_text()) == model.model_json_schema()
