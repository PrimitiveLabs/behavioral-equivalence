import importlib.util
from pathlib import Path
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.core import reachability

spec = importlib.util.spec_from_file_location(
    "scientific_reproduction", Path(__file__).parents[1] / "tools/scientific_reproduction.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def oracle(path, target, horizon, log, **kwargs):
    world = generate("memory", 32, 20261024, 0.01 if "perturbed" in path.name else 0)
    log.write_text("Deterministic fixture backend; not actual PRISM")
    return float(reachability(world.chain, target, horizon)[0])


def test_driver_requires_complete_families_and_retains_certificate_and_backend_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "FAMILIES", ("memory",))
    monkeypatch.setattr(module, "check", oracle)
    r = module.run(tmp_path / "out", "fixture")
    assert r["worlds"] == 2 and r["external_queries"] == 4 and r["certificate_replays"] == 2
    assert r["maximum_external_difference"] == 0
    assert any(n.endswith("-certificate.json") for n in r["artifacts"])
    assert (tmp_path / "out/report.json").exists()
    with pytest.raises(FileExistsError):
        module.run(tmp_path / "out", "fixture")


def test_driver_refuses_disagreeing_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "FAMILIES", ("memory",))
    monkeypatch.setattr(module, "check", lambda *args, **kwargs: 0.9999)
    with pytest.raises(ValueError, match="PRISM prediction differs"):
        module.run(tmp_path / "bad", "fixture")
