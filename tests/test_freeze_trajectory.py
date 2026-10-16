import importlib.util
from pathlib import Path
import json
import pytest

spec = importlib.util.spec_from_file_location(
    "freeze_trajectory", Path(__file__).parents[1] / "tools/freeze_trajectory.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    (root / "configs").mkdir()
    (root / "reports/preflight").mkdir(parents=True)
    (root / "reports/final-human").mkdir()

    def write(path, obj):
        (root / path).write_text(json.dumps(obj))

    write(
        "configs/human-trajectory-supplement-v1.json",
        {"protocols": ["known_users", "unseen_users"], "seeds": list(range(20261015, 20261020))},
    )
    probes = []
    for p in ["known_users", "unseen_users"]:
        for s in range(20261015, 20261020):
            probes.append(
                dict(protocol=p, seed=s, nested_root="private/model", nested_report_sha256="x")
            )
            write(
                f"reports/final-human/direct_horizon-{p}-{s}-24.json",
                dict(
                    status="complete",
                    protocol=p,
                    seed=s,
                    source_digest="d4984c93acbe0b14f2a81e9477738e49dec01bd53115a14ae8f0534ac66ad3be",
                    root="/artifacts/private",
                    output_sha256={f"bucket-{b:02d}-test-forecasts.npz": "sha" for b in range(32)},
                ),
            )
    write("configs/human-final-horizon-v1.json", {"probes": probes})
    write(
        "reports/preflight/trajectory-sequences-f0ea31ab14ee9c5e.json",
        dict(
            status="complete",
            source_digest="f0ea31ab14ee9c5e170bbe5c49fbe7b9ee1c22bd7c8d95b3cc53008b0b27bb89",
            artifact_root="/artifacts/sequences",
            buckets=[{"bucket": b} for b in range(32)],
        ),
    )


def test_binds_complete_ten_fit_matrix_without_predictions_in_config(tmp_path):
    fixture(tmp_path)
    cfg = module.build_config(tmp_path)
    assert cfg["status"] == "execution_frozen" and len(cfg["probes"]) == 10
    assert len(cfg["sequence_report_sha256"]) == 64
    assert set(cfg["probes"][0]["forecast_report"]) == {"root", "output_sha256"}


@pytest.mark.parametrize(
    "change", ["missing_fit", "wrong_seed", "partial_files", "partial_sequence", "duplicate_probe"]
)
def test_refuses_incomplete_or_misaligned_inputs(tmp_path, change):
    fixture(tmp_path)
    path = tmp_path / "reports/final-human/direct_horizon-known_users-20261015-24.json"
    if change == "missing_fit":
        path.unlink()
    elif change in ["wrong_seed", "partial_files"]:
        r = json.loads(path.read_text())
        if change == "wrong_seed":
            r["seed"] = 20261019
        else:
            r["output_sha256"].pop("bucket-00-test-forecasts.npz")
        path.write_text(json.dumps(r))
    elif change == "partial_sequence":
        path = tmp_path / "reports/preflight/trajectory-sequences-f0ea31ab14ee9c5e.json"
        r = json.loads(path.read_text())
        r["buckets"].pop()
        path.write_text(json.dumps(r))
    else:
        path = tmp_path / "configs/human-final-horizon-v1.json"
        r = json.loads(path.read_text())
        r["probes"][-1] = r["probes"][0]
        path.write_text(json.dumps(r))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.build_config(tmp_path)
