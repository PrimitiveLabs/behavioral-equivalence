import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "supplementary_freeze", Path(__file__).parents[1] / "tools/freeze_supplementary.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(root):
    endpoints = [
        dict(name=name, classes=4 if name == "watch_category" else 2)
        for name in [
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "watch_category",
        ]
    ]
    cfg = dict(
        protocols=["known_users", "unseen_users"],
        seeds=list(range(20261015, 20261020)),
        architectures=["context", "gru", "transformer"],
        endpoints=endpoints,
        parent_probes=[],
    )
    for name in [
        "case_root",
        "case_report_sha256",
        "context_root",
        "context_report_sha256",
        "target_attachment_root",
        "target_attachment_report_canonical_digest",
        "fitting_cohort_root",
        "fitting_cohort_canonical_digest",
    ]:
        cfg[name] = "0" * 64
    for protocol in cfg["protocols"]:
        for seed in cfg["seeds"]:
            cfg["parent_probes"].append(
                dict(
                    protocol=protocol,
                    seed=seed,
                    training_root="frozen",
                    fit_report_sha256={a: "0" * 64 for a in cfg["architectures"]},
                )
            )
            for architecture in cfg["architectures"]:
                r = dict(
                    status="complete",
                    source_digest="5d9fae59fec2dbbfeafa086f78d98e16347ff7c3a020fb22cc43c89b5f5a3e5a",
                    protocol=protocol,
                    architecture=architecture,
                    seed=seed,
                    records=[dict(endpoint=e["name"], status="complete") for e in endpoints],
                    output_sha256={"model.npz": "0" * 64},
                )
                path = (
                    root / "reports/supplementary-heads" / f"{protocol}-{architecture}-{seed}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(r))
            r = dict(
                status="complete",
                source_digest="68edad5c7113cb6d09663a86f3c465e0293d9625475e7c8f82999e39646dc531",
                protocol=protocol,
                seed=seed,
                certificates_replayed=120,
                root="/artifacts/frozen",
                output_sha256={"models.npz": "0" * 64},
            )
            path = root / "reports/supplementary-calibration" / f"{protocol}-{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(r))
    (root / "configs").mkdir()
    (root / "configs/human-supplementary-heads-v1.json").write_text(json.dumps(cfg))
    (root / "configs/human-supplementary-calibration-v1.json").write_text(
        json.dumps(dict(state_counts=[16, 32, 64, 128, 256]))
    )
    return root / "reports/supplementary-heads/known_users-context-20261015.json"


def test_complete_inventory_is_frozen_without_scores(tmp_path):
    fixture(tmp_path)
    r = module.build_config(tmp_path)
    assert r["status"] == "execution_frozen" and len(r["probes"]) == 10
    assert sum(len(p["heads"]) for p in r["probes"]) == 30
    assert not (tmp_path / "configs/human-supplementary-final-v1.json").exists()


@pytest.mark.parametrize("change", ["missing", "failed", "endpoint", "seed", "duplicate_probe"])
def test_partial_or_changed_matrix_cannot_be_frozen(tmp_path, change):
    path = fixture(tmp_path)
    if change == "missing":
        path.unlink()
    elif change == "duplicate_probe":
        p = tmp_path / "configs/human-supplementary-heads-v1.json"
        r = json.loads(p.read_text())
        r["parent_probes"][-1] = r["parent_probes"][0]
        p.write_text(json.dumps(r))
    else:
        r = json.loads(path.read_text())
        if change == "failed":
            r["status"] = "failed"
        elif change == "endpoint":
            r["records"][0]["endpoint"] = "different"
        else:
            r["seed"] = 0
        path.write_text(json.dumps(r))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.build_config(tmp_path)
