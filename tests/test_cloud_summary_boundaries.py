"""Local execution of controller function bodies without Modal decoration/submission."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def body(controller, name, **bindings):
    source = ROOT / "cloud" / controller
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    scope = {"Path": Path, "__file__": str(source), **bindings}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("campaign", ["../escape", "a/b", "", "name_with_underscore"])
def test_coverage_controller_rejects_unsafe_campaign_before_io(campaign):
    run = body("coverage_summary.py", "run")
    with pytest.raises(ValueError, match="Invalid campaign"):
        run(campaign)


def fixture(tmp_path):
    records = []
    for support in ["unknown_terminal_rows", "known_closed_terminal_labels"]:
        for i in range(2000):
            contained = support == "known_closed_terminal_labels" and i < 1000
            records.append(
                dict(
                    support=support,
                    properties=[
                        dict(
                            enclosure_containment_count=32 if contained else 0,
                            proven_noncoverage_count=0 if contained or i % 2 else 1,
                        )
                    ],
                )
            )
    manifest = dict(
        status="complete",
        point_evaluations=8000,
        chunks=[
            dict(spec=dict(size=32, regime=regime), records=records)
            for regime in ["uniform_active", "imbalanced_active"]
        ],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def classify(tmp_path):
    calls = []
    volume = SimpleNamespace(
        reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")
    )
    run = body(
        "nominal_failure_counts.py",
        "classify",
        FOLDER=str(tmp_path),
        EXPECTED_SHA=hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest(),
        volume=volume,
    )
    return run, calls


def test_failure_classification_preserves_disjoint_numeric_categories(tmp_path):
    fixture(tmp_path)
    run, calls = classify(tmp_path)
    result = run()
    report = json.loads((Path(result["root"]) / "report.json").read_text())
    assert calls == ["reload", "commit"]
    assert len(report["cells"]) == 4
    assert all(c["datasets"] == 2000 for c in report["cells"])
    assert all(
        c["contained"] + c["proven_noncoverage"] + c["unresolved_without_proven_noncoverage"]
        == 2000
        for c in report["cells"]
    )
    assert {c["contained"] for c in report["cells"]} == {0, 1000}
    assert (
        hashlib.sha256((Path(result["root"]) / "report.json").read_bytes()).hexdigest()
        == result["report_sha256"]
    )


@pytest.mark.parametrize("damage", ["hash", "status", "total", "contradiction", "cell_count"])
def test_failure_classification_rejects_invalid_retained_manifest(tmp_path, damage):
    m = fixture(tmp_path)
    run, calls = classify(tmp_path)
    if damage == "hash":
        (tmp_path / "manifest.json").write_text("changed")
    else:
        if damage == "status":
            m["status"] = "partial"
        elif damage == "total":
            m["point_evaluations"] = 7999
        elif damage == "contradiction":
            m["chunks"][0]["records"][0]["properties"][0].update(
                enclosure_containment_count=32, proven_noncoverage_count=1
            )
        else:
            m["chunks"][0]["records"].pop()
        (tmp_path / "manifest.json").write_text(json.dumps(m))
        run, calls = classify(tmp_path)
    with pytest.raises(ValueError):
        run()
    assert "commit" not in calls
