import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "figure_tables", Path(__file__).parents[1] / "tools/build_human_figure_tables.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frontier_preserves_state_capacity_families_paired_intervals_and_primary_identity():
    metric = dict(users=20, cases=100, nll_equal_user_mean=0.5, brier_equal_user_mean=0.2)
    names = ["parent_k256", "nested.fixed", "teacher_k16", "pooled_count_k32", "gru"]
    report = dict(
        protocol="known_users",
        seed=20261015,
        primary="nested.fixed",
        metrics={"all/" + n: metric for n in names},
        comparisons={
            "all": {
                n + "-vs-parent": dict(candidate_minus_reference_nats=0.01, ci95=[0.005, 0.015])
                for n in names
                if n != "parent_k256"
            }
        },
    )
    rows = module.frontier_rows(
        report, dict(records=[dict(name="fixed", kind="strict", reduced_latent_states=256)])
    )
    assert [r["latent_states"] for r in rows] == [256, 256, 16, 32, None]
    assert [r["family"] for r in rows] == [
        "parent",
        "strict",
        "independent_teacher_quantization",
        "pooled_count_model",
        "sequence_or_markov",
    ]
    assert rows[0]["ci95"] == [0, 0]
    assert rows[1]["primary"]
    assert rows[1]["ci95"] == [0.005, 0.015]


def test_human_figures_reject_partial_seed_matrix(tmp_path, monkeypatch):
    import json
    import pytest

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/final-analysis-freeze.json").write_text(json.dumps({"controllers": {}}))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Complete fixed matrices"):
        module.main()
    assert not (tmp_path / "reports/figures/human-final-tables.json").exists()


def test_statistical_seed_and_cohort_fields_are_preserved_without_duplicate_keywords():
    import pytest

    diagnostic = dict(seed=17, cases=100, users=20, ci95=[0.1, 0.2])
    row = module.annotated_row(diagnostic, seed=42, protocol="known_users", cases=100, users=20)
    assert row["seed"] == 42 and row["bootstrap_seed"] == 17
    assert row["cases"] == 100 and row["users"] == 20 and row["ci95"] == [0.1, 0.2]
    assert diagnostic["seed"] == 17 and "bootstrap_seed" not in diagnostic
    with pytest.raises(ValueError, match="cohort identity"):
        module.annotated_row(diagnostic, users=21)
    cell = module.annotated_row(dict(predicted=0.5), seed=42, method="gru")
    assert cell == dict(predicted=0.5, seed=42, method="gru")
