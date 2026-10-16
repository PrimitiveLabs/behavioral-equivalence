from behavioral_equivalence.parent_identity import parent_job_identity


def test_every_upstream_selector_binds_parent_resume_identity():
    args = ["immutable-training", "known_users", "gru", 20261015, "quantizer-a", "source-a"]
    baseline = parent_job_identity(*args)
    assert baseline == parent_job_identity(*args)
    alternatives = [
        "other-training",
        "unseen_users",
        "context",
        20261016,
        "quantizer-b",
        "source-b",
    ]
    for index, alternative in enumerate(alternatives):
        changed = list(args)
        changed[index] = alternative
        assert baseline != parent_job_identity(*changed)


def test_resume_key_fix_preserves_the_frozen_calibration_estimator():
    import ast
    from pathlib import Path

    def estimator(path):
        module = ast.parse(Path(path).read_text())
        return ast.dump(
            next(
                node
                for node in module.body
                if isinstance(node, ast.FunctionDef) and node.name == "estimate"
            )
        )

    assert estimator("cloud/estimate_calendar_parent.py") == estimator(
        "reports/source-snapshots/estimate-calendar-parent-c0c628bb9aaf.py"
    )
