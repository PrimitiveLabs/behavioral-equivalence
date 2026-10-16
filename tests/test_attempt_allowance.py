import pytest
from behavioral_equivalence.budget import check_attempt_allowance


def test_failed_attempt_remains_counted_while_source_revision_and_full_run_are_bounded():
    kw = dict(maximum_versions=2, maximum_calls=3)
    check_attempt_allowance("revised", ["failed-original"], new_call=True, **kw)
    check_attempt_allowance("revised", ["failed-original", "revised"], new_call=True, **kw)
    with pytest.raises(ValueError, match="source-attempt"):
        check_attempt_allowance("third", ["failed-original", "revised"], new_call=True, **kw)
    with pytest.raises(ValueError, match="coordinator-attempt"):
        check_attempt_allowance(
            "revised", ["failed-original", "revised", "revised"], new_call=True, **kw
        )
    check_attempt_allowance(
        "revised", ["failed-original", "revised", "revised"], new_call=False, **kw
    )
