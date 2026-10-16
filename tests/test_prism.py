import numpy as np
import pytest
from behavioral_equivalence.core import Chain
from behavioral_equivalence.prism import write_model
from behavioral_equivalence.uncertainty import IntervalChain


def test_interval_model_uses_dtmc_keyword_and_interval_edges(tmp_path):
    model = IntervalChain(
        np.array([[0.0, 0.2], [0.0, 1.0]]),
        np.array([[0.8, 1.0], [0.0, 1.0]]),
        ("active", "complete"),
    )
    path = tmp_path / "model.prism"
    write_model(model, path)
    text = path.read_text()
    assert text.startswith("dtmc\n")
    assert "[0,0.80000000000000004]" in text
    assert 'label "complete" = s=1;' in text


def test_export_rejects_label_injection_and_invalid_initial_state(tmp_path):
    c = Chain(np.ones((1, 1)), ('bad"label',), np.ones(1))
    with pytest.raises(ValueError):
        write_model(c, tmp_path / "invalid.prism")
    with pytest.raises(ValueError):
        write_model(c, tmp_path / "invalid.prism", 1)
