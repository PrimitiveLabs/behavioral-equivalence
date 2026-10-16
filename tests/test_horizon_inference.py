"""Cloud numerical bridge between frozen encoders and marginal horizon heads."""

import numpy as np
import pytest


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("architecture", ["context", "gru", "transformer"])
def test_penultimate_origin_features_reproduce_forward_forecasts_and_batches(architecture):
    from behavioral_equivalence.content_neural import build_content_model

    torch.set_num_threads(2)
    torch.manual_seed(20261015)
    model = build_content_model(architecture, 10, 16).eval()
    tokens = torch.zeros((7, 32), dtype=torch.long)
    tokens[:, -3:] = torch.tensor([1, 4, 8])
    tabs = torch.arange(7, dtype=torch.long)
    items = torch.arange(7, dtype=torch.long)
    context = torch.arange(14, dtype=torch.float32).reshape(7, 2) / 14
    args = [tokens, tabs, items, context]
    with torch.inference_mode():
        logits = model(*args)
        features = model.output[:-1](model.representation(*args))
        factored = model.output[-1](features).flatten()
        batched = torch.cat([model(*[a[slice(i, i + 2)] for a in args]) for i in range(0, 7, 2)])
    np.testing.assert_allclose(logits.numpy(), factored.numpy(), atol=1e-7, rtol=0)
    np.testing.assert_allclose(logits.numpy(), batched.numpy(), atol=1e-6, rtol=0)
    assert features.shape == (7, 16)
