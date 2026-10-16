import pytest

torch = pytest.importorskip("torch", reason="neural model tests run in the pinned Modal image")


@pytest.mark.parametrize("architecture", ["gru", "transformer"])
def test_exposure_model_has_finite_predictions_and_trains_all_registered_parameters(architecture):
    from behavioral_equivalence.exposure_neural import ExposurePredictor

    torch.manual_seed(7)
    model = ExposurePredictor(architecture, 10, hidden=16)
    tokens = torch.zeros((3, 32), dtype=torch.long)
    tokens[:, -3:] = torch.tensor([1, 2, 17])
    tabs = torch.tensor([0, 1, 14])
    items = torch.tensor([0, 1, 10])
    logits = model(tokens, tabs, items)
    assert logits.shape == (3,) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    assert model.representation(tokens, tabs, items).shape == (3, 48)
    torch.testing.assert_close(model(tokens, tabs, items), model(tokens, tabs, items))
