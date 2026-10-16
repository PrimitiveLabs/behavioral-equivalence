import pytest

torch = pytest.importorskip("torch", reason="content-aware model tests run on Modal")


@pytest.mark.parametrize("architecture", ["context", "gru", "transformer"])
def test_content_models_train_all_parameters_and_use_intrinsic_context(architecture):
    from behavioral_equivalence.content_neural import build_content_model

    torch.manual_seed(4)
    model = build_content_model(architecture, 10, hidden=16)
    tokens = torch.zeros((4, 32), dtype=torch.long)
    tokens[:, -1] = 17
    tab, item = torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3])
    context = torch.tensor([[0.1, 0], [0, 1], [-0.5, 0], [1.0, 0]], requires_grad=True)
    logits = model(tokens, tab, item, context)
    assert logits.shape == (4,) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert context.grad.abs().sum() > 0
    model.eval()
    if architecture == "context":
        changed = tokens.clone()
        changed[:, -1] = 2
        torch.testing.assert_close(
            model(tokens, tab, item, context), model(changed, tab, item, context)
        )
