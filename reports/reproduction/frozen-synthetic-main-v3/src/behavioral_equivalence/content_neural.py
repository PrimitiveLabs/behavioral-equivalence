"""Content-aware sequence comparators and an explicit no-history ablation."""

import torch
from torch import nn
from .exposure_neural import ExposurePredictor


class ContentSequencePredictor(ExposurePredictor):
    def __init__(self, architecture: str, vocabulary_size: int, hidden: int = 128):
        super().__init__(architecture, vocabulary_size, hidden)
        self.output = nn.Sequential(nn.Linear(hidden + 34, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def representation(self, tokens, tab, item, context):
        return torch.cat([super().representation(tokens, tab, item), context], dim=1)

    def forward(self, tokens, tab, item, context):
        return self.output(self.representation(tokens, tab, item, context)).squeeze(-1)


class ContextOnlyPredictor(nn.Module):
    def __init__(self, vocabulary_size: int, hidden: int = 128):
        super().__init__()
        self.context = nn.Embedding(15, 16)
        self.item = nn.Embedding(vocabulary_size + 1, 16)
        self.output = nn.Sequential(nn.Linear(34, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def representation(self, _tokens, tab, item, context):
        return torch.cat([self.context(tab), self.item(item), context], dim=1)

    def forward(self, tokens, tab, item, context):
        return self.output(self.representation(tokens, tab, item, context)).squeeze(-1)


def build_content_model(architecture: str, vocabulary_size: int, hidden: int = 128):
    if architecture == "context":
        return ContextOnlyPredictor(vocabulary_size, hidden)
    return ContentSequencePredictor(architecture, vocabulary_size, hidden)
