"""Sequence predictors with a training-only current-item vocabulary."""

import torch
from torch import nn
from .neural import SequencePredictor


class ExposurePredictor(SequencePredictor):
    def __init__(self, architecture: str, vocabulary_size: int, hidden: int = 128):
        if vocabulary_size < 0:
            raise ValueError("Nonnegative item vocabulary size required")
        super().__init__(architecture, hidden=hidden, history=32)
        self.item = nn.Embedding(vocabulary_size + 1, 16)
        self.output = nn.Linear(hidden + 32, 1)

    def representation(
        self, tokens: torch.Tensor, tab: torch.Tensor, item: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([self.encode(tokens), self.context(tab), self.item(item)], dim=1)

    def forward(self, tokens: torch.Tensor, tab: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return self.output(self.representation(tokens, tab, item)).squeeze(-1)
