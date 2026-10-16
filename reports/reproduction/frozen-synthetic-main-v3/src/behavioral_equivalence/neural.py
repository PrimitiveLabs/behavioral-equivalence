"""Small sequence predictors for cloud pilot comparisons."""

import torch
from torch import nn


class SequencePredictor(nn.Module):
    def __init__(self, architecture: str, hidden: int = 64, history: int = 32):
        super().__init__()
        if architecture not in ("gru", "transformer"):
            raise ValueError("Unknown sequence architecture")
        self.architecture = architecture
        self.embedding = nn.Embedding(18, hidden, padding_idx=0)
        self.position = nn.Embedding(history, hidden) if architecture == "transformer" else None
        if architecture == "gru":
            self.encoder = nn.GRU(hidden, hidden, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(
                hidden, 4, hidden * 2, dropout=0.1, batch_first=True, norm_first=False
            )
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.context = nn.Embedding(15, 16)
        self.output = nn.Linear(hidden + 16, 1)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        if self.architecture == "gru":
            _, hidden = self.encoder(embedded)
            return hidden[-1]
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        embedded = embedded + self.position(positions)
        # Every nonpadding token is preceding available history. Attention over
        # that history is permitted; current and future targets are absent.
        encoded = self.encoder(embedded, src_key_padding_mask=tokens == 0)
        return encoded[:, -1]

    def forward(self, tokens: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        return self.output(torch.cat([self.encode(tokens), self.context(tab)], dim=1)).squeeze(-1)
