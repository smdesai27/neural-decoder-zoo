"""Causal transformer decoder (PyTorch, optional).

A small self-attention encoder over a causal window of the neural population.
Included for completeness and as a teaching point: on a single small session a
transformer's weak inductive bias and data hunger usually leave it at or below
the GRU/TCN. Transformers earn their keep in neural decoding at scale, in
cross-session pretraining (the NDT / POYO / Brain-JEPA direction), not on one
modest dataset. Reuses the windowing + training loop from the sequence decoders.
"""

from __future__ import annotations

from .sequence import _SeqBase

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional path
    _HAS_TORCH = False


class TransformerDecoder(_SeqBase):
    name = "transformer"

    def __init__(self, win: int = 12, hidden: int = 64, heads: int = 4, layers: int = 2,
                 epochs: int = 40, lr: float = 2e-3, batch: int = 128, seed: int = 0):
        super().__init__(win=win, hidden=hidden, epochs=epochs, lr=lr, batch=batch, seed=seed)
        self.heads = heads
        self.layers = layers

    def _make_net(self, n_channels: int, out_dim: int):
        hidden, win, heads, layers = self.hidden, self.win, self.heads, self.layers

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(n_channels, hidden)
                self.pos = nn.Parameter(torch.zeros(1, win, hidden))
                enc = nn.TransformerEncoderLayer(
                    hidden, heads, dim_feedforward=hidden * 2, batch_first=True, dropout=0.1
                )
                self.tr = nn.TransformerEncoder(enc, num_layers=layers)
                self.head = nn.Linear(hidden, out_dim)
                self.win = win

            def forward(self, x):  # x: (B, win, C)
                h = self.proj(x) + self.pos
                # causal mask: position i may attend only to positions <= i
                mask = torch.triu(torch.ones(self.win, self.win, device=x.device), diagonal=1).bool()
                h = self.tr(h, mask=mask)
                return self.head(h[:, -1])

        return Net()
