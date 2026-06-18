"""Sequence decoders (PyTorch, optional): GRU and a small TCN.

Both consume a sliding causal window of the neural population and predict the
current behavior, capturing temporal dynamics a memoryless linear map cannot.
"""

from __future__ import annotations

import numpy as np

from .base import Decoder

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional path
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for sequence decoders. Install with "
            "`pip install \"neural-decoder-zoo[torch]\"`."
        )


def _windows(X: np.ndarray, win: int) -> np.ndarray:
    """(T, C) -> (T, win, C) causal windows, edge-padded at the start."""
    T, C = X.shape
    pad = np.repeat(X[:1], win - 1, axis=0)
    Xp = np.concatenate([pad, X], axis=0)
    return np.stack([Xp[i:i + win] for i in range(T)], axis=0)


class _SeqBase(Decoder):
    def __init__(self, win: int = 12, hidden: int = 96, epochs: int = 40,
                 lr: float = 2e-3, batch: int = 128, seed: int = 0):
        _require_torch()
        self.win, self.hidden, self.epochs = win, hidden, epochs
        self.lr, self.batch, self.seed = lr, batch, seed

    def _make_net(self, n_channels: int, out_dim: int):  # overridden
        raise NotImplementedError

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "_SeqBase":
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-8
        Xw = _windows((X - self._mu) / self._sd, self.win)
        self._classify = Y.ndim == 1
        out_dim = int(Y.max() + 1) if self._classify else Y.shape[1]
        torch.manual_seed(self.seed)
        self._net = self._make_net(X.shape[1], out_dim)
        xt = torch.tensor(Xw, dtype=torch.float32)
        yt = torch.tensor(Y, dtype=torch.long if self._classify else torch.float32)
        loss_fn = nn.CrossEntropyLoss() if self._classify else nn.MSELoss()
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        n = len(xt)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss = loss_fn(self._net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xw = _windows((X - self._mu) / self._sd, self.win)
        with torch.no_grad():
            out = self._net(torch.tensor(Xw, dtype=torch.float32))
            return out.argmax(1).numpy() if self._classify else out.numpy()


class GRUDecoder(_SeqBase):
    name = "gru"

    def _make_net(self, n_channels: int, out_dim: int):
        hidden, win = self.hidden, self.win

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(n_channels, hidden, batch_first=True)
                self.head = nn.Linear(hidden, out_dim)

            def forward(self, x):
                _, h = self.gru(x)
                return self.head(h[-1])

        return Net()


class TCNDecoder(_SeqBase):
    name = "tcn"

    def _make_net(self, n_channels: int, out_dim: int):
        hidden, win = self.hidden, self.win

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(n_channels, hidden, 3, padding=1, dilation=1), nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(),
                    nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.ReLU(),
                )
                self.head = nn.Linear(hidden, out_dim)

            def forward(self, x):  # x: (B, win, C)
                z = self.net(x.transpose(1, 2))  # (B, hidden, win)
                return self.head(z[:, :, -1])

        return Net()
