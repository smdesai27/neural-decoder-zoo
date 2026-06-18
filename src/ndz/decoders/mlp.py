"""MLP decoder (PyTorch, optional).

Operates on a causal window of spike history flattened into a feature vector.
Torch is an optional dependency; importing this module without torch raises a
clear message rather than failing the whole zoo.
"""

from __future__ import annotations

import numpy as np

from .base import Decoder, add_history

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional path
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for neural decoders. Install with "
            "`pip install \"neural-decoder-zoo[torch]\"`."
        )


class MLPDecoder(Decoder):
    name = "mlp"

    def __init__(self, n_lags: int = 4, hidden: int = 128, epochs: int = 60,
                 lr: float = 1e-3, batch: int = 256, seed: int = 0):
        _require_torch()
        self.n_lags = n_lags
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed

    def _build(self, in_dim: int, out_dim: int):
        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, out_dim),
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MLPDecoder":
        Xh = add_history(X, self.n_lags)
        self._mu, self._sd = Xh.mean(0), Xh.std(0) + 1e-8
        Xn = (Xh - self._mu) / self._sd
        self._classify = Y.ndim == 1
        out_dim = int(Y.max() + 1) if self._classify else Y.shape[1]
        self._net = self._build(Xn.shape[1], out_dim)
        xt = torch.tensor(Xn, dtype=torch.float32)
        if self._classify:
            yt = torch.tensor(Y, dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
        else:
            yt = torch.tensor(Y, dtype=torch.float32)
            loss_fn = nn.MSELoss()
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
        Xn = (add_history(X, self.n_lags) - self._mu) / self._sd
        with torch.no_grad():
            out = self._net(torch.tensor(Xn, dtype=torch.float32))
            if self._classify:
                return out.argmax(1).numpy()
            return out.numpy()
