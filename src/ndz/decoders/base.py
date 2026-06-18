"""Decoder interface. Every decoder is fit(X, Y) -> predict(X)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Decoder(ABC):
    name: str = "decoder"

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "Decoder":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


def add_history(X: np.ndarray, n_lags: int) -> np.ndarray:
    """Augment each time bin with the previous `n_lags` bins (causal Wiener design).

    Returns (T, C*(n_lags+1)); early bins are edge-padded so length is preserved.
    """
    if n_lags <= 0:
        return X
    T, C = X.shape
    cols = [X]
    for lag in range(1, n_lags + 1):
        shifted = np.empty_like(X)
        shifted[lag:] = X[:-lag]
        shifted[:lag] = X[0]
        cols.append(shifted)
    return np.concatenate(cols, axis=1)
