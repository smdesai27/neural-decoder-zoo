"""Reduced-rank (PCA-latent) decoder.

Population activity is low-dimensional: most of the variance across many neurons
lives on a handful of shared dimensions (a 'neural manifold'). This decoder first
projects the spike features onto their top principal components, then does ridge
regression from that compact latent space to behavior. It is the cheap, linear
cousin of latent-dynamics models like LFADS, and a strong, interpretable baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .base import Decoder, add_history


class ReducedRankDecoder(Decoder):
    name = "reduced_rank"

    def __init__(self, n_components: int = 20, alpha: float = 1.0, n_lags: int = 4):
        self.n_components = n_components
        self.alpha = alpha
        self.n_lags = n_lags
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "ReducedRankDecoder":
        Xh = self._scaler.fit_transform(add_history(X, self.n_lags))
        k = min(self.n_components, Xh.shape[1], Xh.shape[0])
        self._pca = PCA(n_components=k)
        self._model.fit(self._pca.fit_transform(Xh), Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xh = self._scaler.transform(add_history(X, self.n_lags))
        return self._model.predict(self._pca.transform(Xh))
