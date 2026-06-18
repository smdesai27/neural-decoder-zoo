"""Linear decoders: ridge-regularized Wiener filter and logistic classifier.

These are the baselines every neural decoder must beat. A surprising amount of
intracortical decoding performance comes from a well-regularized linear map with
a few lags of spike history.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .base import Decoder, add_history


class RidgeDecoder(Decoder):
    name = "ridge_wiener"

    def __init__(self, alpha: float = 1.0, n_lags: int = 4):
        self.alpha = alpha
        self.n_lags = n_lags
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeDecoder":
        Xh = self._scaler.fit_transform(add_history(X, self.n_lags))
        self._model.fit(Xh, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(add_history(X, self.n_lags)))


class LogisticDecoder(Decoder):
    name = "logistic"

    def __init__(self, C: float = 1.0, n_lags: int = 2, max_iter: int = 500):
        self.C = C
        self.n_lags = n_lags
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "LogisticDecoder":
        Xh = self._scaler.fit_transform(add_history(X, self.n_lags))
        self._model.fit(Xh, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(add_history(X, self.n_lags)))
