"""Kalman-filter decoder (Wu et al. 2006 style) — numpy only.

A linear Gaussian state-space model where the latent state is the behavior
(e.g. hand velocity) and observations are the neural population. Parameters are
fit in closed form by least squares; decoding is the standard predict/update
recursion. The canonical online intracortical-BCI decoder.
"""

from __future__ import annotations

import numpy as np

from .base import Decoder


class KalmanDecoder(Decoder):
    name = "kalman"

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "KalmanDecoder":
        Y = Y if Y.ndim == 2 else Y.reshape(-1, 1)
        self._xmu = X.mean(0)
        self._xsd = X.std(0) + 1e-8
        Z = ((X - self._xmu) / self._xsd).T  # (C, T) observations
        S = Y.T.astype(np.float64)           # (D, T) states
        S1, S2 = S[:, :-1], S[:, 1:]
        # state transition  x_t = A x_{t-1} + w,  w ~ N(0, W)
        self.A = S2 @ S1.T @ np.linalg.pinv(S1 @ S1.T)
        rW = S2 - self.A @ S1
        self.W = rW @ rW.T / max(S.shape[1] - 1, 1)
        # observation  z_t = C x_t + q,  q ~ N(0, Q)
        self.C = Z @ S.T @ np.linalg.pinv(S @ S.T)
        rQ = Z - self.C @ S
        self.Q = rQ @ rQ.T / S.shape[1]
        self._state0 = S[:, 0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = ((X - self._xmu) / self._xsd).T
        d = self.A.shape[0]
        x = self._state0.copy()
        P = self.W.copy()
        I = np.eye(d)
        out = np.zeros((Z.shape[1], d))
        for t in range(Z.shape[1]):
            x = self.A @ x
            P = self.A @ P @ self.A.T + self.W
            innov_cov = self.C @ P @ self.C.T + self.Q
            K = P @ self.C.T @ np.linalg.pinv(innov_cov)
            x = x + K @ (Z[:, t] - self.C @ x)
            P = (I - K @ self.C) @ P
            out[t] = x
        return out
