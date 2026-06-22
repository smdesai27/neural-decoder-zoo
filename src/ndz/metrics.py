"""Evaluation metrics for neural decoding."""

from __future__ import annotations

import time

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean coefficient of determination across output dimensions."""
    y_true = np.atleast_2d(y_true.T).T if y_true.ndim == 1 else y_true
    y_pred = np.atleast_2d(y_pred.T).T if y_pred.ndim == 1 else y_pred
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(0)) ** 2, axis=0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Pearson correlation across output dimensions."""
    yt = y_true.reshape(len(y_true), -1)
    yp = y_pred.reshape(len(y_pred), -1)
    cs = []
    for d in range(yt.shape[1]):
        c = np.corrcoef(yt[:, d], yp[:, d])[0, 1]
        cs.append(0.0 if np.isnan(c) else c)
    return float(np.mean(cs))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def score(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {"accuracy": accuracy(y_true, y_pred)}
    return {"r2": r2_score(y_true, y_pred), "corr": corr(y_true, y_pred)}


def inference_latency_us(decoder, X: np.ndarray, n: int = 200) -> float:
    """Median per-sample predict latency in microseconds (single-bin calls)."""
    n = min(n, len(X))
    times = []
    for i in range(n):
        sample = X[max(0, i - 12):i + 1]  # small causal context
        t0 = time.perf_counter()
        decoder.predict(sample)
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(times))
