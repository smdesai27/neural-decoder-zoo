"""Evaluation harness — contiguous temporal splits (no leakage).

Neural time series are autocorrelated, so a shuffled split leaks the future into
training and inflates scores. We always train on an earlier contiguous block and
test on a later one — the honest analogue of calibrate-then-use in a real BCI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.base import NeuralDataset
from .metrics import inference_latency_us, score


@dataclass
class SplitData:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray


def temporal_split(ds: NeuralDataset, test_frac: float = 0.25) -> SplitData:
    n = ds.n_timepoints
    cut = int(round(n * (1.0 - test_frac)))
    return SplitData(ds.X[:cut], ds.Y[:cut], ds.X[cut:], ds.Y[cut:])


def evaluate_decoder(decoder, ds: NeuralDataset, test_frac: float = 0.25,
                     measure_latency: bool = True) -> dict:
    sp = temporal_split(ds, test_frac)
    decoder.fit(sp.X_train, sp.Y_train)
    y_pred = decoder.predict(sp.X_test)
    result = {"decoder": getattr(decoder, "name", type(decoder).__name__)}
    result.update(score(ds.task, sp.Y_test, y_pred))
    if measure_latency:
        result["latency_us"] = round(inference_latency_us(decoder, sp.X_test), 1)
    return result
