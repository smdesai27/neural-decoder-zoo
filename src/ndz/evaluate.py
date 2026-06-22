"""Evaluation harness with contiguous temporal splits.

Neural time series are autocorrelated, so a shuffled split leaks the future into
training and inflates scores. We always train on an earlier contiguous block and
test on a later one.
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


def blocked_cv(make_decoder, ds: NeuralDataset, n_folds: int = 5,
               embargo_frac: float = 0.01) -> tuple[dict, list[dict]]:
    """Blocked k-fold cross-validation for time series, with an embargo gap.

    The timeline is cut into K contiguous test blocks; each is held out in turn
    and the model is trained on the rest, minus a small embargo window on either
    side of the test block so adjacent (autocorrelated) bins don't leak across.
    Gives mean +/- std error bars. (The headline single-number result still uses
    the strictly-forward temporal_split; blocked CV trades a little causality for
    variance estimates.) `make_decoder` is a zero-arg factory returning a fresh
    decoder per fold.
    """
    n = ds.n_timepoints
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    emb = int(embargo_frac * n)
    per_fold: list[dict] = []
    for k in range(n_folds):
        t0, t1 = edges[k], edges[k + 1]
        keep = np.ones(n, dtype=bool)
        keep[max(0, t0 - emb):min(n, t1 + emb)] = False
        dec = make_decoder().fit(ds.X[keep], ds.Y[keep])
        s = score(ds.task, ds.Y[t0:t1], dec.predict(ds.X[t0:t1]))
        per_fold.append(s)
    keys = list(per_fold[0].keys())
    agg = {"decoder": getattr(make_decoder(), "name", "decoder")}
    for key in keys:
        vals = [r[key] for r in per_fold]
        agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    return agg, per_fold
