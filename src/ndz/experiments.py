"""Deployment-oriented experiments: calibration efficiency, nonstationarity,
and a simple closed-loop-style cursor simulation.

These probe the things a real BCI cares about beyond a single accuracy number:
how much calibration data you need, how fast performance decays after
calibration, and how a decoded velocity actually drives a cursor.
"""

from __future__ import annotations

import numpy as np

from .data.base import NeuralDataset
from .evaluate import temporal_split
from .metrics import r2_score


def calibration_curve(make_decoder, ds: NeuralDataset,
                      minutes=(0.5, 1, 2, 4, 8, 16), test_frac: float = 0.25):
    """R^2 on a fixed held-out tail vs amount of (most-recent) calibration data.

    Mirrors a real session: a fixed block is the 'use' period; we train on the
    last m minutes immediately before it. Shows the calibration burden.
    """
    n = ds.n_timepoints
    cut = int(round(n * (1.0 - test_frac)))
    Xte, Yte = ds.X[cut:], ds.Y[cut:]
    bins_per_min = 60.0 / ds.bin_size_s
    rows = []
    for m in minutes:
        nb = int(round(m * bins_per_min))
        if nb < 50:
            continue
        nb = min(nb, cut)
        dec = make_decoder().fit(ds.X[cut - nb:cut], ds.Y[cut - nb:cut])
        rows.append({"minutes": round(min(m, cut / bins_per_min), 2),
                     "train_bins": nb, "r2": round(r2_score(Yte, dec.predict(Xte)), 4)})
        if nb == cut:
            break
    return rows


def nonstationarity_curve(make_decoder, ds: NeuralDataset,
                          train_frac: float = 0.4, n_windows: int = 8):
    """R^2 in successive windows after calibration -> performance decay over time.

    Train once on an early block, then evaluate in equal windows marching forward
    through the rest of the session. A downward slope is drift / nonstationarity:
    the brain's signal statistics shift away from what the decoder was calibrated on.
    """
    n = ds.n_timepoints
    cut = int(round(n * train_frac))
    dec = make_decoder().fit(ds.X[:cut], ds.Y[:cut])
    rest0, rest1 = cut, n
    edges = np.linspace(rest0, rest1, n_windows + 1).astype(int)
    bins_per_min = 60.0 / ds.bin_size_s
    rows = []
    for k in range(n_windows):
        a, b = edges[k], edges[k + 1]
        if b - a < 20:
            continue
        mins_after = round((a - cut) / bins_per_min, 2)
        rows.append({"win": k, "minutes_after_calib": mins_after,
                     "r2": round(r2_score(ds.Y[a:b], dec.predict(ds.X[a:b])), 4)})
    return rows


def cursor_simulation(decoder, ds: NeuralDataset, test_frac: float = 0.25):
    """Integrate decoded vs true velocity into 2-D cursor paths.

    Open-loop trajectory reconstruction: drive a cursor with the decoder's
    velocity output and compare its integrated path to the path the true velocity
    would trace. Quantifies the offline->trajectory gap (a real closed loop adds a
    user adapting in real time, which this does not simulate).
    """
    sp = temporal_split(ds, test_frac)
    decoder.fit(sp.X_train, sp.Y_train)
    vel_pred = decoder.predict(sp.X_test)
    vel_true = sp.Y_test
    dt = ds.bin_size_s
    pos_pred = np.cumsum(vel_pred, axis=0) * dt
    pos_true = np.cumsum(vel_true, axis=0) * dt
    pos_err = np.linalg.norm(pos_pred - pos_true, axis=1)
    scale = np.linalg.norm(pos_true - pos_true.mean(0), axis=1).mean() + 1e-9
    return {
        "pos_pred": pos_pred, "pos_true": pos_true,
        "pos_err": pos_err,
        "final_drift": float(pos_err[-1]),
        "median_drift": float(np.median(pos_err)),
        "normalized_drift": float(np.median(pos_err) / scale),
        "vel_r2": round(r2_score(vel_true, vel_pred), 4),
    }
