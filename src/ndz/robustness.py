"""Robustness and signal-validity probes.

Magnitude metrics can increase under stimulation artifact. Behavior-tied
metrics should not improve when the artifact carries no behavioral information.
"""

from __future__ import annotations

import numpy as np


def channel_dropout(X: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Zero out a random fraction of channels (simulated electrode loss)."""
    if frac <= 0:
        return X
    rng = np.random.default_rng(seed)
    Xd = X.copy()
    n_drop = int(round(frac * X.shape[1]))
    drop = rng.choice(X.shape[1], size=n_drop, replace=False)
    Xd[:, drop] = 0.0
    return Xd


def inject_stim_artifact(
    X: np.ndarray, event_rate_hz: float = 1.0, bin_size_s: float = 0.020,
    amplitude: float = 50.0, seed: int = 0,
) -> np.ndarray:
    """Add periodic saturating transients to mimic stimulation artifact.

    The injected events are short and channel-wide. They inflate peak-based
    activity metrics without adding behavior-predictive information.
    """
    rng = np.random.default_rng(seed)
    Xa = X.copy()
    T = X.shape[0]
    period = max(1, int(round(1.0 / (event_rate_hz * bin_size_s))))
    for t in range(period, T, period):
        jitter = rng.integers(-1, 2)
        idx = np.clip(t + jitter, 0, T - 1)
        Xa[idx] += amplitude * (1.0 + 0.2 * rng.standard_normal(X.shape[1]))
    return Xa


def naive_activity_score(X: np.ndarray) -> float:
    """Peak-deflection activity score.

    This is included as a failure case. A few large artifact transients inflate
    it regardless of behavioral relevance.
    """
    dev = np.abs(X - X.mean(0))
    return float(np.mean(dev.max(axis=0)))


def calibration_sizes(n: int, fractions=(0.1, 0.25, 0.5, 1.0)) -> list[int]:
    return [max(50, int(f * n)) for f in fractions]
