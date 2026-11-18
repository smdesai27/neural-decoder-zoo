"""Multi-subject, multi-recording synthetic data for the validation-gap demo.

The generator has two knobs that the validation protocols are sensitive to.

Target drift
    The decoded target ``y`` is an AR(1) process within each recording,
    ``y[t] = rho * y[t-1] + sqrt(1 - rho**2) * e[t]``. ``rho`` near 1 is a
    slowly drifting target, ``rho = 0`` is white noise.

Subject mapping
    Each subject maps the target to channels through
    ``sqrt(w) * W_shared + sqrt(1 - w) * W_subject``, where ``w`` is
    ``shared_weight``. ``w = 1`` means every subject uses the same mapping.

The subject-specific part also carries a slowly drifting, recording-specific
channel offset (electrode drift, impedance change, posture). It is scaled by
``sqrt(1 - w)``, like the subject-specific mapping, so a fully shared mapping
has no drift. A slow target plus slow channel drift is what lets random slices
look better than they are. A decoder trained on random samples has seen the
same stretch of drift as its test samples, so a chance correlation between
drift and target within a recording holds on the test set too. It does not
hold on a later block or on another recording.

Everything is deterministic given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiSubjectData:
    X: np.ndarray          # (n, channels)
    y: np.ndarray          # (n,)
    subject: np.ndarray    # (n,) subject id (str)
    recording: np.ndarray  # (n,) recording id, unique across subjects (str)
    t: np.ndarray          # (n,) time index within the recording
    params: dict


def _ar1(rng: np.random.Generator, n: int, rho: float, dims: int = 1) -> np.ndarray:
    out = np.empty((n, dims))
    out[0] = rng.normal(size=dims)
    innov = np.sqrt(max(0.0, 1.0 - rho ** 2))
    for i in range(1, n):
        out[i] = rho * out[i - 1] + innov * rng.normal(size=dims)
    return out


def make_multisubject(n_subjects: int = 6, n_recordings: int = 2, n_samples: int = 300,
                      n_channels: int = 48, target_rho: float = 0.99, shared_weight: float = 0.3,
                      drift_rho: float = 0.995, drift_scale: float = 2.5, noise: float = 1.0,
                      signal: float = 0.4, seed: int = 0) -> MultiSubjectData:
    """Generate ``n_subjects * n_recordings * n_samples`` samples.

    Args:
        target_rho: AR(1) autocorrelation of the target (0 = white noise).
        shared_weight: fraction of the latent-to-channel mapping shared by all subjects.
        drift_rho, drift_scale: autocorrelation and size of the recording-specific channel
            drift in the subject-specific part (scaled by ``sqrt(1 - shared_weight)``).
        noise: white channel noise; ``signal``: gain of the target in the channels.
    """
    if not 0.0 <= shared_weight <= 1.0:
        raise ValueError("shared_weight must be in [0, 1]")
    if not 0.0 <= target_rho < 1.0:
        raise ValueError("target_rho must be in [0, 1)")
    rng = np.random.default_rng(seed)
    w_shared = rng.normal(size=n_channels)
    a, b = np.sqrt(shared_weight), np.sqrt(1.0 - shared_weight)
    Xs, ys, subs, recs, ts = [], [], [], [], []
    for s in range(n_subjects):
        w_subject = rng.normal(size=n_channels)
        mapping = signal * (a * w_shared + b * w_subject)
        for r in range(n_recordings):
            y = _ar1(rng, n_samples, target_rho)[:, 0]
            drift = drift_scale * b * _ar1(rng, n_samples, drift_rho, n_channels)
            X = np.outer(y, mapping) + drift + noise * rng.normal(size=(n_samples, n_channels))
            Xs.append(X)
            ys.append(y)
            subs.append(np.full(n_samples, f"S{s:02d}"))
            recs.append(np.full(n_samples, f"S{s:02d}-R{r}"))
            ts.append(np.arange(n_samples))
    params = dict(n_subjects=n_subjects, n_recordings=n_recordings, n_samples=n_samples,
                  n_channels=n_channels, target_rho=target_rho, shared_weight=shared_weight,
                  drift_rho=drift_rho, drift_scale=drift_scale, noise=noise, signal=signal, seed=seed)
    return MultiSubjectData(np.vstack(Xs), np.concatenate(ys), np.concatenate(subs),
                            np.concatenate(recs), np.concatenate(ts), params)


# The three regimes the demo and the tests reproduce. Each changes one thing from the first.
REGIMES: dict[str, dict] = {
    "slow_target_subject_specific": dict(target_rho=0.99, shared_weight=0.15),
    "white_noise_target": dict(target_rho=0.0, shared_weight=0.15),
    "fully_shared_mapping": dict(target_rho=0.99, shared_weight=1.0),
}

REGIME_TITLES = {
    "slow_target_subject_specific": "Slow target, subject-specific mapping",
    "white_noise_target": "White-noise target",
    "fully_shared_mapping": "Fully shared mapping",
}
