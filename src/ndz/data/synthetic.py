"""Synthetic motor-decoding dataset (no external download, no PHI).

Generates a smooth 2-D "reach velocity" latent and a population of neurons with
cosine direction tuning firing Poisson spikes. This is decodable, so baselines
show non-trivial R^2 and the zoo runs end-to-end out of the box. It mimics the
structure of intracortical motor BCI data (e.g. NLB MC_Maze) without being it.
"""

from __future__ import annotations

import numpy as np

from .base import NeuralDataset


def make_reach_dataset(
    n_timepoints: int = 6000,
    n_channels: int = 80,
    bin_size_s: float = 0.020,
    n_reach_targets: int = 8,
    noise: float = 1.0,
    seed: int = 0,
) -> NeuralDataset:
    """Cosine-tuned Poisson population encoding a smooth 2-D velocity."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_timepoints)

    # Smooth 2-D velocity built from a sequence of reaches to random targets.
    seg = max(1, n_timepoints // (n_reach_targets * 4))
    angles = rng.uniform(0, 2 * np.pi, size=n_timepoints // seg + 1)
    ang = np.repeat(angles, seg)[:n_timepoints]
    speed = 0.5 + 0.5 * np.sin(2 * np.pi * t / (seg * 4))
    vx = speed * np.cos(ang)
    vy = speed * np.sin(ang)
    # low-pass smooth (movement is smooth)
    k = np.ones(5) / 5
    vx = np.convolve(vx, k, mode="same")
    vy = np.convolve(vy, k, mode="same")
    Y = np.stack([vx, vy], axis=1)

    # Each neuron has a preferred direction; rate = baseline + gain * cos tuning.
    pref = rng.uniform(0, 2 * np.pi, size=n_channels)
    gain = rng.uniform(0.5, 1.5, size=n_channels)
    base = rng.uniform(0.5, 2.0, size=n_channels)
    direction = np.arctan2(vy, vx)
    mag = np.hypot(vx, vy)[:, None]
    tuning = np.cos(direction[:, None] - pref[None, :])
    rate = base[None, :] + gain[None, :] * mag * tuning
    rate = np.clip(rate, 0.05, None) * noise
    X = rng.poisson(rate).astype(np.float64)

    return NeuralDataset(
        X=X,
        Y=Y,
        bin_size_s=bin_size_s,
        name="synthetic_reach",
        task="regression",
        channel_names=[f"unit_{i:03d}" for i in range(n_channels)],
        meta={"generator": "make_reach_dataset", "seed": seed, "noise": noise},
    )


def make_classification_dataset(
    n_timepoints: int = 4000,
    n_channels: int = 64,
    n_classes: int = 4,
    bin_size_s: float = 0.020,
    seed: int = 0,
) -> NeuralDataset:
    """Discrete movement-intent classification variant."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n_timepoints)
    centers = rng.normal(0, 1.5, size=(n_classes, n_channels))
    X = np.clip(centers[labels] + rng.normal(0, 1.0, size=(n_timepoints, n_channels)), 0, None)
    return NeuralDataset(
        X=X,
        Y=labels,
        bin_size_s=bin_size_s,
        name="synthetic_intent",
        task="classification",
        channel_names=[f"unit_{i:03d}" for i in range(n_channels)],
        meta={"generator": "make_classification_dataset", "seed": seed},
    )
