"""Common dataset container for the decoder zoo.

A `NeuralDataset` is the single object every decoder and metric consumes, so
swapping a synthetic generator for a real public dataset changes nothing
downstream. All data here is synthetic or from public sources — no PHI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NeuralDataset:
    """Time-aligned neural features and behavior.

    X: (T, C) neural features per time bin (binned spike counts or LFP power).
    Y: (T, D) continuous behavior for regression (e.g. hand velocity x/y),
       or (T,) integer labels for classification.
    """

    X: np.ndarray
    Y: np.ndarray
    bin_size_s: float
    name: str
    task: str = "regression"  # "regression" | "classification"
    channel_names: list[str] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.Y = np.asarray(self.Y)
        if self.X.ndim != 2:
            raise ValueError(f"X must be (T, C); got {self.X.shape}")
        if len(self.X) != len(self.Y):
            raise ValueError(f"X and Y must share T; got {len(self.X)} vs {len(self.Y)}")
        if self.task not in {"regression", "classification"}:
            raise ValueError(f"unknown task {self.task!r}")

    @property
    def n_timepoints(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.Y.shape[1] if self.Y.ndim == 2 else int(self.Y.max() + 1)

    def __repr__(self) -> str:
        return (
            f"NeuralDataset(name={self.name!r}, task={self.task!r}, "
            f"T={self.n_timepoints}, C={self.n_channels}, D={self.n_outputs}, "
            f"bin={self.bin_size_s*1000:.0f}ms)"
        )
