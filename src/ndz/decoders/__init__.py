"""Decoder registry. Torch-based decoders are included only if torch is present."""

from __future__ import annotations

from .base import Decoder
from .kalman import KalmanDecoder
from .linear import LogisticDecoder, RidgeDecoder

__all__ = ["Decoder", "RidgeDecoder", "LogisticDecoder", "KalmanDecoder", "build_decoders", "HAS_TORCH"]

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def build_decoders(task: str = "regression", include_torch: bool = True) -> list[Decoder]:
    """Default decoder lineup for a task, in increasing complexity."""
    if task == "regression":
        zoo: list[Decoder] = [RidgeDecoder(), KalmanDecoder()]
        if include_torch and HAS_TORCH:
            from .mlp import MLPDecoder
            from .sequence import GRUDecoder, TCNDecoder

            zoo += [MLPDecoder(), TCNDecoder(), GRUDecoder()]
        return zoo
    if task == "classification":
        zoo = [LogisticDecoder()]
        if include_torch and HAS_TORCH:
            from .mlp import MLPDecoder

            zoo += [MLPDecoder()]
        return zoo
    raise ValueError(f"unknown task {task!r}")
