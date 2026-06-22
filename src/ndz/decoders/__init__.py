"""Decoder registry. Torch-based decoders are included only if torch is present."""

from __future__ import annotations

from .base import Decoder
from .kalman import KalmanDecoder
from .latent import ReducedRankDecoder
from .linear import LogisticDecoder, RidgeDecoder

__all__ = [
    "Decoder",
    "RidgeDecoder",
    "LogisticDecoder",
    "KalmanDecoder",
    "ReducedRankDecoder",
    "build_decoders",
    "HAS_TORCH",
]

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def build_decoders(task: str = "regression", include_torch: bool = True) -> list[Decoder]:
    """Default decoder lineup for a task, in increasing complexity."""
    if task == "regression":
        zoo: list[Decoder] = [RidgeDecoder(), KalmanDecoder(), ReducedRankDecoder()]
        if include_torch and HAS_TORCH:
            from .mlp import MLPDecoder
            from .sequence import GRUDecoder, TCNDecoder
            from .transformer import TransformerDecoder

            zoo += [MLPDecoder(), TCNDecoder(), GRUDecoder(), TransformerDecoder()]
        return zoo
    if task == "classification":
        zoo = [LogisticDecoder()]
        if include_torch and HAS_TORCH:
            from .mlp import MLPDecoder

            zoo += [MLPDecoder()]
        return zoo
    raise ValueError(f"unknown task {task!r}")


def decoder_factories(task: str = "regression", include_torch: bool = True):
    """Return (name, zero-arg factory) pairs for repeated refits."""
    if task == "regression":
        facs = [("ridge_wiener", RidgeDecoder), ("kalman", KalmanDecoder),
                ("reduced_rank", ReducedRankDecoder)]
        if include_torch and HAS_TORCH:
            from .mlp import MLPDecoder
            from .sequence import GRUDecoder, TCNDecoder
            from .transformer import TransformerDecoder

            facs += [("mlp", MLPDecoder), ("tcn", TCNDecoder), ("gru", GRUDecoder),
                     ("transformer", TransformerDecoder)]
        return facs
    if task == "classification":
        facs = [("logistic", LogisticDecoder)]
        if include_torch and HAS_TORCH:
            from .mlp import MLPDecoder

            facs.append(("mlp", MLPDecoder))
        return facs
    raise ValueError(f"unknown task {task!r}")
