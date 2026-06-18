"""Datasets for the decoder zoo (synthetic + public NLB)."""

from .base import NeuralDataset
from .synthetic import make_classification_dataset, make_reach_dataset

__all__ = ["NeuralDataset", "make_reach_dataset", "make_classification_dataset", "load_dataset"]


def load_dataset(name: str, **kwargs) -> NeuralDataset:
    """Resolve a dataset by name. Real datasets load lazily to avoid heavy deps."""
    if name in {"synthetic", "synthetic_reach", "reach"}:
        return make_reach_dataset(**kwargs)
    if name in {"synthetic_intent", "intent", "classification"}:
        return make_classification_dataset(**kwargs)
    if name in {"mc_maze", "nlb", "nlb_mc_maze"}:
        from .nlb import load_mc_maze

        return load_mc_maze(**kwargs)
    raise ValueError(f"unknown dataset {name!r}")
