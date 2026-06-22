"""Neural Decoder Zoo: from linear baselines to sequence models for
brain-to-behavior decoding. All data is synthetic or public; no PHI."""

__version__ = "0.1.0"

from .data import NeuralDataset, load_dataset
from .decoders import build_decoders
from .evaluate import evaluate_decoder

__all__ = ["NeuralDataset", "load_dataset", "build_decoders", "evaluate_decoder", "__version__"]
