"""How a decoder's score depends on how test data is separated from training data.

See docs/VALIDATION.md. Modules: ``splits`` (index generators), ``gap``
(score one decoder under each protocol), ``synthetic`` (no-download data),
``case_studies`` (load real-data results.json files), ``plot`` and ``demo``.
"""

from .gap import validation_gap
from .splits import (PROTOCOL_ORDER, blocked_in_time, held_out_recording, held_out_subject, last_fraction,
                     make_splits, purged_random, random_slices)

__all__ = ["PROTOCOL_ORDER", "blocked_in_time", "held_out_recording", "held_out_subject", "last_fraction",
           "make_splits", "purged_random", "random_slices", "validation_gap"]
