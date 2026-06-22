"""Loader for a public intracortical decoding dataset (Neural Latents Benchmark).

MC_Maze (monkey primary motor / premotor cortex spikes -> hand kinematics) is a
clean, citable brain-to-behavior benchmark on DANDI. This module is a thin,
documented loader; the heavy deps (`dandi`, `pynwb`) are optional so the rest of
the zoo runs without them. No PHI. This is public animal data.

Download once. The ~29 MB Small split is the quick path:
    pip install "neural-decoder-zoo[nlb]"
    dandi download "DANDI:000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
then point --nwb-path at the downloaded .nwb file. (Full set: DANDI:000128.)

Verified against the NWB layout: spikes in `units.spike_times`; hand velocity in
`processing['behavior'].data_interfaces['hand_vel']` (a 2-D TimeSeries).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import NeuralDataset


def load_mc_maze(nwb_path: str | Path, bin_size_s: float = 0.020) -> NeuralDataset:
    """Load MC_Maze spikes + hand velocity from a local NWB file.

    Kept intentionally small; binning logic is explicit so reviewers can audit it.
    """
    try:
        from pynwb import NWBHDF5IO  # noqa: WPS433 (optional dep)
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError(
            "pynwb is required for the NLB loader. Install with "
            "`pip install \"neural-decoder-zoo[nlb]\"` and download the dataset via "
            "`dandi download \"DANDI:000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb\"`."
        ) from exc

    nwb_path = Path(nwb_path)
    if not nwb_path.exists():
        raise FileNotFoundError(
            f"{nwb_path} not found. Download MC_Maze_Small with "
            "`dandi download \"DANDI:000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb\"`."
        )

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        units = nwb.units
        spike_times = [np.asarray(units["spike_times"][i]) for i in range(len(units))]
        hand = nwb.processing["behavior"].data_interfaces["hand_vel"]
        beh = np.asarray(hand.data[:])
        beh_t = np.asarray(hand.timestamps[:])

    t0, t1 = beh_t[0], beh_t[-1]
    edges = np.arange(t0, t1, bin_size_s)
    X = np.stack([np.histogram(st, bins=edges)[0] for st in spike_times], axis=1).astype(np.float64)
    centers = edges[:-1] + bin_size_s / 2
    Y = np.stack([np.interp(centers, beh_t, beh[:, d]) for d in range(beh.shape[1])], axis=1)

    return NeuralDataset(
        X=X, Y=Y, bin_size_s=bin_size_s, name="nlb_mc_maze", task="regression",
        channel_names=[f"unit_{i:03d}" for i in range(X.shape[1])],
        meta={"source": "DANDI:000128 MC_Maze", "nwb_path": str(nwb_path)},
    )
