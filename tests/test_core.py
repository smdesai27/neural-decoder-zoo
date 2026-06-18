"""Core tests — run without torch (linear + Kalman paths only)."""

import numpy as np
import pytest

from ndz.data import load_dataset, make_classification_dataset, make_reach_dataset
from ndz.decoders import KalmanDecoder, LogisticDecoder, RidgeDecoder
from ndz.evaluate import evaluate_decoder, temporal_split
from ndz.metrics import accuracy, r2_score
from ndz.robustness import channel_dropout, inject_stim_artifact, naive_activity_score


def test_synthetic_shapes():
    ds = make_reach_dataset(n_timepoints=1000, n_channels=40)
    assert ds.X.shape == (1000, 40)
    assert ds.Y.shape == (1000, 2)
    assert ds.task == "regression"


def test_loader_resolves():
    assert load_dataset("synthetic").task == "regression"
    assert load_dataset("classification").task == "classification"
    with pytest.raises(ValueError):
        load_dataset("does_not_exist")


def test_temporal_split_no_leakage():
    ds = make_reach_dataset(n_timepoints=1000)
    sp = temporal_split(ds, test_frac=0.25)
    assert len(sp.X_train) == 750 and len(sp.X_test) == 250
    # train block strictly precedes test block (contiguous, no shuffle)
    assert np.array_equal(sp.X_train[-1], ds.X[749])
    assert np.array_equal(sp.X_test[0], ds.X[750])


def test_r2_perfect_and_mean():
    y = np.random.default_rng(0).normal(size=(100, 2))
    assert r2_score(y, y) == pytest.approx(1.0, abs=1e-9)
    assert r2_score(y, np.tile(y.mean(0), (100, 1))) == pytest.approx(0.0, abs=1e-6)


def test_ridge_beats_mean_baseline():
    ds = make_reach_dataset(n_timepoints=3000, seed=1)
    res = evaluate_decoder(RidgeDecoder(), ds, measure_latency=False)
    assert res["r2"] > 0.2  # decodable signal exists; baseline must capture it


def test_kalman_runs_and_decodes():
    ds = make_reach_dataset(n_timepoints=3000, seed=2)
    res = evaluate_decoder(KalmanDecoder(), ds, measure_latency=False)
    assert res["r2"] > 0.1


def test_classification_baseline():
    ds = make_classification_dataset(n_timepoints=2000, n_classes=4, seed=3)
    res = evaluate_decoder(LogisticDecoder(), ds, measure_latency=False)
    assert res["accuracy"] > 0.5  # well above 0.25 chance


def test_channel_dropout_zeros_channels():
    X = np.ones((50, 10))
    Xd = channel_dropout(X, frac=0.5, seed=0)
    assert (Xd == 0).all(axis=0).sum() == 5


def test_artifact_inflates_magnitude_metric():
    ds = make_reach_dataset(n_timepoints=2000, seed=4)
    clean = naive_activity_score(ds.X)
    art = naive_activity_score(inject_stim_artifact(ds.X, bin_size_s=ds.bin_size_s, amplitude=80.0))
    assert art > clean  # the fragile metric is fooled by artifact
