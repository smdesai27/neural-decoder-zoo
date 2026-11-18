"""Validation-protocol tests: no leakage, the three synthetic regimes, the case-study schema, the plot."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ndz.validation import splits as S
from ndz.validation.case_studies import (SchemaError, load_case_studies, load_results, scan_case_studies,
                                         to_rows, validate_results)
from ndz.validation.gap import bootstrap_subject_ci, validation_gap
from ndz.validation.plot import plot_validation_gap
from ndz.validation.synthetic import REGIMES, make_multisubject

FIXTURE = Path(__file__).resolve().parents[1] / "case_studies" / "_fixture" / "results.json"


def _layout(n_sub=4, n_rec=3, n=60):
    subject = np.repeat([f"S{i}" for i in range(n_sub)], n_rec * n)
    recording = np.tile(np.repeat([f"R{j}" for j in range(n_rec)], n), n_sub)  # ids reused across subjects
    t = np.tile(np.arange(n), n_sub * n_rec)
    return subject, recording, t


ALL = [
    ("random_slices", {}), ("purged_random", {"embargo": 4}), ("blocked_in_time", {"n_blocks": 4}),
    ("blocked_in_time", {"n_blocks": 4, "embargo": 3}), ("last_fraction", {"embargo": 2}),
    ("held_out_recording", {}), ("held_out_subject", {}),
]


# ---------------------------------------------------------------- leakage


@pytest.mark.parametrize("name,params", ALL)
def test_train_and_test_are_disjoint_and_non_empty(name, params):
    subject, recording, t = _layout()
    folds = S.make_splits(name, subject, recording, t, **params)
    assert folds
    for train, test in folds:
        assert len(train) and len(test)
        assert not np.intersect1d(train, test).size
        assert train.max() < len(t) and test.max() < len(t)


def _rec(subject, recording):
    return np.char.add(np.char.add(subject.astype(str), "|"), recording.astype(str))


@pytest.mark.parametrize("name,params", [p for p in ALL if p[1].get("embargo")])
def test_embargo_is_respected(name, params):
    subject, recording, t = _layout()
    rec = _rec(subject, recording)
    for train, test in S.make_splits(name, subject, recording, t, **params):
        for i in test:
            same = train[rec[train] == rec[i]]
            assert np.all(np.abs(t[same] - t[i]) > params["embargo"])


def test_purged_random_removes_training_neighbours_only_in_the_same_recording():
    subject, recording, t = _layout(n_sub=2, n_rec=2, n=50)
    (train, test), = S.purged_random(subject, recording, t, n_splits=1, test_frac=0.1, embargo=2)
    unpurged, _ = S.random_slices(subject, recording, t, n_splits=1, test_frac=0.1)[0]
    assert len(train) < len(unpurged)
    assert set(train) <= set(unpurged)


def test_held_out_subject_never_in_train():
    subject, recording, t = _layout()
    folds = S.held_out_subject(subject, recording, t)
    assert len(folds) == 4
    for train, test in folds:
        held = set(subject[test])
        assert len(held) == 1
        assert held.isdisjoint(subject[train])


def test_held_out_recording_keeps_the_subject_in_train():
    subject, recording, t = _layout()
    rec = _rec(subject, recording)
    folds = S.held_out_recording(subject, recording, t)
    assert len(folds) == 12  # recording ids are scoped by subject
    for train, test in folds:
        assert len(set(rec[test])) == 1 and set(rec[test]).isdisjoint(rec[train])
        assert set(subject[test]) <= set(subject[train])


def test_held_out_recording_skips_a_subject_with_one_recording():
    subject = np.array(["A"] * 20 + ["B"] * 10)
    recording = np.array(["r1"] * 10 + ["r2"] * 10 + ["r1"] * 10)
    t = np.tile(np.arange(10), 3)
    folds = S.held_out_recording(subject, recording, t)
    assert len(folds) == 2 and all("B" not in set(subject[te]) for _, te in folds)


def test_blocked_in_time_blocks_are_contiguous_and_cover_everything():
    subject, recording, t = _layout()
    rec = _rec(subject, recording)
    folds = S.blocked_in_time(subject, recording, t, n_blocks=4)
    covered = np.concatenate([te for _, te in folds])
    assert sorted(covered) == list(range(len(t)))
    for _, test in folds:
        for r in np.unique(rec[test]):
            tt = np.sort(t[test][rec[test] == r])
            assert np.all(np.diff(tt) == 1)


def test_last_fraction_tests_on_the_end_of_each_recording():
    subject, recording, t = _layout(n=40)
    (train, test), = S.last_fraction(subject, recording, t, test_frac=0.25)
    assert set(t[test]) == set(range(30, 40)) and set(t[train]) == set(range(30))


class _Spy(BaseEstimator, TransformerMixin):
    seen: list = []

    def fit(self, X, y=None):
        _Spy.seen.append(set(map(int, X[:, 0])))
        return self

    def transform(self, X):
        return X


def test_preprocessing_is_fit_on_the_training_fold_only():
    subject, recording, t = _layout(n_sub=3, n_rec=2, n=40)
    n = len(t)
    X = np.column_stack([np.arange(n), np.random.default_rng(0).normal(size=(n, 3))])
    y = np.random.default_rng(1).normal(size=n)
    _Spy.seen = []
    gap = validation_gap(X, y, subject, recording, t, lambda: Pipeline([("spy", _Spy()), ("m", Ridge())]),
                         protocols=["held_out_subject"], n_boot=50)
    folds = S.held_out_subject(subject, recording, t)
    assert [set(map(int, tr)) for tr, _ in folds] == _Spy.seen
    assert set(gap["protocols"]["held_out_subject"]["per_subject"]) == {"S0", "S1", "S2"}


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_subject_ci({"a": 0.2, "b": 0.5, "c": 0.8, "d": 0.4}, n_boot=500)
    assert lo < np.mean([0.2, 0.5, 0.8, 0.4]) < hi
    assert bootstrap_subject_ci({"a": 0.2}) == (None, None)


# ---------------------------------------------------------------- synthetic regimes

PROTOS = ["random_slices", "blocked_in_time", "held_out_subject"]


def _scores(regime, seed):
    d = make_multisubject(seed=seed, **REGIMES[regime])
    g = validation_gap(d.X, d.y, d.subject, d.recording, d.t, lambda: Ridge(alpha=1.0), protocols=PROTOS,
                       metric="pearson_r", n_boot=50)
    return {k: v["score"] for k, v in g["protocols"].items()}, g["optimism"]


# Margins checked over seeds 0-19 with the current scoring (Ridge alpha=1, Pearson r):
# random - blocked >= 0.105, blocked - held-out subject >= 0.123, optimism >= 0.288 (regime 1);
# |held-out subject - random| <= 0.0010 (regime 3).
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_slow_target_and_subject_mapping_order_the_protocols(seed):
    s, optimism = _scores("slow_target_subject_specific", seed)
    assert s["random_slices"] - s["blocked_in_time"] > 0.08
    assert s["blocked_in_time"] - s["held_out_subject"] > 0.03
    assert optimism > 0.15


# White-noise target (regime 2). Time carries no information about the target, but a small
# random-vs-blocked gap remains, and it comes from the slow channel drift, not from the scoring.
# Seeds 0-19, Ridge alpha=1, Pearson r, random_slices vs blocked_in_time:
#   drift_scale=0.0: random 0.807, blocked 0.807, gap in [-0.0012, +0.0016]  (max |gap| 0.0016)
#   drift_scale=2.5: random 0.641, blocked 0.583, gap in [+0.040, +0.074], mean +0.059
# Drift lowers both scores (on every seed); it lowers blocked more (-0.224) than random (-0.165).
WHITE_NOISE_SEEDS = range(20)
NO_DRIFT_MAX_GAP = 0.005       # observed max 0.0016 (3x headroom)
DRIFT_GAP_RANGE = (0.02, 0.10)  # observed 0.040 to 0.074 (2x headroom below, 1.35x above)


def _white_noise_scores(seed, **overrides):
    d = make_multisubject(seed=seed, **{**REGIMES["white_noise_target"], **overrides})
    g = validation_gap(d.X, d.y, d.subject, d.recording, d.t, lambda: Ridge(alpha=1.0),
                       protocols=["random_slices", "blocked_in_time"], metric="pearson_r", n_boot=10)
    return g["protocols"]["random_slices"]["score"], g["protocols"]["blocked_in_time"]["score"]


@pytest.mark.parametrize("seed", WHITE_NOISE_SEEDS)
def test_white_noise_target_without_drift_random_matches_blocked(seed):
    random, blocked = _white_noise_scores(seed, drift_scale=0.0)
    assert abs(random - blocked) < NO_DRIFT_MAX_GAP


@pytest.mark.parametrize("seed", WHITE_NOISE_SEEDS)
def test_white_noise_target_with_default_drift_leaves_a_small_gap(seed):
    assert make_multisubject(seed=seed, n_samples=3).params["drift_scale"] == 2.5  # the default
    random, blocked = _white_noise_scores(seed)  # default drift
    assert DRIFT_GAP_RANGE[0] < random - blocked < DRIFT_GAP_RANGE[1]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fully_shared_mapping_held_out_subject_matches_random(seed):
    s, optimism = _scores("fully_shared_mapping", seed)
    assert abs(optimism) < 0.03


def test_synthetic_is_deterministic_and_well_formed():
    a, b = make_multisubject(seed=3), make_multisubject(seed=3)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
    assert a.X.shape == (6 * 2 * 300, 48) and len(np.unique(a.recording)) == 12
    with pytest.raises(ValueError):
        make_multisubject(shared_weight=1.5)


# ---------------------------------------------------------------- schema


def test_fixture_is_valid_and_labelled_synthetic():
    obj = load_results(FIXTURE)
    assert "SYNTHETIC" in obj["title"]
    rows = to_rows(obj)
    main = [r for r in rows if r["variant"] is None]
    assert {r["protocol"] for r in main} == {"random_slices", "purged_random", "blocked_in_time",
                                             "held_out_subject"}  # a protocol may be missing
    assert any(r["ci_low"] is None for r in main)  # CI fields may be null
    assert any(r["variant"] == "ridge_alpha_1000" for r in rows)


def _close(a, b, tol=2e-4):
    """Same structure and values. The fixture stores four decimals, so a number may differ in
    its last digit on another platform."""
    numeric = (int, float)
    if isinstance(a, numeric) and isinstance(b, numeric) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def test_fixture_matches_its_generator():
    spec = importlib.util.spec_from_file_location("make_fixture", FIXTURE.with_name("make_fixture.py"))
    assert spec is not None and spec.loader is not None
    make_fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_fixture)
    committed = json.load(open(FIXTURE))
    assert committed["script"] == make_fixture.SCRIPT
    built = json.loads(json.dumps(make_fixture.build(generated=committed["generated"])))
    assert _close(built, committed)


def test_close_notices_a_changed_score_and_a_missing_field():
    committed = json.load(open(FIXTURE))
    changed = copy.deepcopy(committed)
    changed["protocols"][0]["score"] += 0.001
    assert not _close(changed, committed)
    missing = copy.deepcopy(committed)
    del missing["protocols"][0]["skipped"]
    assert not _close(missing, committed)
    assert _close(copy.deepcopy(committed), committed)


def test_variants_may_be_absent():
    obj = json.load(open(FIXTURE))
    del obj["variants"]
    obj["published_claim"] = None
    assert validate_results(obj) == []


@pytest.mark.parametrize("breakage", [
    lambda o: o.update(schema_version=2),
    lambda o: o.update(metric="r2"),
    lambda o: o.update(task="segmentation"),
    lambda o: o.update(generated="17 Nov 2026"),
    lambda o: o["protocols"][0].update(name="shuffle"),
    lambda o: o["protocols"][0].update(score=None),
    lambda o: o["protocols"][0].update(per_subject=[0.5]),
    lambda o: o["baseline"].update(value=None),
    lambda o: o["protocols"].append(copy.deepcopy(o["protocols"][0])),
    lambda o: o.pop("source"),
])
def test_schema_rejects_a_broken_file(tmp_path, breakage):
    obj = json.load(open(FIXTURE))
    breakage(obj)
    assert validate_results(obj)
    bad = tmp_path / "bad" / "results.json"
    bad.parent.mkdir()
    bad.write_text(json.dumps(obj))
    with pytest.raises(SchemaError):
        load_results(bad)


def test_directory_scan_skips_broken_files_and_private_fixtures(tmp_path):
    good = json.load(open(FIXTURE))
    (tmp_path / "_fixture").mkdir()
    (tmp_path / "_fixture" / "results.json").write_text(json.dumps(good))
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "results.json").write_text(json.dumps({**good, "dataset": "real"}))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "results.json").write_text("{not json")
    rows, problems = scan_case_studies(tmp_path)
    assert {r["dataset"] for r in rows} == {"synthetic_fixture", "real"}
    assert list(problems) == [str(tmp_path / "broken" / "results.json")]
    with pytest.warns(UserWarning):
        rows = load_case_studies(tmp_path, include_private=False)
    assert {r["dataset"] for r in rows} == {"real"}


# ---------------------------------------------------------------- plot


def test_plot_writes_a_file(tmp_path):
    rows = to_rows(load_results(FIXTURE))
    rows += [{**r, "dataset": "second", "title": "Second panel", "claim_low": None, "claim_high": None}
             for r in rows if r["variant"] is None]
    out = plot_validation_gap(rows, tmp_path / "figs" / "gap.png")
    assert out.exists() and out.stat().st_size > 10_000
    with pytest.raises(ValueError):
        plot_validation_gap([], tmp_path / "empty.png")
