"""Acceptance tests for ndz.validation.

Part 1 checks the split generators against brute-force definitions on awkward input:
uneven recordings, non-integer time, shuffled sample order, recording ids reused across subjects.
Part 2 checks that preprocessing is fitted on training rows only.
Part 3 checks that a decoder with nothing to exploit shows no gap between protocols.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from ndz.validation.gap import evaluate_protocol, validation_gap
from ndz.validation.splits import make_splits
from ndz.validation.synthetic import make_multisubject

SINGLE = ("S2", "S5")  # subjects given exactly one recording


def structure(seed=0, shuffle_seed=1):
    rng = np.random.default_rng(seed)
    subject, recording, t = [], [], []
    for s in range(7):
        n_rec = 1 if f"S{s}" in SINGLE else int(rng.integers(2, 4))
        for r in range(n_rec):
            n = int(rng.integers(23, 61))
            subject += [f"S{s}"] * n
            recording += [f"R{r}"] * n  # ids reused across subjects on purpose
            t += list(np.cumsum(rng.uniform(0.5, 1.5, n)))
    subject, recording, t = np.asarray(subject), np.asarray(recording), np.asarray(t)
    perm = np.random.default_rng(shuffle_seed).permutation(len(t))
    return subject[perm], recording[perm], t[perm]


def key(subject, recording):
    return np.char.add(np.char.add(subject, "|"), recording)


def expected_purged_train(test_mask, k, t, embargo):
    keep = ~test_mask
    for i in np.flatnonzero(test_mask):
        keep &= ~((k == k[i]) & (np.abs(t - t[i]) <= embargo))
    return np.flatnonzero(keep)


CASES = [("random_slices", {}), ("purged_random", {"embargo": 2.0}), ("blocked_in_time", {}),
         ("blocked_in_time", {"embargo": 2.0}), ("last_fraction", {}), ("last_fraction", {"embargo": 3.0}),
         ("held_out_recording", {}), ("held_out_subject", {})]


def test_every_protocol_gives_disjoint_nonempty_folds():
    s, r, t = structure()
    for name, params in CASES:
        folds = make_splits(name, s, r, t, **params)
        assert folds, name
        for tr, te in folds:
            assert len(tr) and len(te), name
            assert not set(tr) & set(te), name
            assert tr.min() >= 0 and max(tr.max(), te.max()) < len(t), name


def test_held_out_subject_never_shares_a_person():
    s, r, t = structure()
    folds = make_splits("held_out_subject", s, r, t)
    assert len(folds) == len(set(s))
    tested = np.concatenate([te for _, te in folds])
    assert sorted(tested) == list(range(len(t)))
    for tr, te in folds:
        assert len(set(s[te])) == 1
        assert not set(s[te]) & set(s[tr])


def test_held_out_recording_keeps_the_person_and_scopes_ids_by_subject():
    s, r, t = structure()
    k = key(s, r)
    folds = make_splits("held_out_recording", s, r, t)
    multi = {x for x in set(k) if x.split("|")[0] not in SINGLE}
    assert len(folds) == len(multi)
    for tr, te in folds:
        assert len(set(k[te])) == 1
        assert not set(k[te]) & set(k[tr])
        assert set(s[te]) <= set(s[tr])
        assert not set(s[te]) & set(SINGLE)
        # another subject's recording with the same id must stay in training
        same_id_other_subject = (r == r[te][0]) & (s != s[te][0])
        assert set(np.flatnonzero(same_id_other_subject)) <= set(tr)


def test_purged_random_matches_brute_force():
    s, r, t = structure()
    k = key(s, r)
    for tr, te in make_splits("purged_random", s, r, t, embargo=2.0, n_splits=3):
        mask = np.zeros(len(t), bool)
        mask[te] = True
        assert list(tr) == list(expected_purged_train(mask, k, t, 2.0))


def test_last_fraction_trains_strictly_before_it_tests():
    s, r, t = structure()
    k = key(s, r)
    for embargo in (0.0, 3.0):
        (tr, te), = make_splits("last_fraction", s, r, t, embargo=embargo)
        in_tr, in_te = np.zeros(len(t), bool), np.zeros(len(t), bool)
        in_tr[tr], in_te[te] = True, True
        for rec in set(k):
            m = k == rec
            assert in_te[m].any()
            if in_tr[m].any():
                assert t[m & in_te].min() - t[m & in_tr].max() > embargo
            share = in_te[m].sum() / m.sum()
            assert 0.15 < share < 0.35
        assert list(tr) == list(expected_purged_train(in_te, k, t, embargo))


def test_blocked_in_time_tests_each_sample_once_in_contiguous_blocks():
    s, r, t = structure()
    k = key(s, r)
    folds = make_splits("blocked_in_time", s, r, t, n_blocks=5)
    tested = np.concatenate([te for _, te in folds])
    assert sorted(tested) == list(range(len(t)))
    for _, te in folds:
        in_te = np.zeros(len(t), bool)
        in_te[te] = True
        for rec in set(k):
            idx = np.flatnonzero(k == rec)
            ranks = np.flatnonzero(in_te[idx[np.argsort(t[idx])]])
            assert len(ranks) and ranks.max() - ranks.min() + 1 == len(ranks)


def test_deterministic_protocols_ignore_sample_order():
    def members(shuffle_seed, name, params):
        s, r, t = structure(shuffle_seed=shuffle_seed)
        out = []
        for tr, te in make_splits(name, s, r, t, **params):
            out.append((frozenset(zip(s[tr], r[tr], t[tr])), frozenset(zip(s[te], r[te], t[te]))))
        return set(out)

    for name, params in CASES:
        if name in ("random_slices", "purged_random"):
            continue
        assert members(1, name, params) == members(2, name, params), name


class SpyScaler(BaseEstimator, TransformerMixin):
    seen = []

    def fit(self, X, y=None):
        SpyScaler.seen.append(len(X))
        self.mean_ = X.mean(0)
        return self

    def transform(self, X):
        return X - self.mean_


def test_preprocessing_is_fitted_on_training_rows_only():
    d = make_multisubject(n_subjects=4, n_recordings=2, n_samples=30, seed=3)
    for name in ("random_slices", "blocked_in_time", "held_out_subject"):
        SpyScaler.seen = []
        evaluate_protocol(d.X, d.y, d.subject, d.recording, d.t,
                          lambda: Pipeline([("spy", SpyScaler()), ("m", Ridge())]), name, "pearson_r", n_boot=10)
        assert SpyScaler.seen == [len(tr) for tr, _ in make_splits(name, d.subject, d.recording, d.t)], name


def short_recordings(seed=0, n_sub=24, n_rec=2, n=8):
    subject = np.repeat([f"S{i:02d}" for i in range(n_sub)], n_rec * n)
    recording = np.tile(np.repeat([f"R{j}" for j in range(n_rec)], n), n_sub)
    t = np.tile(np.arange(n), n_sub * n_rec)
    return subject, recording, t


def test_a_no_skill_classifier_shows_no_gap():
    """Labels are coin flips (70% positive) unrelated to anything. A constant guesser has
    balanced accuracy 0.5 by definition, so every protocol must report about 0.5."""
    values = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        s, r, t = short_recordings()
        X = rng.normal(size=(len(t), 4))
        y = (rng.random(len(t)) < 0.7).astype(int)
        gap = validation_gap(X, y, s, r, t, lambda: DummyClassifier(strategy="most_frequent"),
                             protocols=["random_slices", "held_out_subject"], metric="balanced_accuracy", n_boot=10)
        values.append((gap["protocols"]["random_slices"]["score"], gap["protocols"]["held_out_subject"]["score"]))
    print("no-skill balanced accuracy (random_slices, held_out_subject) per seed:", np.round(values, 3).tolist())
    rs, hs = np.mean(values, 0)
    assert abs(rs - 0.5) < 0.03 and abs(hs - 0.5) < 0.03, (rs, hs)


def test_no_time_structure_and_one_shared_mapping_shows_no_gap_in_r2():
    """White-noise target, one mapping shared by everyone, no drift: nothing separates a random
    slice from a new person, so R2 must agree across protocols even when recordings are short."""
    values = []
    for seed in range(5):
        d = make_multisubject(n_subjects=24, n_recordings=2, n_samples=8, target_rho=0.0,
                              shared_weight=1.0, signal=0.15, seed=seed)
        gap = validation_gap(d.X, d.y, d.subject, d.recording, d.t, lambda: Ridge(alpha=1.0),
                             protocols=["random_slices", "held_out_subject"], metric="r2", n_boot=10)
        values.append((gap["protocols"]["random_slices"]["score"], gap["protocols"]["held_out_subject"]["score"]))
    print("no-structure R2 (random_slices, held_out_subject) per seed:", np.round(values, 3).tolist())
    rs, hs = np.mean(values, 0)
    assert abs(rs - hs) < 0.05, (rs, hs)
