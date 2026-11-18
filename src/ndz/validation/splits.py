"""Train/test index generators for structured neural data.

Every generator takes per-sample arrays ``subject``, ``recording`` and ``t``
(time order within a recording) and returns a list of ``(train_idx, test_idx)``
integer numpy arrays. Train and test are always disjoint.

Recording ids are scoped by subject: ``("S1", "R0")`` and ``("S2", "R0")`` are
different recordings, so callers may reuse ids such as ``"session1"`` across
subjects.

Protocols, from most optimistic to most honest:

- ``random_slices``: shuffled samples, ignoring all structure (the usual check).
- ``purged_random``: random test samples, with training samples within
  ``embargo`` steps of a test sample in the same recording removed.
- ``blocked_in_time``: contiguous blocks in each recording; test on one block,
  train on the others (``last_fraction``: train on the start, test on the end).
- ``held_out_recording``: leave one recording out; its subject stays in training.
- ``held_out_subject``: leave one subject out.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Split = tuple[np.ndarray, np.ndarray]

PROTOCOL_ORDER = ("random_slices", "purged_random", "blocked_in_time",
                  "held_out_recording", "held_out_subject")


def _arrays(subject, recording, t):
    subject = np.asarray(subject).astype(str)
    recording = np.asarray(recording).astype(str)
    t = np.asarray(t)
    if not (len(subject) == len(recording) == len(t)):
        raise ValueError("subject, recording and t must have the same length")
    # recording ids are scoped by subject
    rec_key = np.char.add(np.char.add(subject, "\x1f"), recording)
    return subject, rec_key, t


def _purge(train_mask: np.ndarray, test_mask: np.ndarray, rec_key: np.ndarray, t: np.ndarray,
           embargo: float) -> np.ndarray:
    """Drop training samples within ``embargo`` time steps of any test sample in the same recording."""
    if embargo <= 0:
        return train_mask
    keep = train_mask.copy()
    for rec in np.unique(rec_key[test_mask]):
        in_rec = rec_key == rec
        test_t = np.sort(t[in_rec & test_mask].astype(float))
        cand = np.flatnonzero(in_rec & keep)
        if cand.size == 0:
            continue
        ct = t[cand].astype(float)
        pos = np.searchsorted(test_t, ct)
        left = np.abs(ct - test_t[np.clip(pos - 1, 0, len(test_t) - 1)])
        right = np.abs(test_t[np.clip(pos, 0, len(test_t) - 1)] - ct)
        keep[cand[np.minimum(left, right) <= embargo]] = False
    return keep


def _finish(train_mask: np.ndarray, test_mask: np.ndarray) -> Split:
    return np.flatnonzero(train_mask & ~test_mask), np.flatnonzero(test_mask)


def random_slices(subject, recording, t, n_splits: int = 5, test_frac: float = 0.2,
                  seed: int = 0) -> list[Split]:
    """Repeated shuffled splits: each fold draws ``test_frac`` of all samples at random."""
    subject, rec_key, t = _arrays(subject, recording, t)
    n = len(t)
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_frac * n)))
    out = []
    for _ in range(n_splits):
        test = np.zeros(n, bool)
        test[rng.choice(n, n_test, replace=False)] = True
        out.append(_finish(~test, test))
    return out


def purged_random(subject, recording, t, n_splits: int = 5, test_frac: float = 0.1,
                  embargo: float = 3, seed: int = 0) -> list[Split]:
    """Random test samples; training samples within ``embargo`` steps of a test sample in the
    same recording are removed (|t_train - t_test| <= embargo)."""
    subject, rec_key, t = _arrays(subject, recording, t)
    out = []
    for train, test in random_slices(subject, recording, t, n_splits, test_frac, seed):
        test_mask = np.zeros(len(t), bool)
        test_mask[test] = True
        train_mask = _purge(~test_mask, test_mask, rec_key, t, embargo)
        out.append(_finish(train_mask, test_mask))
    return out


def _block_of(rec_key: np.ndarray, t: np.ndarray, n_blocks: int) -> np.ndarray:
    """Block number (0..n_blocks-1) of every sample, by time rank within its recording."""
    block = np.empty(len(t), int)
    for rec in np.unique(rec_key):
        idx = np.flatnonzero(rec_key == rec)
        order = idx[np.argsort(t[idx], kind="stable")]
        block[order] = np.arange(len(order)) * n_blocks // len(order)
    return block


def blocked_in_time(subject, recording, t, n_blocks: int = 5, embargo: float = 0) -> list[Split]:
    """Fold k tests on the k-th contiguous block of every recording and trains on the other
    blocks, minus ``embargo`` steps on either side of the test block."""
    subject, rec_key, t = _arrays(subject, recording, t)
    block = _block_of(rec_key, t, n_blocks)
    out = []
    for k in range(n_blocks):
        test = block == k
        out.append(_finish(_purge(~test, test, rec_key, t, embargo), test))
    return out


def last_fraction(subject, recording, t, test_frac: float = 0.25, embargo: float = 0) -> list[Split]:
    """One split: train on the first part of every recording, test on its last ``test_frac``."""
    subject, rec_key, t = _arrays(subject, recording, t)
    test = np.zeros(len(t), bool)
    for rec in np.unique(rec_key):
        idx = np.flatnonzero(rec_key == rec)
        order = idx[np.argsort(t[idx], kind="stable")]
        n_test = max(1, int(round(test_frac * len(order))))
        test[order[len(order) - n_test:]] = True
    return [_finish(_purge(~test, test, rec_key, t, embargo), test)]


def held_out_recording(subject, recording, t) -> list[Split]:
    """Leave one recording out. Only recordings whose subject has another recording are tested,
    so the same subjects appear on both sides."""
    subject, rec_key, t = _arrays(subject, recording, t)
    out = []
    for rec in np.unique(rec_key):
        test = rec_key == rec
        subj = subject[test][0]
        train = ~test
        if not np.any(train & (subject == subj)):
            continue  # the subject's only recording: that would be a held-out subject
        out.append(_finish(train, test))
    return out


def held_out_subject(subject, recording, t) -> list[Split]:
    """Leave one subject out (all of their recordings)."""
    subject, rec_key, t = _arrays(subject, recording, t)
    subjects = np.unique(subject)
    if len(subjects) < 2:
        raise ValueError("held_out_subject needs at least two subjects")
    return [_finish(subject != s, subject == s) for s in subjects]


PROTOCOLS: dict[str, Callable[..., list[Split]]] = {
    "random_slices": random_slices,
    "purged_random": purged_random,
    "blocked_in_time": blocked_in_time,
    "held_out_recording": held_out_recording,
    "held_out_subject": held_out_subject,
    "last_fraction": last_fraction,
}

DESCRIPTIONS = {
    "random_slices": "Shuffled samples; test samples sit next to training samples in time.",
    "purged_random": "Random test samples; training samples within the embargo of a test sample are removed.",
    "blocked_in_time": "Contiguous blocks within each recording; test on one block, train on the rest.",
    "held_out_recording": "Leave one recording out; the same subjects are in training.",
    "held_out_subject": "Leave one subject out; the decoder never saw this person.",
    "last_fraction": "Train on the start of each recording, test on its end.",
}


def make_splits(protocol: str, subject, recording, t, **params) -> list[Split]:
    """Splits for a protocol by name."""
    try:
        fn = PROTOCOLS[protocol]
    except KeyError:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {sorted(PROTOCOLS)}") from None
    return fn(subject, recording, t, **params)
