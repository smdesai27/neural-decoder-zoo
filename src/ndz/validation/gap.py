"""Score one decoder under several validation protocols and measure the gap between them.

Every fold builds a fresh sklearn ``Pipeline`` and fits it on the training
indices only, so scalers and feature selection never see test samples.

Scoring: each subject is scored **once per protocol**, on all of that
subject's out-of-fold predictions pooled across folds. A subject is never
scored on a handful of rows from one fold, so the score does not depend on
how many of their rows a fold happens to hold out. That size differs between
protocols, so per-fold scoring would itself create a gap. The protocol score
is the mean of the subject scores. Its 95% CI is a percentile bootstrap that
resamples subjects, the unit that generalises.

A subject whose score is undefined is left out of the mean and reported in
``skipped`` with the reason. For example, balanced accuracy is undefined when
the subject's true labels contain one class, and Pearson r or R² when the
target or prediction is constant. Subjects the protocol never tests are
reported the same way (``held_out_recording`` tests only subjects with two or
more recordings).
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .splits import DESCRIPTIONS, PROTOCOL_ORDER, make_splits


class UndefinedScore(ValueError):
    """The metric is undefined for this subject's data; the message says why."""


HIGHER_IS_BETTER = {"r2": True, "pearson_r": True, "accuracy": True, "balanced_accuracy": True}


def _2d(v):
    v = np.asarray(v, float)
    return v.reshape(len(v), -1)


def _r2(y, p):
    y, p = _2d(y), _2d(p)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    if np.any(ss_tot == 0):
        raise UndefinedScore("R² undefined: the true target is constant")
    return float(np.mean(1.0 - ((y - p) ** 2).sum(0) / ss_tot))


def _pearson(y, p):
    y, p = _2d(y), _2d(p)
    if len(y) < 2 or np.any(y.std(0) == 0):
        raise UndefinedScore("Pearson r undefined: the true target is constant")
    if np.any(p.std(0) == 0):
        raise UndefinedScore("Pearson r undefined: the prediction is constant")
    return float(np.mean([np.corrcoef(y[:, d], p[:, d])[0, 1] for d in range(y.shape[1])]))


def _balanced_accuracy(y, p):
    if len(np.unique(y)) < 2:
        raise UndefinedScore("balanced accuracy undefined: the true labels contain one class")
    return float(balanced_accuracy_score(y, p))


METRICS: dict[str, Callable] = {
    "r2": _r2,
    "pearson_r": _pearson,
    "accuracy": lambda y, p: float(accuracy_score(y, p)),
    "balanced_accuracy": _balanced_accuracy,
}


def score_subject(score_fn: Callable, y, p) -> tuple[float | None, str | None]:
    """``(score, None)`` or ``(None, reason)`` when the score is undefined (a custom metric may
    raise ``UndefinedScore`` or return a non-finite value)."""
    try:
        v = score_fn(y, p)
    except UndefinedScore as exc:
        return None, str(exc)
    if v is None or not np.isfinite(v):
        return None, "score undefined (non-finite)"
    return float(v), None


def make_pipeline(make_model: Callable[[], object]) -> Pipeline:
    """A fresh per-fold pipeline: ``make_model()`` itself if it returns a Pipeline, else
    ``StandardScaler -> model``."""
    model = make_model()
    if isinstance(model, Pipeline):
        return clone(model)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _percentile_ci(stats: np.ndarray, level: float) -> tuple[float, float]:
    a = (1.0 - level) / 2.0
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1.0 - a))


def bootstrap_subject_ci(per_subject: Mapping[str, float], n_boot: int = 2000, level: float = 0.95,
                         seed: int = 0) -> tuple[float | None, float | None]:
    """Percentile CI of the mean subject score, resampling subjects with replacement."""
    vals = np.array([v for v in per_subject.values() if v is not None and np.isfinite(v)], float)
    if len(vals) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    return _percentile_ci(vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(1), level)


def paired_difference_ci(a: Mapping[str, float], b: Mapping[str, float], n_boot: int = 2000,
                         level: float = 0.95, seed: int = 0) -> tuple[float | None, float | None, int]:
    """Paired percentile bootstrap of mean(a - b) over the subjects scored in both:
    ``(ci_low, ci_high, n_subjects)``."""
    common = sorted(set(a) & set(b))
    diff = np.array([a[s] - b[s] for s in common], float)
    if len(diff) < 2:
        return None, None, len(diff)
    rng = np.random.default_rng(seed)
    lo, hi = _percentile_ci(diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(1), level)
    return lo, hi, len(diff)


def evaluate_protocol(X, y, subject, recording, t, make_model, protocol: str, metric="r2",
                      n_boot: int = 2000, seed: int = 0, **split_params) -> dict:
    """Run one protocol. Each subject is scored once, on all of their out-of-fold predictions
    pooled across folds; returns score, CI, per-subject scores, skipped subjects and counts."""
    X, y = np.asarray(X), np.asarray(y)
    subject = np.asarray(subject).astype(str)
    score_fn = METRICS[metric] if isinstance(metric, str) else metric
    splits = make_splits(protocol, subject, recording, t, **split_params)
    if not splits:
        raise ValueError(f"protocol {protocol!r} produced no folds for this data")
    test_rows, preds = [], []
    for train, test in splits:
        model = make_pipeline(make_model).fit(X[train], y[train])
        test_rows.append(test)
        preds.append(np.asarray(model.predict(X[test])))
    rows = np.concatenate(test_rows)
    pred = np.concatenate(preds)
    per_subject: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for s in sorted(np.unique(subject)):
        m = subject[rows] == s
        if not m.any():
            skipped[s] = f"not tested by {protocol}"
            continue
        value, reason = score_subject(score_fn, y[rows[m]], pred[m])
        if value is None:
            skipped[s] = reason
        else:
            per_subject[s] = value
    lo, hi = bootstrap_subject_ci(per_subject, n_boot=n_boot, seed=seed)
    return {
        "name": protocol,
        "description": DESCRIPTIONS.get(protocol, ""),
        "score": float(np.mean(list(per_subject.values()))) if per_subject else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
        "ci_method": f"percentile bootstrap over subjects (n={n_boot})" if lo is not None else None,
        "per_subject": per_subject,
        "n_subjects_scored": len(per_subject),
        "skipped": skipped,
        "n_test_samples": int(len(rows)),
        "n_folds": len(splits),
    }


def validation_gap(X, y, subject, recording, t, make_model: Callable[[], object],
                   protocols: Iterable[str] | Mapping[str, dict] | None = None, metric="r2",
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """Score ``make_model`` under each protocol.

    Args:
        make_model: zero-argument factory returning an sklearn estimator (or Pipeline).
        protocols: protocol names, or ``{name: split_params}``; default all five.
        metric: ``"r2" | "pearson_r" | "accuracy" | "balanced_accuracy"`` or a callable
            ``(y_true, y_pred) -> float`` (higher is better).

    Returns ``{"metric", "protocols": {name: result}, "optimism", "optimism_ci",
    "optimism_n_subjects"}``. ``optimism = score(random_slices) - score(held_out_subject)``, and
    ``optimism_ci`` is a paired percentile bootstrap of the per-subject difference over the
    subjects scored under both (None if either protocol is missing).
    """
    if protocols is None:
        protocols = {p: {} for p in PROTOCOL_ORDER}
    elif not isinstance(protocols, Mapping):
        protocols = {p: {} for p in protocols}
    results = {}
    for name, params in protocols.items():
        results[name] = evaluate_protocol(X, y, subject, recording, t, make_model, name, metric,
                                          n_boot=n_boot, seed=seed, **(params or {}))
    optimism, optimism_ci, n_paired = None, None, 0
    if "random_slices" in results and "held_out_subject" in results:
        rs, ho = results["random_slices"], results["held_out_subject"]
        optimism = rs["score"] - ho["score"]
        lo, hi, n_paired = paired_difference_ci(rs["per_subject"], ho["per_subject"], n_boot=n_boot, seed=seed)
        optimism_ci = None if lo is None else (lo, hi)
    return {"metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
            "protocols": results, "optimism": optimism, "optimism_ci": optimism_ci,
            "optimism_n_subjects": n_paired}


def to_results_json(gap: dict, *, dataset: str, title: str, source: dict, task: str, n_subjects: int,
                    unit: str, pipeline: str, baseline: dict, generated: str, script: str,
                    published_claim: dict | None = None, notes: list[str] | None = None,
                    variants: list[dict] | None = None) -> dict:
    """Shape a ``validation_gap`` result as a case-study ``results.json`` object (schema v1)."""
    metric = gap["metric"]
    protocols = [{k: p[k] for k in ("name", "description", "score", "ci_low", "ci_high", "ci_method",
                                    "per_subject", "n_subjects_scored", "skipped", "n_test_samples")}
                 for p in gap["protocols"].values() if p["name"] in PROTOCOL_ORDER]
    out = {
        "schema_version": 1, "dataset": dataset, "title": title, "source": source, "task": task,
        "metric": metric, "higher_is_better": HIGHER_IS_BETTER.get(metric, True),
        "n_subjects": n_subjects, "unit": unit, "pipeline": pipeline, "baseline": baseline,
        "published_claim": published_claim, "protocols": protocols, "notes": list(notes or []),
        "generated": generated, "script": script,
    }
    if variants:
        out["variants"] = variants
    return out
