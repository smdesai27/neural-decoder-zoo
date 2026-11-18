"""Score one decoder under several validation protocols and measure the gap between them.

Every fold builds a fresh sklearn ``Pipeline`` and fits it on the training
indices only, so scalers and feature selection never see test samples.

Scores are aggregated per subject first: a subject's score is the mean over
folds of the metric on that subject's test samples in the fold. The protocol
score is the mean of the subject scores, and its confidence interval comes
from a bootstrap that resamples subjects (the unit that generalises), not
samples.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .splits import DESCRIPTIONS, PROTOCOL_ORDER, make_splits

HIGHER_IS_BETTER = {"r2": True, "pearson_r": True, "accuracy": True, "balanced_accuracy": True}


def _r2(y, p):
    y, p = np.asarray(y, float).reshape(len(y), -1), np.asarray(p, float).reshape(len(p), -1)
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1.0 - ss_res / ss_tot))


def _pearson(y, p):
    y, p = np.asarray(y, float).reshape(len(y), -1), np.asarray(p, float).reshape(len(p), -1)
    rs = []
    for d in range(y.shape[1]):
        if y[:, d].std() == 0 or p[:, d].std() == 0:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(y[:, d], p[:, d])[0, 1]))
    return float(np.mean(rs))


METRICS: dict[str, Callable] = {
    "r2": _r2,
    "pearson_r": _pearson,
    "accuracy": lambda y, p: float(accuracy_score(y, p)),
    "balanced_accuracy": lambda y, p: float(balanced_accuracy_score(y, p)),
}


def make_pipeline(make_model: Callable[[], object]) -> Pipeline:
    """A fresh per-fold pipeline: ``make_model()`` itself if it returns a Pipeline, else
    ``StandardScaler -> model``."""
    model = make_model()
    if isinstance(model, Pipeline):
        return clone(model)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def bootstrap_subject_ci(per_subject: Mapping[str, float], n_boot: int = 2000, level: float = 0.95,
                         seed: int = 0) -> tuple[float | None, float | None]:
    """Percentile CI of the mean subject score, resampling subjects with replacement."""
    vals = np.array([v for v in per_subject.values() if np.isfinite(v)], float)
    if len(vals) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(1)
    a = (1.0 - level) / 2.0
    return float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def evaluate_protocol(X, y, subject, recording, t, make_model, protocol: str, metric="r2",
                      n_boot: int = 2000, seed: int = 0, **split_params) -> dict:
    """Run one protocol; returns score, CI, per-subject scores and sample counts."""
    X, y = np.asarray(X), np.asarray(y)
    subject = np.asarray(subject).astype(str)
    score_fn = METRICS[metric] if isinstance(metric, str) else metric
    splits = make_splits(protocol, subject, recording, t, **split_params)
    if not splits:
        raise ValueError(f"protocol {protocol!r} produced no folds for this data")
    fold_scores: dict[str, list[float]] = {}
    n_test = 0
    for train, test in splits:
        model = make_pipeline(make_model).fit(X[train], y[train])
        pred = model.predict(X[test])
        n_test += len(test)
        for s in np.unique(subject[test]):
            m = subject[test] == s
            if m.sum() < 2:
                continue
            fold_scores.setdefault(s, []).append(score_fn(y[test][m], pred[m]))
    per_subject = {s: float(np.mean(v)) for s, v in sorted(fold_scores.items())}
    lo, hi = bootstrap_subject_ci(per_subject, n_boot=n_boot, seed=seed)
    return {
        "name": protocol,
        "description": DESCRIPTIONS.get(protocol, ""),
        "score": float(np.mean(list(per_subject.values()))),
        "ci_low": lo,
        "ci_high": hi,
        "ci_method": f"percentile bootstrap over subjects (n={n_boot})" if lo is not None else None,
        "per_subject": per_subject,
        "n_test_samples": int(n_test),
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

    Returns ``{"metric", "protocols": {name: result}, "optimism"}`` where
    ``optimism = score(random_slices) - score(held_out_subject)`` (None if either is missing).
    """
    if protocols is None:
        protocols = {p: {} for p in PROTOCOL_ORDER}
    elif not isinstance(protocols, Mapping):
        protocols = {p: {} for p in protocols}
    results = {}
    for name, params in protocols.items():
        results[name] = evaluate_protocol(X, y, subject, recording, t, make_model, name, metric,
                                          n_boot=n_boot, seed=seed, **(params or {}))
    optimism = None
    if "random_slices" in results and "held_out_subject" in results:
        optimism = results["random_slices"]["score"] - results["held_out_subject"]["score"]
    return {"metric": metric if isinstance(metric, str) else getattr(metric, "__name__", "custom"),
            "protocols": results, "optimism": optimism}


def to_results_json(gap: dict, *, dataset: str, title: str, source: dict, task: str, n_subjects: int,
                    unit: str, pipeline: str, baseline: dict, generated: str, script: str,
                    published_claim: dict | None = None, notes: list[str] | None = None,
                    variants: list[dict] | None = None) -> dict:
    """Shape a ``validation_gap`` result as a case-study ``results.json`` object (schema v1)."""
    metric = gap["metric"]
    protocols = [{k: p[k] for k in ("name", "description", "score", "ci_low", "ci_high", "ci_method",
                                    "per_subject", "n_test_samples")}
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
