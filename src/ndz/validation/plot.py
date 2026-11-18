"""The validation-gap figure: one panel per dataset, protocols on the x-axis."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .splits import PROTOCOL_ORDER

# Okabe-Ito: distinguishable with the common colour-vision deficiencies.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
PROTOCOL_LABELS = {
    "random_slices": "random\nslices",
    "purged_random": "purged\nrandom",
    "blocked_in_time": "blocked\nin time",
    "held_out_recording": "held-out\nrecording",
    "held_out_subject": "held-out\nsubject",
}
METRIC_LABELS = {"pearson_r": "Pearson r", "accuracy": "accuracy", "balanced_accuracy": "balanced accuracy",
                 "r2": "R²"}


def plot_validation_gap(rows: list[dict], out_path: str | Path, title: str | None = None,
                        max_variants: int = 3, ncols: int | None = None, sharey: bool = False) -> Path:
    """Draw one panel per dataset and save to ``out_path``.

    In each panel, the main pipeline is a point with its CI and per-subject dots behind it.
    Up to ``max_variants`` variants are drawn as offset hollow markers. The dataset baseline
    is a dashed line, and a published claim, when present, is a shaded band. ``sharey`` puts
    every panel on one y-scale (use it when all datasets share a metric).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("no rows to plot")
    datasets: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in rows:
        datasets.setdefault(r["dataset"], []).append(r)
    n = len(datasets)
    ncols = ncols or min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.3 * nrows), squeeze=False, sharey=sharey)
    rng = np.random.default_rng(0)
    xs = np.arange(len(PROTOCOL_ORDER))
    for ax, (dataset, drows) in zip(axes.flat, datasets.items()):
        first = drows[0]
        variants = [v for v in dict.fromkeys(r["variant"] for r in drows) if v is not None][:max_variants]
        if first.get("claim_low") is not None and first.get("claim_high") is not None:
            ax.axhspan(first["claim_low"], first["claim_high"], color="#999999", alpha=0.25, lw=0,
                       label="published claim")
        ax.axhline(first["baseline_value"], ls="--", lw=1.2, color="#555555",
                   label=f"baseline: {first['baseline_name']}"[:60])
        groups = [None] + variants
        width = 0.18 if variants else 0.0
        for gi, variant in enumerate(groups):
            offset = (gi - (len(groups) - 1) / 2) * width
            color = OKABE_ITO[gi % len(OKABE_ITO)]
            pts = sorted((r for r in drows if r["variant"] == variant), key=lambda r: r["order"])
            seen = set()
            pts = [r for r in pts if not (r["protocol"] in seen or seen.add(r["protocol"]))]
            if not pts:
                continue
            px = np.array([r["order"] for r in pts]) + offset
            py = np.array([r["score"] for r in pts])
            if variant is None:
                for x, r in zip(px, pts):
                    sub = np.array(list(r["per_subject"].values()), float)
                    if sub.size:
                        ax.scatter(x + rng.uniform(-0.12, 0.12, sub.size), sub, s=9, color="#888888",
                                   alpha=0.55, lw=0, zorder=2)
            lo = np.array([r["ci_low"] if r["ci_low"] is not None else r["score"] for r in pts])
            hi = np.array([r["ci_high"] if r["ci_high"] is not None else r["score"] for r in pts])
            ax.errorbar(px, py, yerr=[py - lo, hi - py], fmt="none", ecolor=color, elinewidth=1.6,
                        capsize=3, zorder=3)
            ax.plot(px, py, ls="-", lw=1, color=color, alpha=0.5, zorder=3)
            ax.scatter(px, py, s=46, marker=MARKERS[gi % len(MARKERS)], zorder=4,
                       facecolor=color if variant is None else "white", edgecolor=color, linewidth=1.6,
                       label=first["pipeline"][:40] if variant is None else variant[:40])
        ax.set_xticks(xs)
        ax.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOL_ORDER], fontsize=8)
        ax.set_xlim(-0.5, len(PROTOCOL_ORDER) - 0.5)
        ax.set_title(first["title"], fontsize=10)
        metric = METRIC_LABELS.get(first["metric"], first["metric"])
        ax.set_ylabel(f"{metric}" + (f" ({first['unit']})" if first.get("unit") else ""), fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=1)
        if sharey:
            ax.tick_params(labelleft=True)
        ax.tick_params(axis="y", labelsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
