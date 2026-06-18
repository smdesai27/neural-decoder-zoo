"""CLI: run the decoder zoo on a dataset, print a results table, save figures.

Examples
--------
    python -m ndz.run --dataset synthetic                 # full zoo (torch if present)
    python -m ndz.run --dataset synthetic --no-torch      # linear + Kalman only
    python -m ndz.run --dataset synthetic --robustness    # + signal-validity demo
    python -m ndz.run --dataset mc_maze --nwb-path data/mc_maze.nwb
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_dataset
from .decoders import HAS_TORCH, build_decoders
from .decoders.linear import RidgeDecoder
from .evaluate import evaluate_decoder, temporal_split
from .metrics import r2_score
from .robustness import channel_dropout, inject_stim_artifact, naive_activity_score


def _print_table(rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(f"{r[c]}") for r in rows)) for c in cols}
    line = "  ".join(c.ljust(widths[c]) for c in cols)
    print(line)
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]}".ljust(widths[c]) for c in cols))


def run_zoo(dataset: str, task: str | None, test_frac: float, no_torch: bool,
            out_dir: Path, **load_kwargs) -> list[dict]:
    ds = load_dataset(dataset, **load_kwargs)
    task = task or ds.task
    print(f"\nDataset: {ds}\n")
    decoders = build_decoders(task=task, include_torch=not no_torch)
    rows = [evaluate_decoder(d, ds, test_frac=test_frac) for d in decoders]
    key = "accuracy" if task == "classification" else "r2"
    rows.sort(key=lambda r: r.get(key, 0), reverse=True)
    _print_table(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # figure: score bar + predicted-vs-true for the best regressor
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].bar([r["decoder"] for r in rows], [r[key] for r in rows], color="#3182bd")
    axes[0].set_ylabel(key.upper()); axes[0].set_title(f"{ds.name}: decoder comparison")
    axes[0].tick_params(axis="x", rotation=30)
    if task == "regression":
        best = build_decoders(task=task, include_torch=not no_torch)[
            [d.name for d in build_decoders(task=task, include_torch=not no_torch)].index(rows[0]["decoder"])
        ]
        sp = temporal_split(ds, test_frac)
        best.fit(sp.X_train, sp.Y_train)
        pred = best.predict(sp.X_test)
        n = min(400, len(pred))
        axes[1].plot(sp.Y_test[:n, 0], color="k", lw=1.2, label="true")
        axes[1].plot(pred[:n, 0], color="#e6550d", lw=1.0, alpha=0.8, label=f"{best.name} pred")
        axes[1].set_title("Best decoder: predicted vs true (dim 0)"); axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "results.png", dpi=130); plt.close(fig)
    print(f"\nwrote {out_dir/'results.csv'} and {out_dir/'results.png'}")
    return rows


def run_robustness(dataset: str, test_frac: float, out_dir: Path, **load_kwargs) -> None:
    """Channel-dropout sweep + the signal-validity (artifact) demonstration."""
    ds = load_dataset(dataset, **load_kwargs)
    sp = temporal_split(ds, test_frac)

    # 1) channel dropout
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75]
    r2s = []
    for fr in fracs:
        dec = RidgeDecoder().fit(channel_dropout(sp.X_train, fr, seed=1), sp.Y_train)
        r2s.append(r2_score(sp.Y_test, dec.predict(channel_dropout(sp.X_test, fr, seed=1))))

    # 2) signal-validity demo: a magnitude metric is fooled by artifact; R^2 is not
    dec = RidgeDecoder().fit(sp.X_train, sp.Y_train)
    X_art = inject_stim_artifact(sp.X_test, event_rate_hz=1.0, bin_size_s=ds.bin_size_s)
    clean = {"activity": naive_activity_score(sp.X_test), "r2": r2_score(sp.Y_test, dec.predict(sp.X_test))}
    art = {"activity": naive_activity_score(X_art), "r2": r2_score(sp.Y_test, dec.predict(X_art))}

    print("\n--- Robustness ---")
    print("channel dropout  R2:", {f: round(r, 3) for f, r in zip(fracs, r2s)})
    print(f"naive activity score   clean={clean['activity']:.2f}  +artifact={art['activity']:.2f}  "
          f"(x{art['activity']/clean['activity']:.1f})")
    print(f"behavior-tied R2       clean={clean['r2']:.3f}  +artifact={art['r2']:.3f}")
    print("Lesson: artifact inflates the magnitude metric but not behavior-tied R2 — "
          "validate against behavior, not signal size.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot([f * 100 for f in fracs], r2s, "o-", color="#3182bd")
    ax[0].set_xlabel("% channels dropped"); ax[0].set_ylabel("R2"); ax[0].set_title("Robustness to channel loss")
    labels = ["activity (clean)", "activity (+artifact)", "R2 (clean)", "R2 (+artifact)"]
    vals = [clean["activity"], art["activity"], clean["r2"], art["r2"]]
    ax[1].bar(labels, vals, color=["#9ecae1", "#cc4c4c", "#a1d99b", "#cc4c4c"])
    ax[1].set_title("Signal-validity: artifact fools magnitude, not behavior R2")
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out_dir / "robustness.png", dpi=130); plt.close(fig)
    print(f"wrote {out_dir/'robustness.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--task", default=None, choices=[None, "regression", "classification"])
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--no-torch", action="store_true", help="linear + Kalman only")
    ap.add_argument("--robustness", action="store_true", help="run robustness + signal-validity demo")
    ap.add_argument("--nwb-path", default=None, help="path to a downloaded NWB file (real datasets)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    if not HAS_TORCH and not args.no_torch:
        print("[note] PyTorch not installed — running linear + Kalman only "
              "(install with `pip install -e \".[torch]\"` for the full zoo).")

    load_kwargs = {"nwb_path": args.nwb_path} if args.nwb_path else {}
    run_zoo(args.dataset, args.task, args.test_frac, args.no_torch, args.out, **load_kwargs)
    if args.robustness:
        run_robustness(args.dataset, args.test_frac, args.out, **load_kwargs)


if __name__ == "__main__":
    main()
