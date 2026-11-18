"""No-download demonstration of the validation gap.

    python -m ndz.validation.demo

Runs the three synthetic regimes with a ridge decoder, prints a table, and
writes ``docs/validation_gap_synthetic.png``. When real case studies
(``case_studies/<name>/results.json``, excluding ``_``-prefixed fixtures) are
present, it also writes ``docs/validation_gap.png``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from sklearn.linear_model import Ridge

from .case_studies import find_results, scan_case_studies, to_rows
from .gap import to_results_json, validation_gap
from .plot import plot_validation_gap
from .splits import PROTOCOL_ORDER
from .synthetic import REGIME_TITLES, REGIMES, make_multisubject

METRIC = "pearson_r"


def make_ridge():
    return Ridge(alpha=1.0)


def run_regime(name: str, seed: int = 0, n_boot: int = 2000) -> dict:
    """``validation_gap`` for one synthetic regime, shaped as a results.json object."""
    data = make_multisubject(seed=seed, **REGIMES[name])
    gap = validation_gap(data.X, data.y, data.subject, data.recording, data.t, make_ridge,
                         metric=METRIC, n_boot=n_boot, seed=seed)
    p = data.params
    return to_results_json(
        gap, dataset=f"synthetic_{name}", title=REGIME_TITLES[name],
        source={"name": "ndz.validation.synthetic (generated, no real data)", "url": "", "license": "MIT"},
        task="regression", n_subjects=p["n_subjects"], unit="", pipeline="StandardScaler + Ridge(alpha=1)",
        baseline={"name": "chance (r = 0)", "value": 0.0}, generated=_dt.date.today().isoformat(),
        script="python -m ndz.validation.demo",
        notes=[f"SYNTHETIC. params: {p}", f"optimism (random_slices - held_out_subject) = {gap['optimism']:.3f}"],
    )


def format_table(results: dict[str, dict]) -> str:
    names = list(results)
    head = f"{'protocol':<20}" + "".join(f"{REGIME_TITLES[n][:26]:>28}" for n in names)
    lines = [head, "-" * len(head)]
    for proto in PROTOCOL_ORDER:
        cells = []
        for n in names:
            p = next((q for q in results[n]["protocols"] if q["name"] == proto), None)
            cells.append(f"{p['score']:.2f} [{p['ci_low']:.2f}, {p['ci_high']:.2f}]" if p else "n/a")
        lines.append(f"{proto:<20}" + "".join(f"{c:>28}" for c in cells))
    opt = []
    for n in names:
        s = {q["name"]: q["score"] for q in results[n]["protocols"]}
        opt.append(f"{s['random_slices'] - s['held_out_subject']:+.2f}")
    lines.append("-" * len(head))
    lines.append(f"{'optimism':<20}" + "".join(f"{o:>28}" for o in opt))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="No-download demonstration of the validation gap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--docs", default="docs", help="where to write the figures")
    ap.add_argument("--case-studies", default="case_studies")
    args = ap.parse_args(argv)

    results = {name: run_regime(name, seed=args.seed) for name in REGIMES}
    print(f"Synthetic validation gap ({METRIC}, ridge decoder, seed {args.seed}; "
          "score [95% CI, bootstrap over subjects])\n")
    print(format_table(results))
    rows = [r for obj in results.values() for r in to_rows(obj)]
    out = plot_validation_gap(rows, Path(args.docs) / "validation_gap_synthetic.png",
                              title="Synthetic: the same decoder under five validation protocols", sharey=True)
    print(f"\nwrote {out}")

    if find_results(args.case_studies, include_private=False):
        case_rows, problems = scan_case_studies(args.case_studies, include_private=False)
        for path, errs in problems.items():
            print(f"skipped {path} (does not match the schema): " + "; ".join(errs))
        if case_rows:
            out = plot_validation_gap(case_rows, Path(args.docs) / "validation_gap.png",
                                      title="Real data: one decoder per dataset under five validation protocols")
            print(f"wrote {out} ({len({r['dataset'] for r in case_rows})} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
