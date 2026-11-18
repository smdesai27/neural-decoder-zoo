"""Load and validate case-study ``results.json`` files (schema v1, see case_studies/README.md).

Each file is the scores for one dataset under the five validation protocols.
Raw recordings are never stored here, only scores and the scripts that made them.

The loader is tolerant where the schema allows it: a protocol may be missing,
``variants`` may be absent, and the CI fields may be null. Anything else that
does not match is reported with its path in the file.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

from .splits import PROTOCOL_ORDER

SCHEMA_VERSION = 1
TASKS = ("classification", "regression")
METRICS = ("accuracy", "balanced_accuracy", "pearson_r")
KNOWN_DATASETS = ("mer_stn", "grip_force_ecog", "meg_muscle_drive")  # any other string is allowed too
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SchemaError(ValueError):
    """A results.json does not match the case-study schema."""

    def __init__(self, path: str, problems: list[str]):
        self.path, self.problems = path, problems
        super().__init__(f"{path}: " + "; ".join(problems))


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_protocols(protocols: Any, where: str, errs: list[str]) -> None:
    if not isinstance(protocols, list):
        errs.append(f"{where}: must be a list")
        return
    seen = set()
    for i, p in enumerate(protocols):
        w = f"{where}[{i}]"
        if not isinstance(p, dict):
            errs.append(f"{w}: must be an object")
            continue
        name = p.get("name")
        if name not in PROTOCOL_ORDER:
            errs.append(f"{w}.name: {name!r} is not one of {list(PROTOCOL_ORDER)}")
        elif name in seen:
            errs.append(f"{w}.name: duplicate protocol {name!r}")
        seen.add(name)
        if not isinstance(p.get("description"), str):
            errs.append(f"{w}.description: must be a string")
        if not _num(p.get("score")):
            errs.append(f"{w}.score: must be a finite number")
        for k in ("ci_low", "ci_high"):
            if p.get(k) is not None and not _num(p.get(k)):
                errs.append(f"{w}.{k}: must be a number or null")
        if _num(p.get("ci_low")) and _num(p.get("ci_high")) and p["ci_low"] > p["ci_high"]:
            errs.append(f"{w}: ci_low > ci_high")
        if p.get("ci_method") is not None and not isinstance(p.get("ci_method"), str):
            errs.append(f"{w}.ci_method: must be a string or null")
        ps = p.get("per_subject")
        if not isinstance(ps, dict) or not all(isinstance(k, str) and (_num(v) or v is None)
                                               for k, v in ps.items()):
            errs.append(f"{w}.per_subject: must map subject id -> number")
        n = p.get("n_test_samples")
        if not (isinstance(n, int) and not isinstance(n, bool) and n >= 0):
            errs.append(f"{w}.n_test_samples: must be a non-negative integer")


def validate_results(obj: Any) -> list[str]:
    """Problems with a parsed results.json; empty when it matches schema v1."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["top level: must be a JSON object"]
    if obj.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version: expected {SCHEMA_VERSION}, got {obj.get('schema_version')!r}")
    for k in ("dataset", "title", "pipeline", "script"):
        if not isinstance(obj.get(k), str) or not obj.get(k):
            errs.append(f"{k}: must be a non-empty string")
    if not isinstance(obj.get("unit"), str):
        errs.append("unit: must be a string (empty for unitless metrics)")
    src = obj.get("source")
    if not isinstance(src, dict) or not all(isinstance(src.get(k), str) for k in ("name", "url", "license")):
        errs.append("source: must be an object with string name, url and license")
    if obj.get("task") not in TASKS:
        errs.append(f"task: {obj.get('task')!r} is not one of {list(TASKS)}")
    if obj.get("metric") not in METRICS:
        errs.append(f"metric: {obj.get('metric')!r} is not one of {list(METRICS)}")
    if obj.get("higher_is_better") is not True:
        errs.append("higher_is_better: must be true")
    n = obj.get("n_subjects")
    if not (isinstance(n, int) and not isinstance(n, bool) and n >= 1):
        errs.append("n_subjects: must be a positive integer")
    base = obj.get("baseline")
    if not isinstance(base, dict) or not isinstance(base.get("name"), str) or not _num(base.get("value")):
        errs.append("baseline: must be an object with string name and numeric value")
    claim = obj.get("published_claim")
    if claim is not None:
        if not isinstance(claim, dict):
            errs.append("published_claim: must be an object or null")
        else:
            for k in ("text", "citation"):
                if not isinstance(claim.get(k), str):
                    errs.append(f"published_claim.{k}: must be a string")
            for k in ("low", "high"):
                if not _num(claim.get(k)):
                    errs.append(f"published_claim.{k}: must be a number")
            if _num(claim.get("low")) and _num(claim.get("high")) and claim["low"] > claim["high"]:
                errs.append("published_claim: low > high")
    _check_protocols(obj.get("protocols"), "protocols", errs)
    variants = obj.get("variants")
    if variants is not None:
        if not isinstance(variants, list):
            errs.append("variants: must be a list when present")
        else:
            for i, v in enumerate(variants):
                if not isinstance(v, dict) or not isinstance(v.get("name"), str) \
                        or not isinstance(v.get("pipeline"), str):
                    errs.append(f"variants[{i}]: must be an object with string name and pipeline")
                    continue
                _check_protocols(v.get("protocols"), f"variants[{i}].protocols", errs)
    notes = obj.get("notes")
    if not isinstance(notes, list) or not all(isinstance(x, str) for x in notes):
        errs.append("notes: must be a list of strings")
    if not isinstance(obj.get("generated"), str) or not _DATE.match(obj.get("generated", "")):
        errs.append("generated: must be a YYYY-MM-DD string")
    return errs


def load_results(path: str | Path) -> dict:
    """Parse and validate one results.json; raises ``SchemaError`` listing every problem."""
    path = Path(path)
    try:
        obj = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SchemaError(str(path), [f"cannot read JSON: {exc}"]) from None
    errs = validate_results(obj)
    if errs:
        raise SchemaError(str(path), errs)
    return obj


def to_rows(obj: dict, source_path: str = "") -> list[dict]:
    """Tidy rows: one per (variant, protocol). The main pipeline has ``variant=None``."""
    claim = obj.get("published_claim") or {}
    common = {
        "dataset": obj["dataset"], "title": obj["title"], "task": obj["task"], "metric": obj["metric"],
        "higher_is_better": obj["higher_is_better"], "unit": obj["unit"], "n_subjects": obj["n_subjects"],
        "baseline_name": obj["baseline"]["name"], "baseline_value": float(obj["baseline"]["value"]),
        "claim_text": claim.get("text"), "claim_low": claim.get("low"), "claim_high": claim.get("high"),
        "claim_citation": claim.get("citation"), "path": source_path,
    }
    rows = []
    groups = [(None, obj["pipeline"], obj["protocols"])]
    groups += [(v["name"], v["pipeline"], v["protocols"]) for v in obj.get("variants") or []]
    for variant, pipeline, protocols in groups:
        for p in protocols:
            rows.append({**common, "variant": variant, "pipeline": pipeline, "protocol": p["name"],
                         "order": PROTOCOL_ORDER.index(p["name"]), "description": p["description"],
                         "score": float(p["score"]), "ci_low": p.get("ci_low"), "ci_high": p.get("ci_high"),
                         "ci_method": p.get("ci_method"),
                         "per_subject": {k: v for k, v in p["per_subject"].items() if v is not None},
                         "n_test_samples": p["n_test_samples"]})
    return rows


def find_results(directory: str | Path = "case_studies", include_private: bool = True) -> list[Path]:
    """``results.json`` files in ``directory`` and its immediate subdirectories, sorted.
    ``include_private=False`` skips subdirectories starting with ``_`` (e.g. ``_fixture``)."""
    directory = Path(directory)
    paths = sorted({*directory.glob("results.json"), *directory.glob("*/results.json")})
    if not include_private:
        paths = [p for p in paths if not p.parent.name.startswith("_") or p.parent == directory]
    return paths


def scan_case_studies(directory: str | Path = "case_studies",
                      include_private: bool = True) -> tuple[list[dict], dict[str, list[str]]]:
    """Rows from every valid results.json, plus ``{path: problems}`` for the files that were skipped."""
    rows, problems = [], {}
    for path in find_results(directory, include_private):
        try:
            rows.extend(to_rows(load_results(path), str(path)))
        except SchemaError as exc:
            problems[str(path)] = exc.problems
    return rows, problems


def load_case_studies(directory: str | Path = "case_studies", include_private: bool = True) -> list[dict]:
    """Tidy rows from every results.json in ``directory`` that matches the schema; files that do
    not match are skipped with a warning naming each problem."""
    rows, problems = scan_case_studies(directory, include_private)
    for path, errs in problems.items():
        warnings.warn(f"skipping {path}: " + "; ".join(errs), stacklevel=2)
    return rows
