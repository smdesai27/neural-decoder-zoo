# Case studies: real-data validation gaps

Each subdirectory holds one dataset's `results.json`: the score of one decoding
pipeline under the five validation protocols in `ndz.validation.splits`.

**No raw data is stored here.** The clinical and human recordings behind these
scores are not redistributed. Get them from their original source under its
own licence (see `source` in each file). This directory holds only scores,
per-subject scores and the name of the script that produced them.

`_fixture/` is a **synthetic** example used by the tests. It contains no real
data and is skipped when the real-data figure is drawn. Any directory whose
name starts with `_` is skipped the same way. `_fixture/make_fixture.py`
regenerates it, and a test checks that the committed file matches.

Load and plot:

```python
from ndz.validation.case_studies import load_case_studies
from ndz.validation.plot import plot_validation_gap

rows = load_case_studies("case_studies", include_private=False)  # tidy rows; bad files skipped with a warning
plot_validation_gap(rows, "docs/validation_gap.png")
```

`python -m ndz.validation.demo` draws `docs/validation_gap.png` automatically
when real case studies are present.

## Schema (version 1)

One JSON object per file:

| field | type | notes |
|---|---|---|
| `schema_version` | `1` | |
| `dataset` | string | `mer_stn`, `grip_force_ecog`, `meg_muscle_drive`, or any other id |
| `title` | string | panel title |
| `source` | `{name, url, license}` | where the raw data lives |
| `task` | `"classification"` \| `"regression"` | |
| `metric` | `"accuracy"` \| `"balanced_accuracy"` \| `"pearson_r"` | |
| `higher_is_better` | `true` | |
| `n_subjects` | int | |
| `unit` | string | may be empty for unitless metrics |
| `pipeline` | string | the main decoding pipeline |
| `baseline` | `{name, value}` | `value` is a number (drawn as a dashed line) |
| `published_claim` | `{text, low, high, citation}` or `null` | drawn as a band |
| `protocols` | list of protocol results | see below; a protocol may be missing |
| `variants` | optional list of `{name, pipeline, protocols}` | same protocol shape |
| `notes` | list of strings | |
| `generated` | `"YYYY-MM-DD"` | |
| `script` | string | the script that produced the file |

A protocol result:

| field | type | notes |
|---|---|---|
| `name` | one of `random_slices`, `purged_random`, `blocked_in_time`, `held_out_recording`, `held_out_subject` | at most once per list |
| `description` | string | what was held out, in plain words |
| `score` | number | |
| `ci_low`, `ci_high` | number or `null` | |
| `ci_method` | string or `null` | e.g. "percentile bootstrap over subjects" |
| `per_subject` | `{subject_id: score}` | one score per scored subject, pooled over that subject's out-of-fold predictions |
| `n_subjects_scored` | int, optional | number of subjects in `per_subject` |
| `skipped` | `{subject_id: reason}`, optional | subjects left out of the mean: not tested by the protocol (`held_out_recording` tests only subjects with two or more recordings) or score undefined |
| `n_test_samples` | int | rows tested; fewer than all rows when the protocol does not test some subjects |

`ndz.validation.gap.to_results_json` turns a `validation_gap` result into this
shape. `ndz.validation.case_studies.validate_results` lists every problem with
a file.

Parameter sweeps (for example several embargo values) go in separate variants,
because a protocol name may appear only once per list. The embargo value goes
in the variant name, for example `purged_embargo_10s`.
