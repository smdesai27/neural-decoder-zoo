# Neural Decoder Zoo

**From linear baselines to sequence models for brain-to-behavior decoding.**

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![tests](https://img.shields.io/badge/tests-pytest-informational)

A compact, reproducible benchmark of neural decoders — from a ridge/Wiener filter and a Kalman filter to MLP / TCN / GRU sequence models — evaluated the way a brain–computer interface actually cares about: held-out **temporal** generalization, **inference latency**, **robustness to channel loss**, and **signal validity**.

It runs end-to-end on a synthetic motor-decoding dataset out of the box (no downloads), and plugs into the public **Neural Latents Benchmark (MC_Maze)** for real intracortical data.

> **Thesis — signal validity before model complexity.** This project is the shareable companion to private lab work on intracranial stimulation data, where I learned that the hard part of neural decoding is not the model — it is making sure the thing you're decoding is signal, not artifact. So this repo benchmarks decoders *and* ships a robustness/signal-validity module that shows how a magnitude-based "responsiveness" metric is fooled by injected artifact while a behavior-tied metric is not.

---

## Quickstart

```bash
pip install -e ".[dev]"          # core: numpy, scikit-learn, matplotlib, pytest
python -m ndz.run --dataset synthetic            # run the zoo, print a table, save figures
python -m ndz.run --dataset synthetic --robustness   # + channel-dropout + signal-validity demo
pytest -q                        # tests (no torch required)

# full model zoo (adds MLP / GRU / TCN):
pip install -e ".[dev,torch]"
python -m ndz.run --dataset synthetic
```

## Example results

Two datasets, identical code, 25% held-out **temporal** tail. Metrics: **R²** + **corr** (regression) and per-sample **inference latency** (µs).

**Synthetic reach** (cosine-tuned population, no download):

| decoder | R² | corr | latency (µs) |
|---|---|---|---|
| `ridge_wiener` | **0.95** | 0.97 | 124 |
| `tcn` | 0.94 | 0.97 | 2817 |
| `gru` | 0.92 | 0.96 | 363 |
| `mlp` | 0.91 | 0.95 | 81 |
| `kalman` | 0.80 | 0.97 | 23039 |

**MC_Maze** (real macaque M1/PMd → hand velocity; NLB, DANDI:000140):

| decoder | R² | corr | latency (µs) |
|---|---|---|---|
| `tcn` | **0.51** | 0.76 | 2728 |
| `gru` | 0.47 | 0.70 | 483 |
| `mlp` | 0.25 | 0.65 | 128 |
| `ridge_wiener` | 0.23 | 0.48 | 93 |
| `kalman` | −0.02 | 0.38 | 21967 |

**The contrast is the point.** On the synthetic data a well-regularized **linear** baseline wins; on real **MC_Maze** the **sequence models roughly double** the linear R² (0.51 vs 0.23). Fancy models earn their keep only when the data's temporal structure rewards them — and reporting *where they don't* (synthetic) is the same signal-validity discipline that makes the win on real data credible.

_Regenerate: `make demo` (synthetic) or `python -m ndz.run --dataset mc_maze --nwb-path <file> --robustness` (after the NLB download below)._

## The decoder lineup

| Decoder | Family | Notes |
|---|---|---|
| `ridge_wiener` | Linear | Ridge-regularized Wiener filter with spike-history lags — the baseline to beat |
| `kalman` | Linear-Gaussian state space | Wu et al.–style kinematic Kalman filter; canonical online BCI decoder |
| `mlp` | Feedforward NN | Windowed spike history → MLP *(torch)* |
| `tcn` | Temporal conv net | Dilated causal convolutions over the population *(torch)* |
| `gru` | Recurrent | GRU over a causal window; captures temporal dynamics *(torch)* |

Every decoder implements the same `fit(X, Y) → predict(X)` interface (`ndz.decoders.base.Decoder`), so adding one is a single file.

## Evaluation methodology

- **Contiguous temporal split — no shuffling.** Neural time series are autocorrelated; a shuffled split leaks the future into training. We train on an earlier block and test on a later one — the honest analogue of *calibrate, then use*.
- **BCI-relevant metrics**, not just accuracy: decode quality (R²/corr/accuracy), **inference latency** (µs/sample), **channel-dropout robustness**, and calibration-data efficiency.

## Robustness & signal validity (the differentiating part)

`python -m ndz.run --robustness` runs two probes:

1. **Channel dropout** — decode quality as a function of % electrodes lost (graceful degradation matters for chronic BCI).
2. **Signal-validity demo** — inject periodic saturating "stim-like" artifact into the held-out features, then compare two ways of judging "is there signal here?":
   - a tempting **magnitude/variance metric** (mean |z|) — which the artifact *inflates*, and
   - a **behavior-tied metric** (R² against true kinematics) — which it does *not*.

   The takeaway, learned the hard way on real stimulation data: **validate against behavior, not signal size.** A decoder or screen that rewards raw response magnitude will happily "discover" artifact.

## Datasets

- **Synthetic** (default) — cosine-tuned Poisson population encoding a smooth 2-D reach velocity. Decodable, deterministic by seed, zero downloads.
- **MC_Maze (Neural Latents Benchmark)** — macaque M1/PMd spikes → hand velocity; public, de-identified animal data on DANDI. The **Small** split (DANDI:000140, ~29 MB) is the quick path; the full set is DANDI:000128 (~0.7 GB).
  ```bash
  pip install -e ".[nlb]"
  # download the ~29 MB MC_Maze_Small train file (behavior+ecephys):
  dandi download "DANDI:000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
  python -m ndz.run --dataset mc_maze --nwb-path sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb
  ```
  Results above were produced with this file (14,683 bins × 142 units, 20 ms bins).

## Repository structure

```
src/ndz/
  data/        synthetic generator + NLB loader + NeuralDataset container
  decoders/    ridge, kalman, mlp, tcn, gru (common Decoder interface)
  metrics.py   R2 / corr / accuracy / inference latency
  robustness.py channel dropout + stim-artifact injection + magnitude metric
  evaluate.py  contiguous temporal split + decoder evaluation
  run.py       CLI: table + figures (+ robustness demo)
tests/         pytest suite (runs without torch)
configs/       default experiment config
```

## Roadmap

- [x] Wire MC_Maze numbers alongside synthetic in the results table
- [ ] Add a latent-dynamics baseline (e.g. reduced-rank / LFADS-lite) above the sequence models
- [ ] Calibration-efficiency curve (R² vs minutes of training data)
- [ ] Closed-loop-style online evaluation harness (causal, fixed-latency)

## Data & privacy

This repository contains **only synthetic data and loaders for public datasets** — no human/patient data, no PHI. `data/`, model checkpoints, and all neural file formats are git-ignored by default.

## License

MIT © 2026 Sanil Desai
