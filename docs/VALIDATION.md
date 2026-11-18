# How you split the data decides the score

## The question

A decoder's reported score depends on how the test data was separated from the
training data, not just on the decoder. Neural and behavioural signals change
slowly. If test samples are drawn at random from a recording, each one has
near neighbours in time in the training set, and the decoder is partly graded
on data it has effectively already seen. A later stretch of the same recording
is harder. A person the decoder has never seen is harder still, and that is
the case that matters for a clinical device.

This page asks how large that difference is, on synthetic data where the
answer is known and on public human datasets.

## The protocols in plain words

| protocol | what is held out | what it answers |
|---|---|---|
| `random_slices` | random samples from anywhere | the usual check; the most flattering |
| `purged_random` | random samples, and training samples within an embargo of them are removed | how much of the random-slice score comes from immediate neighbours |
| `blocked_in_time` | a contiguous block of each recording (`last_fraction`: its end) | will it keep working later in the same session? |
| `held_out_recording` | a whole recording, with the same people still in training | will it work in this person's next session? |
| `held_out_subject` | every recording of one person | will it work for a new patient? |

The **optimism** is `random_slices − held_out_subject`: how much the usual
check overstates performance for a new person.

Implementation: `ndz.validation.splits` (index generators), `ndz.validation.gap`
(scoring). Scalers and feature selection are fit inside each training fold
through an sklearn `Pipeline`. Scores are averaged per subject first, and the
95% CI is a percentile bootstrap that resamples subjects.

## Synthetic result

Run it with `python -m ndz.validation.demo` (no downloads, under a minute).

The data has 6 subjects × 2 recordings × 300 samples and 48 channels. The
decoder is StandardScaler + Ridge, scored by Pearson r. Each regime changes
one setting from the first (`ndz.validation.synthetic.REGIMES`):

1. **Slow target, subject-specific mapping.** The target is AR(1) with
   ρ = 0.99. 85% of the variance of the target-to-channel mapping is specific to each subject,
   and that part also carries slow, recording-specific channel drift.
2. **White-noise target.** The same as 1, with ρ = 0.
3. **Fully shared mapping.** The same as 1, with every subject using one
   mapping, so there is no subject-specific part and no drift.

Seed 0, score [95% CI]:

| protocol | slow target, subject-specific | white-noise target | fully shared mapping |
|---|---|---|---|
| `random_slices` | 0.73 [0.60, 0.82] | 0.66 [0.60, 0.72] | 0.88 [0.83, 0.91] |
| `purged_random` | 0.69 [0.57, 0.78] | 0.64 [0.58, 0.70] | 0.88 [0.84, 0.90] |
| `blocked_in_time` | 0.49 [0.34, 0.65] | 0.66 [0.62, 0.70] | 0.80 [0.75, 0.85] |
| `held_out_recording` | 0.37 [0.23, 0.46] | 0.50 [0.41, 0.58] | 0.84 [0.81, 0.87] |
| `held_out_subject` | 0.16 [−0.03, 0.35] | 0.30 [0.22, 0.38] | 0.88 [0.84, 0.91] |
| **optimism** | **+0.57** | **+0.37** | **+0.00** |

![synthetic validation gap](validation_gap_synthetic.png)

What the regimes show:

- **A slow target with a subject-specific mapping** is the realistic case. The
  usual check reports r = 0.73, but a new person gets 0.16, and its CI includes
  zero. Purging immediate neighbours removes little of that gap (0.69). Most
  of the inflation comes from sharing the same stretch of slow drift, not from
  adjacent samples. The embargo would need to be on the scale of the drift to
  remove it.
- **A white-noise target** makes random slices and blocked splits agree (0.66
  and 0.66), because time no longer carries information about the target. A
  new person still costs a lot, because the mapping is individual.
- **A fully shared mapping** makes a new person as easy as a random slice
  (0.88 and 0.88).

The tests (`tests/test_validation.py`) check these three orderings on seeds
0–2. Their margins hold on seeds 0–19: random minus blocked is at least 0.16
and blocked minus held-out subject at least 0.05 in regime 1. The white-noise
gap between random and blocked stays at most 0.04, and the shared-mapping
optimism at most 0.004.

## Real-data case studies (pending)

> **Pending.** These results are being produced from three public human
> datasets and will be loaded from `case_studies/<dataset>/results.json` (schema:
> [`case_studies/README.md`](../case_studies/README.md)). No raw data is
> redistributed. Once results are present, `python -m ndz.validation.demo`
> writes `docs/validation_gap.png`.

| dataset | task / metric | n subjects | random slices | purged random | blocked in time | held-out recording | held-out subject | optimism | published claim |
|---|---|---|---|---|---|---|---|---|---|
| `mer_stn` | | | | | | | | | |
| `grip_force_ecog` | | | | | | | | | |
| `meg_muscle_drive` | | | | | | | | | |

## Limitations

- The synthetic generator is built to show the mechanism. The size of the gap
  in real data is an empirical question, which is what the case studies are for.
- The bootstrap resamples only a handful of subjects, so the CIs are wide and
  approximate.
- Only offline decoding is evaluated. A closed-loop user adapts to the decoder,
  which changes what "held out" means.
