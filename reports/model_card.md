# Model Card: VR Fear Response Prediction

## Purpose

An exploratory model predicting a participant's self-reported fear (SUDS, 0-10) during a
VR spider-exposure stage from physiological features and/or contextual information about
that stage. Not a clinical or production system.

## Dataset

Krumnikl et al. (2026), "Do Virtual Spiders Feel Real? Dissociation Between Subjective and
Physiological Fear During Virtual Reality Exposure" (Zenodo, DOI 10.5281/zenodo.20796399,
CC-BY-4.0). 28 participants x 10 VR exposure levels (0 = baseline, 1-9 = escalating spider
tasks) = 280 observations. See `data/README.md` for full dataset documentation.

## Target

`suds`: self-reported Subjective Units of Distress (0-10), collected once per exposure
level. Continuous regression target, present in the source data (not derived from the VR
condition or from physiology).

## Features

Three feature sets are compared (see README.md for the full column lists):

- **context_only**: exposure level (0-9) + participant trait/demographic questionnaire
  scores (SPQ, STAI trait/pre-exposure state anxiety, age, gender, prior VR/AR experience,
  spider fear/contact history). No physiological measurement.
- **physiology_only**: precomputed per-level physiological features (heart rate, HRV
  [SDNN, RMSSD, LF/HF], respiratory rate, EDA, smartwatch-derived reactivity/trend
  statistics) plus each feature's change from the participant's own level-0 baseline.
- **combined**: union of the above.

Excluded from all feature sets (see `data/README.md` "Excluded observations" and
`vr_fear_prediction.data.LEAKAGE_EXCLUDED_COLUMNS`): post-exposure questionnaire scores,
cortisol (misaligned sampling times), sensor cross-agreement QA metrics, free-text
post-session reflections, and any `suds_*` column other than the current level's target.

## Validation design

Participant-aware throughout (`vr_fear_prediction.splitting`): a `GroupShuffleSplit` holds
out ~20% of participants entirely for a final test; hyperparameters and model choice are
selected via `GroupKFold` cross-validation *on the training participants only*, so no
participant ever appears in both a training and an evaluation fold. All preprocessing is
fit inside a scikit-learn `Pipeline`, so it only sees training data.

Hyperparameter selection and the reported CV performance estimate come from the same
`GroupKFold` folds (not a fully nested CV), which is somewhat optimistic. Full nesting was
judged impractical at this sample size (each outer fold would leave very few participants
for inner tuning).

## Metrics (see `reports/metrics.json` for full detail)

| Feature set | Best model | CV MAE (mean ± sd) | Held-out test MAE | Held-out test R² |
|---|---|---|---|---|
| context_only | Ridge | 1.82 ± 0.12 | 1.31 | 0.19 |
| physiology_only | Dummy (mean) | 2.37 ± 0.51 | 2.56 | -0.86 |
| combined | Ridge | 2.09 ± 0.20 | 0.89 | 0.57 |

**Does physiology add value beyond context? Inconclusive.** The held-out test favors
combined (0.89 vs. 1.31); participant-aware CV favors context_only (1.82 vs. 2.09). A
repeated evaluation across 50 outer `GroupShuffleSplit`s (`vr_fear_prediction.robustness.
repeated_participant_holdout`, alpha re-selected each time from the outer-training
participants only) put combined ahead in only 54% of splits, with a 95% interval on the
paired MAE difference of [-0.32, +1.22] -- straddling zero. A sensitivity check restricted
to exposure levels 1-9 (excluding the level-0 floor effect) gives the same picture (42%
combined-better; see `reports/metrics.json` -> `sensitivity_excluding_baseline`).

Physiology alone is weak: no model beats the dummy baseline on physiology_only under CV,
and only modestly on the held-out test. This mirrors the source paper's own finding that
subjective and physiological fear are related but substantially dissociated.

## Intended use

Educational: a leakage-safe modeling workflow on a small repeated-measures
psychophysiology dataset, including participant-aware validation and a stability check on
the main comparison.

## Out-of-scope uses

- Not a phobia diagnosis tool. No clinical diagnosis labels exist in this dataset;
  `spider_fear` is a self-reported screening item, not a diagnosis.
- Not a replacement for self-report. The model predicts SUDS from physiology/context; it
  does not validate or override what a participant says they feel.
- Not validated for real-time or closed-loop control of VR exposure intensity.
- Not applicable to other phobias, VR content, sensor hardware, or populations outside
  this sample without new validation.

## Limitations

- N = 28 participants; every metric here has wide uncertainty.
- Whether physiology adds value beyond context is inconclusive on this dataset -- a
  50-split repeated holdout evaluation shows the comparison is close to a coin flip, and
  this does not change when the baseline floor-effect level is excluded.
- The repeated-holdout evaluation resamples the same 28 participants; it quantifies
  sensitivity to which participants are held out, not a substitute for a larger sample.
- Feature importance would identify association, not causation; this model card does not
  otherwise attempt causal claims.
- Physiological arousal is not unique to fear -- effort, excitement, novelty, and sensor
  motion artifacts can all raise heart rate or EDA.
- Individual physiological baselines vary substantially, which is why baseline-corrected
  delta features are included alongside raw levels.

## Ethical considerations

This project explores whether physiological signals contain useful information about
self-reported fear -- it does not detect fear from the body, and physiological arousal
should not be treated as overriding what a person reports about their own internal state.

## Generalization

Results are specific to this VR spider-exposure protocol, this sensor hardware (PPG/EDA/
respiration + one consumer smartwatch), and this population (predominantly university
students). They should not be assumed to transfer to other phobias, hardware, age groups,
or clinical populations without independent validation.
