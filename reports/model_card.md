# Model Card: VR Fear Response Prediction

## Purpose

An **educational, exploratory** model that predicts a participant's self-reported fear
(SUDS, 0-10) during a specific VR spider-exposure stage from physiological features and/or
contextual information about that stage. Built as a student portfolio project combining
VR, psychophysiology, and classical machine learning -- **not** a clinical or
production system.

## Dataset

Krumnikl et al. (2026), "Do Virtual Spiders Feel Real? Dissociation Between Subjective and
Physiological Fear During Virtual Reality Exposure" (Zenodo, DOI
10.5281/zenodo.20796399, CC-BY-4.0). 28 participants x 10 VR exposure levels (0 =
baseline, 1-9 = escalating spider tasks) = 280 observations. See `data/README.md` for
full dataset documentation.

## Target

`suds`: self-reported Subjective Units of Distress (0-10), collected once per exposure
level. Continuous regression target, genuinely present in the source data (not derived
from the VR condition or from physiology).

## Features

Three feature sets are compared (see README.md for the full column lists):

- **context_only**: exposure level (0-9) + participant trait/demographic questionnaire
  scores (SPQ, STAI trait/pre-exposure state anxiety, age, gender, prior VR/AR experience,
  spider fear/contact history). No physiological measurement.
- **physiology_only**: precomputed per-level physiological features (heart rate, HRV
  [SDNN, RMSSD, LF/HF], respiratory rate, EDA, smartwatch-derived reactivity/trend
  statistics) plus each feature's change from the participant's own level-0 baseline.
- **combined**: union of the above.

Explicitly excluded from all feature sets (see `data/README.md` "Excluded observations"
and `vr_fear_prediction.data.LEAKAGE_EXCLUDED_COLUMNS`): post-exposure questionnaire
scores, cortisol (misaligned sampling times), sensor cross-agreement QA metrics, free-text
post-session reflections, and any `suds_*` column other than the current level's target.

## Validation design

Participant-aware throughout (`vr_fear_prediction.splitting`): a `GroupShuffleSplit`
holds out ~20% of participants entirely for a final test; hyperparameters and model choice
are selected via `GroupKFold` cross-validation *on the training participants only*, so no
participant ever appears in both a training and an evaluation fold. All preprocessing
(imputation, scaling, one-hot encoding) is fit inside a scikit-learn `Pipeline`, so it only
ever sees training data. Tested explicitly in `tests/test_splitting.py`.

**Limitation on validation rigor:** with only 28 participants, hyperparameter selection
and the reported CV performance estimate come from the same `GroupKFold` folds (not a
fully nested CV), which is somewhat optimistic. A fully nested CV was judged impractical
at this sample size (each outer fold would leave very few participants for inner tuning)
and is noted here rather than hidden.

## Metrics (see `reports/metrics.json` for full precision and per-fold detail)

| Feature set | Best model | CV MAE (mean ± sd) | Held-out test MAE | Held-out test R² |
|---|---|---|---|---|
| context_only | Ridge | 1.82 ± 0.12 | 1.31 | 0.19 |
| physiology_only | Dummy (mean) | 2.37 ± 0.51 | 2.56 | -0.86 |
| combined | Ridge | 2.09 ± 0.20 | **0.89** | **0.57** |

Dummy (mean) baseline CV MAE: 2.37 ± 0.51 (constant across feature sets, as expected).

**Does physiology add value beyond context? Verdict: inconclusive.** On the held-out test
participants, combined (MAE 0.89) outperforms context_only (MAE 1.31). On
participant-aware CV, that ranking flips: context_only (1.82) beats combined (2.09). A
single six-participant test set is not automatically more informative than a five-fold
cross-validation estimate, so this project does not treat either number alone as decisive.

To resolve which criterion to trust, we ran the same context-only-vs-combined Ridge
comparison across **50 repeated outer `GroupShuffleSplit`s** (`vr_fear_prediction.
robustness.repeated_participant_holdout`), each holding out a different set of
participants with alpha re-selected each time from the outer-training participants only.
Result: combined had a lower MAE in only **54% of 50 splits**, and the 95% interval on the
paired MAE difference (`context_only_MAE - combined_MAE`) is **[-0.32, +1.22]** --
straddling zero. A sensitivity check restricted to exposure levels 1-9 (excluding the
level-0 floor effect) gives the same qualitative picture (46% -> now 42% combined-better,
still straddling zero; see `reports/metrics.json` -> `sensitivity_excluding_baseline`).

> Evidence that physiology adds predictive value beyond context is inconclusive. The
> combined model performs better on one held-out participant split, but worse under
> participant-aware cross-validation. With only 28 participants, this difference is
> unstable and may depend strongly on which participants are held out.

**Physiology alone is weak**: no model beats the dummy baseline on physiology_only in CV,
and only modestly on the held-out test. This mirrors the source paper's central finding --
subjective and physiological fear are related but substantially dissociated.

## Intended educational use

Demonstrating an end-to-end, leakage-safe ML workflow on a small repeated-measures
psychophysiology dataset: participant-aware validation, honest feature-set comparison,
interpretation (permutation importance, standardized coefficients), and transparent
reporting of a small-sample result that does not look like a polished commercial demo.

## Out-of-scope uses

- **Not** a phobia diagnosis tool. No clinical diagnosis labels exist in this dataset;
  `spider_fear` is a self-reported categorical screening item, not a diagnosis.
- **Not** a replacement for self-report. The model predicts SUDS from physiology/context;
  it does not validate or override what a participant says they feel.
- **Not** validated for real-time / closed-loop control of VR exposure intensity. It could
  inform *research* into adaptive systems (see README.md "Future integration"), but has not
  been evaluated for that use.
- **Not** applicable to other phobias, other VR content, other sensor hardware, or
  populations outside this specific sample (mostly students, one VR headset + PPG/EDA/RESP
  sensor rig) without new validation.

## Limitations

- **N = 28 participants.** Every metric here has wide uncertainty; point estimates should
  not be treated as precise.
- **Whether physiology adds value beyond context is inconclusive on this dataset**, not
  merely uncertain due to a single unlucky split -- a 50-split repeated holdout evaluation
  confirms the comparison is close to a coin flip (54% combined-better) with a paired-MAE
  95% interval straddling zero, and this does not change materially when the baseline
  floor-effect level is excluded (see README "Results").
- **The repeated-holdout evaluation resamples the same 28 participants**; it quantifies
  sensitivity to which participants are held out, but is not a substitute for testing on
  an independent, larger sample.
- **Floor effect at baseline.** Level 0 SUDS is 0 for most participants, which is easy to
  predict and likely inflates apparent performance somewhat -- tested directly via the
  levels-1-9-only sensitivity analysis, which did not change the qualitative conclusion.
- **From-scratch raw-signal processing (notebook demo) does not closely match the
  official precomputed features** in absolute terms, particularly for short-window RMSSD.
  This is disclosed, not hidden, in the notebook, and is why the main pipeline uses the
  dataset authors' precomputed features rather than the from-scratch demo pipeline.
- **No causal claims.** Feature importance identifies association, not causation.
- **Physiological arousal is not unique to fear.** Heart rate and EDA respond to many
  states (effort, excitement, general VR novelty, sensor motion artifacts) besides fear;
  a rise in these signals is not proof that a participant is afraid.
- **Individual physiological baselines vary substantially** (see the notebook's
  within- vs. between-participant variance analysis), which is why baseline-corrected
  delta features are included alongside raw levels, but between-participant differences
  remain a substantial share of the variance in every physiological feature.

## Ethical considerations

This project explores whether physiological signals contain *useful information about*
self-reported fear -- it does not claim to *detect* fear from the body, and physiological
arousal should never be treated as ground truth that overrides what a person reports about
their own internal state. Any future system built on this kind of model that adjusts a
person's exposure therapy intensity based on physiology alone, without their own report or
clinical oversight, would be a misuse of results like these.

## Generalization limitations

Results are specific to: this VR spider-exposure protocol, this sensor hardware (PPG/EDA/
respiration + one consumer smartwatch), this population (predominantly university
students), and Czech/university-recruited samples. Nothing here should be assumed to
transfer to other phobias, other VR hardware, other age groups, or clinical populations
without independent validation.
