# Predicting Fear During Virtual Reality Exposure

A small-sample behavioral/physiological machine-learning study: this project explores
whether physiological signals contain useful information about self-reported fear during
VR spider exposure, and how that fear-related information compares with simply knowing
the exposure stage a participant is in.

It is a student portfolio project (Computer Science + Psychology) demonstrating an
end-to-end, leakage-safe ML pipeline on a real, publicly available psychophysiology
dataset -- not a clinical tool and not a production prediction system.

## Motivation

**Psychological:** In VR exposure therapy for phobias, a therapist typically calibrates
exposure intensity using what the client reports. Physiological signals (heart rate, skin
conductance) are sometimes proposed as an additional, more continuous source of
information a system could use alongside self-report. But physiological arousal and
subjective fear are not the same thing -- they can dissociate, as the dataset's own title
suggests ("Do Virtual Spiders Feel Real? Dissociation Between Subjective and Physiological
Fear During VR Exposure"). This project treats that dissociation as a first-class research
question rather than assuming physiology is a proxy for fear.

**Technical:** The project demonstrates a repeated-measures ML workflow done correctly at
small sample size: participant-aware splitting/cross-validation (no participant leakage),
honest use of precomputed vs. raw physiological data, an explicit three-way feature-set
comparison, and transparent reporting of results that don't look like a polished
commercial demo (because they shouldn't, at N=28).

## Research questions

1. **Primary:** Can physiological signals recorded during virtual spider exposure predict
   a participant's self-reported fear level?
2. **Secondary:** Do physiological responses and subjective fear tell the same story, or
   is there a meaningful dissociation between them?

## What this project is **not**

It does not diagnose a phobia, does not replace self-report, does not make treatment
decisions, and is not a clinical system. See "Responsible use" below and
`reports/model_card.md` for the full scope statement.

## Dataset

**Krumnikl, M., Nemec, M., Vilímková, A., Huser, F., & Jaremová, V. (2026).** Dataset for
*"Do Virtual Spiders Feel Real? Dissociation Between Subjective and Physiological Fear
During Virtual Reality Exposure"* [Data set]. Zenodo.
[https://doi.org/10.5281/zenodo.20796399](https://doi.org/10.5281/zenodo.20796399)

- **License:** CC-BY-4.0 (verified via the Zenodo API on 2026-08-20; open access, single
  ~467 MB archive, MD5-checksum-verified on download)
- **Participants:** 28 (pseudonymized ids `P04`-`P31`)
- **Unit of observation:** one (participant, VR exposure level) pair -- **280
  observations** (28 participants x 10 levels: level 0 = neutral baseline, levels 1-9 =
  escalating spider-exposure tasks)
- Full documentation, exact column definitions, sampling rates, exclusions, and licensing
  detail: [`data/README.md`](data/README.md)

## Physiological measurements used

Precomputed per-level features from the dataset authors (used honestly rather than
reprocessed, given the raw recordings are ~9 GB total -- see "Signal processing" below):
heart rate and HRV (SDNN, RMSSD, LF/HF) from PPG, smartwatch heart rate plus derived
reactivity/trend/range statistics, respiratory rate, and mean electrodermal activity (EDA)
-- each also expressed as a **change from the participant's own level-0 baseline**.

## Prediction target

`suds`: self-reported **Subjective Units of Distress** (0-10 continuous scale), collected
once per exposure level. This is a genuine subjective-fear rating present in the source
data -- **not** the VR condition/level relabeled as a proxy, and not derived from
physiology. Modeled as **regression** (the preferred task per the project brief, since the
rating is a genuine continuous self-report, not an artificially binned category).

## Signal processing and feature-extraction decisions

- **Main pipeline:** uses the dataset authors' precomputed per-level physiological
  features (`summary.csv`), plus baseline-corrected deltas computed by this project
  (`vr_fear_prediction.data.reshape_wide_to_long`). This is the honest choice given the
  raw data's size (150 MB-1.2 GB per participant, ~9 GB total) -- reprocessing all of it is
  impractical for a student project or CI.
- **Raw-signal demonstration:** `src/vr_fear_prediction/signal_processing.py` implements a
  real NeuroKit2-based extraction pipeline (EDA tonic/phasic decomposition + SCR
  detection, PPG peak-based heart rate and RMSSD, respiration rate), unit-tested with
  small synthetic fixtures (`tests/test_signal_processing.py`), and demonstrated on one
  real participant's raw recording in `notebooks/01_exploration.ipynb` as a sanity check
  against the official precomputed values. **RMSSD is only computed when a window is at
  least 30 seconds long and contains at least 15 detected beats**; shorter windows return
  `NaN` rather than an unreliable number.
- **Baseline correction:** every physiological feature is also expressed as
  `value_at_level_N - value_at_level_0` for the same participant, controlling for large
  between-participant differences in resting physiology (see the notebook's
  within-/between-participant variance analysis).

## Participant-aware validation strategy (leakage prevention)

Every participant contributes 10 rows (one per exposure level). **All splitting and
cross-validation groups by `participant_id`** (`vr_fear_prediction.splitting`):

- A `GroupShuffleSplit` holds out ~20% of participants entirely for a final held-out test
  -- no participant appears in both train and test.
- Model/hyperparameter selection uses `GroupKFold` cross-validation *on the training
  participants only*.
- All preprocessing (imputation, scaling, one-hot encoding) is fit inside a scikit-learn
  `Pipeline`, so it only ever sees training folds.
- `tests/test_splitting.py` asserts no participant overlap under both the held-out split
  and every `GroupKFold` fold; `tests/test_features.py` asserts the target and participant
  id are never present in the feature matrix.

**Other leakage risks identified and excluded** (full list and reasons in
[`data/README.md`](data/README.md) and `vr_fear_prediction.data.LEAKAGE_EXCLUDED_COLUMNS`):
post-exposure questionnaire scores (measured *after* the session, would leak the outcome),
cortisol (sampling times don't align with individual levels), sensor cross-agreement QA
metrics, free-text post-session reflections, and the two SUDS ratings with no matching
physiological feature block.

## Model comparison: context-only vs. physiology-only vs. combined

Three feature sets, compared head-to-head, to test whether physiology adds predictive
value beyond simply knowing the exposure stage and the participant's trait questionnaire
scores:

- **context_only**: exposure level + trait/demographic questionnaire scores (SPQ, STAI
  trait/pre-exposure state anxiety, age, gender, prior VR/AR experience, spider fear/
  contact history). No physiological measurement.
- **physiology_only**: the precomputed physiological features + baseline deltas above.
- **combined**: union of both.

Each is evaluated with `DummyRegressor` (mean baseline), `Ridge` (regularized linear,
`alpha` tuned via a small `GridSearchCV` grid), and `RandomForestRegressor` (`max_depth`/
`min_samples_leaf` tuned via the same small grid) -- see `vr_fear_prediction.modeling` for
the exact search spaces.

## Results

From a completed run of `python -m vr_fear_prediction.modeling`, reported in full (all
models, both CV and held-out-test metrics) in [`reports/metrics.json`](reports/metrics.json).

| Feature set | Best model (by CV) | CV MAE (mean ± sd) | Held-out test MAE | Held-out test RMSE | Held-out test R² |
|---|---|---|---|---|---|
| context_only | Ridge (α=10) | 1.82 ± 0.12 | 1.31 | 1.86 | 0.19 |
| physiology_only | Dummy (mean) | 2.37 ± 0.51 | 2.56 | 2.82 | -0.86 |
| combined | Ridge (α=50) | 2.09 ± 0.20 | **0.89** | **1.36** | **0.57** |

(Dummy-mean baseline, shown for reference: CV MAE 2.37 ± 0.51, held-out test MAE 2.56,
R² -0.86 -- identical across feature sets, as expected, since it ignores all features.)

**Does physiology add value beyond context?** `reports/metrics.json` records this
explicitly (`physiology_adds_value`): on the held-out test participants, **yes** --
combined (MAE 0.89) clearly beats context_only (MAE 1.31). **Physiology alone is weak**:
no model beats the dummy baseline on physiology_only under cross-validation, and only
modestly on the held-out test -- consistent with the source paper's central finding that
subjective and physiological fear are related but substantially dissociated.

### A note on model selection instability

Under participant-aware CV, `context_only` has a marginally *better* mean MAE (1.82) than
`combined` (2.09) -- the opposite ranking from the held-out test. With only ~22 training
participants split into 5 CV folds (and only ~6 participants in the held-out test), this
kind of disagreement between CV and a single held-out split is expected small-sample
noise, not a contradiction to paper over. This project reports **both** criteria rather
than picking whichever looks better, and treats the held-out-test result for `combined`
(the feature set most relevant to the primary research question) as the more informative
number for the "does physiology help" question, while still using CV MAE as a defensible,
principled criterion for *automated* model/feature-set selection
(`vr_fear_prediction.modeling.run_full_pipeline`). A larger sample would be needed to
resolve this instability with confidence.

## Figures

![Self-reported fear and physiology across exposure levels](reports/figures/suds_and_physiology_by_level.png)

*Self-reported fear rises with exposure level with a clear floor effect at baseline; the
physiological measure (EDA) shows a related but visibly noisier pattern.*

![Agreement vs. dissociation between subjective and physiological fear](reports/figures/subjective_vs_physiological_dissociation.png)

*Per-participant mean self-reported fear vs. mean physiological reactivity during
exposure (both z-scored). Points near the diagonal show agreement; points far from it
(labeled) show dissociation -- some participants report high fear with little
physiological change, or vice versa.*

![Held-out test MAE by feature set and model](reports/figures/feature_set_model_comparison.png)

*The three-way feature-set comparison: physiology alone rarely beats the dummy baseline,
but combining it with context clearly improves held-out test performance.*

Additional figures (predicted-vs-observed, residuals, permutation importance -- generated
both for the CV-selected best feature set and for `combined` specifically) are in
[`reports/figures/`](reports/figures/).

## Interpretation

- **Permutation importance** (computed on held-out participants only,
  `reports/figures/permutation_importance*.png` and `reports/permutation_importance*.csv`)
  identifies `SPQ_F` (Fear/Disgust Reaction subscale of the Spider Phobia Questionnaire)
  as the strongest predictor in the context-only model -- a psychologically sensible
  result (trait spider-related fear/disgust predicts state fear during spider exposure).
- **Standardized Ridge coefficients** for the combined model
  (`reports/ridge_coefficients_combined.csv`) are directly comparable in magnitude because
  all numeric inputs are standardized and categoricals one-hot encoded inside the
  pipeline.
- Feature importance here identifies **association, not causation**. Physiological
  arousal is not unique to fear (effort, excitement, novelty, and sensor motion artifacts
  can all raise heart rate or EDA) -- a rise in these signals is not proof that someone is
  afraid. Individual physiological baselines vary substantially (see the notebook's
  within-/between-participant variance analysis), which is exactly why this project models
  baseline-corrected deltas alongside raw levels rather than raw values alone.

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m vr_fear_prediction.download_data          # fetch summary.csv (~467 MB download, ~few hundred KB extracted)
python -m vr_fear_prediction.data                   # validate + reshape -> data/processed/features_long.csv
python -m vr_fear_prediction.modeling                # train/evaluate all models -> reports/metrics.json, reports/figures/, models/best_pipeline.joblib
pytest -v                                            # run the test suite (uses small local fixtures, no download needed)
```

Optional, for the raw-signal-processing demo in the notebook:

```bash
python -m vr_fear_prediction.download_data --with-raw-signals p31   # fetch one raw recording (~150 MB)
jupyter notebook notebooks/01_exploration.ipynb
```

All randomness is seeded (`vr_fear_prediction.splitting.RANDOM_SEED = 42`); re-running
`vr_fear_prediction.modeling` reproduces `reports/metrics.json` exactly.

## Repository structure

```text
vr-fear-response-prediction/
├── .github/workflows/tests.yml       # CI: installs deps, runs pytest
├── data/
│   ├── README.md                     # full dataset documentation, citation, exclusions
│   ├── raw/                          # downloaded from Zenodo (gitignored)
│   ├── interim/                      # reserved (gitignored)
│   └── processed/features_long.csv   # reshaped (participant, level) table (committed)
├── models/best_pipeline.joblib       # the CV-selected fitted pipeline
├── notebooks/01_exploration.ipynb    # EDA, dissociation analysis, raw-signal demo
├── reports/
│   ├── figures/                      # generated plots
│   ├── metrics.json                  # full results (all feature sets x models)
│   ├── permutation_importance*.csv
│   ├── ridge_coefficients_combined.csv
│   └── model_card.md
├── src/vr_fear_prediction/
│   ├── download_data.py              # Zenodo download + checksum verification
│   ├── data.py                       # load, validate, reshape wide -> long
│   ├── signal_processing.py          # NeuroKit2 raw-signal feature extraction
│   ├── features.py                   # context/physiology/combined feature matrices
│   ├── splitting.py                  # participant-aware split/CV helpers
│   ├── modeling.py                   # pipelines, GridSearchCV, evaluation orchestration
│   ├── evaluation.py                 # metrics, permutation importance, rm-correlation
│   └── visualization.py              # all plotting functions
├── tests/                            # pytest suite, small local fixtures only
├── LICENSE                           # MIT (code only; data is CC-BY-4.0, see data/README.md)
├── requirements.txt
└── pyproject.toml
```

## Limitations

- **N = 28 participants** -- every metric here has wide uncertainty; treat point estimates
  as illustrative, not precise.
- **A single held-out test split** at this sample size can flip which feature set "wins"
  (see "A note on model selection instability" above); a larger sample or repeated-split
  analysis would be needed to resolve this with confidence.
- **Hyperparameter selection and CV performance estimation share the same folds** (not a
  fully nested CV), which is somewhat optimistic; full nesting was judged impractical at
  N≈22 training participants.
- **The from-scratch raw-signal demo does not closely match the official precomputed
  features** in absolute terms (see the notebook) -- disclosed as an honest finding, not
  hidden.
- Results are specific to this VR spider-exposure protocol, this sensor hardware, and this
  population; they do not generalize to other phobias, VR content, devices, or clinical
  populations without independent validation.

See `reports/model_card.md` for the complete limitations and ethical-considerations
discussion.

## Responsible use

> This project explores whether physiological signals contain useful information about
> self-reported fear during VR exposure.

It does **not** claim to accurately detect fear, does not diagnose a phobia, and cannot
replace a participant's own report or a clinician's judgment. Physiological arousal is not
unique to fear and should never be treated as ground truth that overrides what a person
says about their own internal state.

## Future integration with adaptive VR exposure systems

A cautious, research-only direction this project's results could motivate: a future
adaptive exposure system might use a model like `combined` as **one additional signal**
alongside continuous self-report (not a replacement for it) when deciding whether to hold,
advance, or step back the exposure intensity -- always with the participant able to
override the system via their own report, and always under clinical supervision. This
project does not implement, validate, or recommend deploying such a system; it only
establishes that physiology-plus-context carries more information about self-reported fear
than context alone, on this one dataset.
