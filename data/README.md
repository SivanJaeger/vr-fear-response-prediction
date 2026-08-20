# Dataset

## Source

**Title:** Dataset for "Do Virtual Spiders Feel Real? Dissociation Between Subjective and
Physiological Fear During Virtual Reality Exposure"

**Authors:** Michal Krumnikl, Martin Nemec, Adéla Vilímková, Filip Huser, Vladěna
Jaremová (VSB - Technical University of Ostrava; University of Ostrava)

**Repository:** Zenodo, record 20796399
**DOI:** [10.5281/zenodo.20796399](https://doi.org/10.5281/zenodo.20796399)
**License:** CC-BY-4.0
**Publication date (Zenodo record):** 2026-06-22
**Access:** Open, publicly downloadable, verified via the Zenodo REST API
(`https://zenodo.org/api/records/20796399`) on 2026-08-20.

**Citation:**

> Krumnikl, M., Nemec, M., Vilímková, A., Huser, F., & Jaremová, V. (2026). Dataset for
> "Do Virtual Spiders Feel Real? Dissociation Between Subjective and Physiological Fear
> During Virtual Reality Exposure" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20796399

## Contents of the archive (`vr_spider_dataset.zip`, ~467 MB compressed / ~9 GB uncompressed)

- `pXX.csv` (28 files, one per participant, 150 MB-1.2 GB each): raw time-series
  recordings during the VR session -- respiration (`RESP`), photoplethysmography (`PPG`),
  electrodermal activity (`EDA`), smartwatch heart rate (`pulse_rate`), VR headset
  telemetry (position/orientation), and JSON-encoded spider-scene metadata (`Spiders`,
  `Count`, `Size`). `ECG` is present as a column header but was not recorded (always
  empty). Sampled at ~2000 Hz for the physiological channels.
- `pX_description.txt`: column definitions for the `pXX.csv` files.
- `summary.csv` (28 rows x 179 columns): one row per participant, with demographics,
  questionnaire scores (STAI, SPQ), cortisol samples, subjective SUDS fear ratings
  (`suds_0` .. `suds_11`), and physiological features precomputed by the dataset authors
  for VR exposure levels 0-9 (`PPG_HR_levelN`, `EDA_MEAN_levelN`, `HRV_RMSSD_levelN`, etc.).
- `summary_description.txt`: column definitions for `summary.csv`.

## Participants and unit of observation

**28 participants** (pseudonymized ids `P04`-`P31`; ids `P01`-`P03` are not present in the
public release). Each participant went through a fixed VR protocol: **level 0** is a
neutral baseline (no spiders), **levels 1-9** are nine escalating spider-exposure tasks
(closer/larger/more spiders). The dataset authors precomputed physiological features for
levels 0-9 only (level 10 and SUDS ratings 10-11 exist in the raw data/questionnaire but
have no matching precomputed physiological feature block -- see "Excluded observations"
below).

This project's unit of analysis is **one row per (participant, exposure level)** --
**280 observations from 28 participants x 10 levels**, produced by
`vr_fear_prediction.data.reshape_wide_to_long` from the wide `summary.csv`. Every
participant contributes exactly 10 rows, which is why **all splitting and
cross-validation in this project groups by participant** (see README.md "Validation approach").

## Physiological signals available

Raw, per-sample (`pXX.csv`, ~2000 Hz): PPG, EDA, respiration, smartwatch heart rate, VR
telemetry.

Precomputed, per-level (`summary.csv`, used by the main pipeline): heart rate from PPG
peak detection, heart-rate variability (SDNN, RMSSD, LF/HF ratio) from PPG, smartwatch
heart rate and derived reactivity/trend statistics, respiratory rate, mean EDA, and
PPG-vs-smartwatch agreement metrics (correlation, MAE, RMSE, bias).

## Subjective fear ratings

SUDS (Subjective Units of Distress Scale, 0-10) collected once per exposure level:
`suds_0` (baseline) through `suds_9` (level 9). Two additional ratings, `suds_10` and
`suds_11`, were collected after the main protocol but have no matching physiological
feature block and are excluded from modeling (see below).

## Synchronization

Raw signals carry a Unix-microsecond `Timestamp` column and a `Level` column identifying
the currently active VR exposure level (verified directly from a raw recording: level
transitions are contiguous blocks of consecutive samples, e.g. level 0 lasted ~125s for
one participant, level 1 began immediately after). The precomputed `summary.csv` features
are aggregated within each level's contiguous block. SUDS ratings were collected once per
level, immediately associated with that level's physiological data.

## Sampling rates and units

Raw physiological channels: ~2000-2070 Hz (estimated from median inter-sample interval;
varies slightly per participant/session). PPG/EDA/RESP are raw sensor units (not
pre-calibrated to physical units by the manufacturer's SDK). `pulse_rate` (smartwatch) is
in BPM. Precomputed `PPG_HR`/`PULSE_HR` are in BPM, `HRV_SDNN`/`HRV_RMSSD` in ms,
`RESP_RATE` in breaths/minute, `EDA_MEAN` in raw sensor units (participant- and
session-relative; not calibrated to microsiemens), `SUDS` is unitless 0-10.

## Raw vs. precomputed features

The dataset contains both raw per-sample signals and precomputed per-level features. This
project's analysis uses the dataset authors' precomputed features (`summary.csv`) rather
than reprocessing the raw recordings, which are ~9 GB in total across 28 participants --
impractical to download and process for this project's scope.

## Repeated observations per participant

Yes -- every participant contributes 10 rows (levels 0-9), i.e. this is a repeated-measures
design. See README.md "Validation approach" for how this shapes the splitting strategy.

## Sample size and appropriate analysis

With 28 participants (280 observations, but only 28 independent units for grouping
purposes), this project treats the problem as **cautious exploratory regression**, not a
production-grade predictive system: participant-aware cross-validation with wide
uncertainty bands, a single held-out-participant test split for a final sanity check
(rather than a fully independent test set with strong generalization claims), and no
claims about clinical validity. See `reports/model_card.md` for the full set of
limitations.

## Excluded observations and columns (documented, not silent)

| Excluded | Reason |
|---|---|
| Level 10 / `suds_10`, `suds_11` | No matching precomputed physiological feature block (features only go up to level 9) |
| `questionnaire_stai_x1_after`, `questionnaire_stai_x1_diff` | Measured *after* the full exposure session; would leak session-outcome information into predictions for individual (earlier) levels |
| `cort_0`...`cort_6` (cortisol) | Sampling times do not align with individual exposure levels (session-level aggregate, not level-level) |
| `HR_corr`, `HR_mae`, `HR_rmse`, `HR_bias` | Sensor cross-agreement QA metrics, not physiological responses to the stimulus |
| `vr_feelings` | Free-text post-session reflection that effectively restates the outcome |

The full, programmatically-checked list is `vr_fear_prediction.data.LEAKAGE_EXCLUDED_COLUMNS`.
No row is silently dropped by the pipeline; `vr_fear_prediction.data.validate_raw_summary`
explicitly checks for missing columns, duplicate participants, and out-of-range SUDS
values and raises rather than silently coercing.

## How to obtain the data

The raw data is **not** committed to this repository (large, and better fetched
reproducibly). Run:

```bash
python -m vr_fear_prediction.download_data
```

This downloads the archive from Zenodo (~467 MB), verifies its MD5 checksum against the
value published in the Zenodo record, and extracts only `summary.csv`,
`summary_description.txt`, and `pX_description.txt` into `data/raw/` (a few hundred KB).

## Directory layout

- `data/raw/` -- downloaded, unmodified files from Zenodo (gitignored)
- `data/processed/` -- `features_long.csv`, the reshaped (participant, level) table
  produced by `python -m vr_fear_prediction.data` (gitignored; regenerate it rather than
  committing it)
