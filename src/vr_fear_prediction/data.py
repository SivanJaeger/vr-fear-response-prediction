"""Load, validate, and reshape the participant-level summary dataset.

The raw ``summary.csv`` is one row per participant, wide across 10 VR exposure levels
(level 0 = neutral baseline, levels 1-9 = escalating spider-exposure tasks). Each level
carries both a precomputed physiological feature block and a matching self-reported
SUDS (Subjective Units of Distress Scale, 0-10) fear rating.

This module reshapes that wide table into a long, participant-aware table with one row
per (participant, level) observation -- the unit of analysis used throughout the rest of
the pipeline. It also runs a set of explicit, documented data-validation checks rather
than silently dropping or coercing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "summary.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_LONG_PATH = PROCESSED_DIR / "features_long.csv"

# Levels 0-9 have precomputed physiological features in summary.csv (level 10 and the
# suds_10/suds_11 ratings exist in the raw telemetry / questionnaire but have no matching
# precomputed physiological feature block, so they are documented and excluded -- see
# data/README.md "Excluded observations").
LEVELS: list[int] = list(range(10))

PHYSIO_FEATURE_TEMPLATES: list[str] = [
    "PPG_HR",
    "PULSE_HR",
    "RESP_RATE",
    "EDA_MEAN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_LFHF",
    "WATCH_STD",
    "WATCH_RMSSD",
    "WATCH_REACTIVITY",
    "WATCH_SLOPE",
    "WATCH_RANGE",
]

# Participant-level trait / demographic columns used as CONTEXT features. Deliberately
# excludes questionnaire_stai_x1_after and questionnaire_stai_x1_diff, which are measured
# *after* the full exposure session and would leak session-outcome information into
# predictions for individual (earlier) exposure levels.
CONTEXT_STATIC_COLUMNS: list[str] = [
    "age",
    "gender",
    "spider_fear",
    "spider_contact",
    "housing",
    "education",
    "vr_experience",
    "ar_experience",
    "vr_ar_frequency",
    "SPQ_total",
    "SPQ_A (Avoidance Behavior)",
    "SPQ_F (Fear/Disgust Reaction)",
    "SPQ_I (Cognitive Beliefs)",
    "questionnaire_stai_x2",
    "questionnaire_stai_x1_before",
]

# Columns explicitly excluded from every feature set, with the reason. Kept here so the
# exclusion is visible and testable rather than implicit.
LEAKAGE_EXCLUDED_COLUMNS: dict[str, str] = {
    "questionnaire_stai_x1_after": "measured after the full exposure session ends; leaks session outcome",
    "questionnaire_stai_x1_diff": "derived from questionnaire_stai_x1_after; same leakage risk",
    "suds_10": "no matching precomputed physiological feature block (only levels 0-9 exist)",
    "suds_11": "no matching precomputed physiological feature block (only levels 0-9 exist)",
    "vr_feelings": "free-text post-session reflection; effectively restates the outcome",
    "cort_0": "cortisol sampling times do not align with individual exposure levels (session-level aggregate)",
    "cort_1": "cortisol sampling times do not align with individual exposure levels (session-level aggregate)",
    "cort_2": "cortisol sampling times do not align with individual exposure levels (session-level aggregate)",
    "cort_3": "cortisol sampling times do not align with individual exposure levels (session-level aggregate)",
    "cort_4": "cortisol sampling times do not align with individual exposure levels (session-level aggregate)",
    "cort_5": "cortisol sampling times do not align with individual exposure levels (session-level aggregate)",
    "cort_6": "cortisol sampling times do not align with individual exposure levels (session-level aggregate)",
    "HR_corr": "sensor cross-agreement QA metric, not a physiological response to the stimulus",
    "HR_mae": "sensor cross-agreement QA metric, not a physiological response to the stimulus",
    "HR_rmse": "sensor cross-agreement QA metric, not a physiological response to the stimulus",
    "HR_bias": "sensor cross-agreement QA metric, not a physiological response to the stimulus",
}

REQUIRED_RAW_COLUMNS = (
    ["participant"]
    + [f"suds_{lvl}" for lvl in LEVELS]
    + [f"{tmpl}_level{lvl}" for tmpl in PHYSIO_FEATURE_TEMPLATES for lvl in LEVELS]
    + CONTEXT_STATIC_COLUMNS
)

TARGET_COLUMN = "suds"
PARTICIPANT_COLUMN = "participant_id"
LEVEL_COLUMN = "level"

SUDS_MIN, SUDS_MAX = 0, 10


@dataclass
class ValidationReport:
    """Result of validating the raw summary table. Never silently drops rows."""

    n_rows_in: int
    n_participants: int
    missing_columns: list[str] = field(default_factory=list)
    duplicate_participants: list[str] = field(default_factory=list)
    out_of_range_suds: list[tuple[str, str, float]] = field(default_factory=list)
    fully_missing_columns: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not (self.missing_columns or self.duplicate_participants or self.out_of_range_suds)

    def raise_if_invalid(self) -> None:
        if self.missing_columns:
            raise ValueError(f"summary.csv is missing required columns: {self.missing_columns}")
        if self.duplicate_participants:
            raise ValueError(f"Duplicate participant ids found: {self.duplicate_participants}")
        if self.out_of_range_suds:
            raise ValueError(f"SUDS ratings outside [{SUDS_MIN}, {SUDS_MAX}]: {self.out_of_range_suds}")


def load_raw_summary(path: Path | None = None) -> pd.DataFrame:
    """Load the raw wide-format participant summary table.

    Raises a clear error (rather than a confusing pandas traceback) if the file has not
    been downloaded yet. ``path`` defaults to the module-level RAW_SUMMARY_PATH, resolved
    at call time (not import time) so it can be overridden via monkeypatching in tests.
    """
    if path is None:
        path = RAW_SUMMARY_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m vr_fear_prediction.download_data` first."
        )
    return pd.read_csv(path)


def validate_raw_summary(df: pd.DataFrame) -> ValidationReport:
    """Run explicit, documented validation checks on the raw summary table."""
    missing_columns = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]

    duplicate_participants = (
        df["participant"][df["participant"].duplicated()].tolist() if "participant" in df.columns else []
    )

    out_of_range: list[tuple[str, str, float]] = []
    for lvl in LEVELS:
        col = f"suds_{lvl}"
        if col not in df.columns:
            continue
        bad = df[(df[col] < SUDS_MIN) | (df[col] > SUDS_MAX)]
        for _, row in bad.iterrows():
            out_of_range.append((row.get("participant", "?"), col, row[col]))

    fully_missing = [c for c in df.columns if df[c].isna().all()]

    return ValidationReport(
        n_rows_in=len(df),
        n_participants=df["participant"].nunique() if "participant" in df.columns else 0,
        missing_columns=missing_columns,
        duplicate_participants=duplicate_participants,
        out_of_range_suds=out_of_range,
        fully_missing_columns=fully_missing,
    )


def _normalize_participant_id(raw_id: str) -> str:
    """Normalize e.g. 'P04' / 'p4' / 4 into the canonical 'P04' form."""
    digits = "".join(ch for ch in str(raw_id) if ch.isdigit())
    return f"P{int(digits):02d}"


def reshape_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide per-participant summary table into a long (participant, level) table.

    One output row per participant x exposure level (0-9). Adds baseline-corrected
    "delta from level 0" physiological features, since baseline correction is a standard
    and interpretable way to control for each participant's individual physiological
    baseline (see reports/model_card.md for rationale).
    """
    records = []
    for _, row in df.iterrows():
        participant_id = _normalize_participant_id(row["participant"])
        baseline = {tmpl: row[f"{tmpl}_level0"] for tmpl in PHYSIO_FEATURE_TEMPLATES}
        for lvl in LEVELS:
            record: dict[str, object] = {
                PARTICIPANT_COLUMN: participant_id,
                LEVEL_COLUMN: lvl,
                TARGET_COLUMN: row[f"suds_{lvl}"],
            }
            for tmpl in PHYSIO_FEATURE_TEMPLATES:
                value = row[f"{tmpl}_level{lvl}"]
                record[tmpl] = value
                record[f"{tmpl}_delta_baseline"] = value - baseline[tmpl]
            for col in CONTEXT_STATIC_COLUMNS:
                record[col] = row[col]
            records.append(record)
    long_df = pd.DataFrame.from_records(records)
    long_df = long_df.sort_values([PARTICIPANT_COLUMN, LEVEL_COLUMN]).reset_index(drop=True)
    return long_df


def build_long_dataset(raw_path: Path | None = None) -> tuple[pd.DataFrame, ValidationReport]:
    """Full load -> validate -> reshape pipeline. Raises on invalid data."""
    raw_df = load_raw_summary(raw_path)
    report = validate_raw_summary(raw_df)
    report.raise_if_invalid()
    long_df = reshape_wide_to_long(raw_df)
    return long_df, report


def save_processed(long_df: pd.DataFrame, out_path: Path = PROCESSED_LONG_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_path, index=False)


def main() -> None:
    long_df, report = build_long_dataset()
    print(f"Loaded {report.n_rows_in} participants from raw summary.csv")
    print(f"Reshaped into {len(long_df)} (participant, level) observations "
          f"({report.n_participants} participants x {len(LEVELS)} levels)")
    print(f"Target ('{TARGET_COLUMN}') distribution:\n{long_df[TARGET_COLUMN].describe()}")
    save_processed(long_df)
    print(f"Saved processed long-format table to {PROCESSED_LONG_PATH}")


if __name__ == "__main__":
    main()
