"""Assemble context-only, physiology-only, and combined feature matrices.

The three feature sets exist to answer the project's practical motivation directly: does
physiology add predictive value beyond simply knowing which exposure stage a participant
is in plus their trait questionnaire scores? See README.md "Model comparison".
"""

from __future__ import annotations

import pandas as pd

from vr_fear_prediction.data import (
    CONTEXT_STATIC_COLUMNS,
    LEVEL_COLUMN,
    PARTICIPANT_COLUMN,
    PHYSIO_FEATURE_TEMPLATES,
    TARGET_COLUMN,
)

# CONTEXT: the exposure stage/level itself plus stable participant traits collected before
# the physiological session even starts. Represents "what an adaptive VR system would know
# about this participant and this stage without measuring their body at all".
CONTEXT_NUMERIC_COLUMNS: list[str] = [
    LEVEL_COLUMN,
    "age",
    "SPQ_total",
    "SPQ_A (Avoidance Behavior)",
    "SPQ_F (Fear/Disgust Reaction)",
    "SPQ_I (Cognitive Beliefs)",
    "questionnaire_stai_x2",
    "questionnaire_stai_x1_before",
]
CONTEXT_CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "spider_fear",
    "spider_contact",
    "housing",
    "education",
    "vr_experience",
    "ar_experience",
    "vr_ar_frequency",
]
assert set(CONTEXT_NUMERIC_COLUMNS + CONTEXT_CATEGORICAL_COLUMNS) - {LEVEL_COLUMN} == set(
    CONTEXT_STATIC_COLUMNS
)

# PHYSIOLOGY: precomputed per-level physiological features plus their baseline-corrected
# deltas (value at this level minus the participant's own level-0 baseline).
PHYSIO_COLUMNS: list[str] = PHYSIO_FEATURE_TEMPLATES + [
    f"{tmpl}_delta_baseline" for tmpl in PHYSIO_FEATURE_TEMPLATES
]

FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "context_only": {"numeric": CONTEXT_NUMERIC_COLUMNS, "categorical": CONTEXT_CATEGORICAL_COLUMNS},
    "physiology_only": {"numeric": PHYSIO_COLUMNS, "categorical": []},
    "combined": {
        "numeric": CONTEXT_NUMERIC_COLUMNS + PHYSIO_COLUMNS,
        "categorical": CONTEXT_CATEGORICAL_COLUMNS,
    },
}

NON_FEATURE_COLUMNS = {TARGET_COLUMN, PARTICIPANT_COLUMN}


def get_feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    """Return (numeric_columns, categorical_columns) for a named feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{feature_set}'. Choose from {list(FEATURE_SETS)}")
    spec = FEATURE_SETS[feature_set]
    return list(spec["numeric"]), list(spec["categorical"])


def build_feature_matrix(long_df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, groups) for a named feature set.

    X never contains the target column or any column in data.LEAKAGE_EXCLUDED_COLUMNS.
    groups is the participant id, for use with participant-aware splitting/CV.
    """
    numeric_cols, categorical_cols = get_feature_columns(feature_set)
    feature_cols = numeric_cols + categorical_cols
    missing = [c for c in feature_cols if c not in long_df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from input dataframe: {missing}")

    leaked = [c for c in feature_cols if c in NON_FEATURE_COLUMNS]
    if leaked:
        raise ValueError(f"Refusing to build feature matrix: target/id columns requested as features: {leaked}")

    X = long_df[feature_cols].copy()
    y = long_df[TARGET_COLUMN].copy()
    groups = long_df[PARTICIPANT_COLUMN].copy()
    return X, y, groups
