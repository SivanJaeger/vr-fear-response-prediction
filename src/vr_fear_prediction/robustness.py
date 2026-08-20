"""Robustness checks for the context-vs-physiology comparison.

The primary analysis (`vr_fear_prediction.modeling.compare_feature_sets`) compares
context-only and combined models on a *single* held-out-participant split. With only 28
participants, that single split is not automatically more informative than
participant-aware cross-validation, and the two can disagree (as they do on this
dataset). This module answers two follow-up questions:

1. **Is the single-split result stable?** `repeated_participant_holdout` reruns the
   context-only vs. combined Ridge comparison across many different outer
   `GroupShuffleSplit` seeds (different choices of which participants are held out), so
   the comparison can be judged across a *distribution* of splits instead of one.
2. **Is the result driven by the level-0 floor effect?** `sensitivity_excluding_baseline`
   repeats the primary analysis and the repeated-holdout evaluation restricted to exposure
   levels 1-9, since level 0 has near-zero SUDS for most participants and all
   baseline-corrected physiological deltas are trivially 0 there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.pipeline import Pipeline

from vr_fear_prediction.features import build_feature_matrix, get_feature_columns
from vr_fear_prediction.modeling import build_preprocessor, compare_feature_sets
from vr_fear_prediction.splitting import assert_no_participant_overlap, participant_group_kfold

RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 50.0]
RIDGE_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 50
DEFAULT_TEST_SIZE = 0.2
DEFAULT_BASE_SEED = 1000
INNER_CV_FOLDS = 5


def _fit_ridge_and_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, float]:
    """Select Ridge's alpha via inner participant-aware GroupKFold on the training
    participants only, then predict on the (untouched-by-selection) test participants."""
    pipeline = Pipeline(
        [("preprocess", build_preprocessor(numeric_cols, categorical_cols)), ("model", Ridge(random_state=RIDGE_RANDOM_STATE))]
    )
    n_inner_splits = min(INNER_CV_FOLDS, groups_train.nunique())
    inner_cv = participant_group_kfold(n_splits=n_inner_splits)
    search = GridSearchCV(
        pipeline, param_grid={"model__alpha": RIDGE_ALPHA_GRID}, cv=inner_cv, scoring="neg_mean_absolute_error", refit=True
    )
    search.fit(X_train, y_train, groups=groups_train)
    y_pred = search.best_estimator_.predict(X_test)
    return y_pred, float(search.best_params_["model__alpha"])


def repeated_participant_holdout(
    long_df: pd.DataFrame,
    n_splits: int | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    base_seed: int = DEFAULT_BASE_SEED,
) -> pd.DataFrame:
    """Repeat the context-only vs. combined Ridge comparison across `n_splits` different
    outer `GroupShuffleSplit` seeds.

    For every outer split: complete participants are held out (never split by row), Ridge's
    alpha is selected using an inner GroupKFold on the outer-training participants only,
    and -- because both feature sets are built from the same `long_df` and therefore have
    identical row order/participant assignment -- the same `(train_idx, test_idx)` pair is
    reused for both context_only and combined, so the paired MAE difference is a fair,
    same-participants comparison. Every split is checked for participant overlap before
    being used.

    `n_splits` defaults to the module-level DEFAULT_N_SPLITS, resolved at call time (not
    bound at import time) so tests can monkeypatch DEFAULT_N_SPLITS for a fast run.
    """
    if n_splits is None:
        n_splits = DEFAULT_N_SPLITS
    X_context, y, groups = build_feature_matrix(long_df, "context_only")
    X_combined, _, _ = build_feature_matrix(long_df, "combined")
    context_numeric, context_categorical = get_feature_columns("context_only")
    combined_numeric, combined_categorical = get_feature_columns("combined")

    rows = []
    for i in range(n_splits):
        seed = base_seed + i
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X_context, y, groups=groups))
        assert_no_participant_overlap(groups, train_idx, test_idx)

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        context_pred, context_alpha = _fit_ridge_and_predict(
            X_context.iloc[train_idx], y_train, groups_train, X_context.iloc[test_idx], context_numeric, context_categorical
        )
        combined_pred, combined_alpha = _fit_ridge_and_predict(
            X_combined.iloc[train_idx], y_train, groups_train, X_combined.iloc[test_idx], combined_numeric, combined_categorical
        )

        context_mae = float(mean_absolute_error(y_test, context_pred))
        combined_mae = float(mean_absolute_error(y_test, combined_pred))

        rows.append(
            {
                "split_id": i,
                "seed": seed,
                "n_train_participants": int(groups_train.nunique()),
                "n_test_participants": int(groups.iloc[test_idx].nunique()),
                "context_only_MAE": context_mae,
                "combined_MAE": combined_mae,
                "diff_context_minus_combined": context_mae - combined_mae,
                "context_only_best_alpha": context_alpha,
                "combined_best_alpha": combined_alpha,
            }
        )

    return pd.DataFrame(rows)


def summarize_repeated_holdout(results_df: pd.DataFrame) -> dict:
    """Summary statistics for a repeated_participant_holdout result.

    `fraction_splits_combined_better` and the percentile interval on the paired
    difference are the key numbers for judging stability: values far from 50% / entirely
    on one side of zero indicate a consistent winner, values near 50% / straddling zero
    indicate the single-split comparison is not trustworthy on its own.
    """
    diff = results_df["diff_context_minus_combined"]
    return {
        "n_splits": int(len(results_df)),
        "context_only": {
            "MAE_mean": float(results_df["context_only_MAE"].mean()),
            "MAE_median": float(results_df["context_only_MAE"].median()),
            "MAE_std": float(results_df["context_only_MAE"].std()),
        },
        "combined": {
            "MAE_mean": float(results_df["combined_MAE"].mean()),
            "MAE_median": float(results_df["combined_MAE"].median()),
            "MAE_std": float(results_df["combined_MAE"].std()),
        },
        "paired_difference_context_minus_combined": {
            "mean": float(diff.mean()),
            "median": float(diff.median()),
            "std": float(diff.std()),
            "ci_2.5pct": float(np.percentile(diff, 2.5)),
            "ci_97.5pct": float(np.percentile(diff, 97.5)),
            "fraction_splits_combined_better": float((diff > 0).mean()),
        },
    }


_SUMMARY_TEXT = {
    "inconclusive": (
        "Evidence that physiology adds predictive value beyond context is inconclusive. "
        "The combined model performs better on one held-out participant split, but worse "
        "under participant-aware cross-validation. With only 28 participants, this "
        "difference is unstable and may depend strongly on which participants are held out."
    ),
    "combined_likely_better": (
        "Combining physiology with context tends to outperform context alone across the "
        "original held-out split, participant-aware cross-validation, and a repeated "
        "participant-level holdout evaluation -- the most consistent evidence this project "
        "has that physiology adds predictive value, though still based on only 28 "
        "participants."
    ),
    "context_only_likely_sufficient": (
        "Across the original held-out split, participant-aware cross-validation, and a "
        "repeated participant-level holdout evaluation, adding physiology to context did "
        "not consistently improve prediction here -- context alone appears sufficient on "
        "this dataset, though this is based on only 28 participants."
    ),
}


def build_physiology_value_assessment(
    context_test_mae: float,
    combined_test_mae: float,
    context_cv_mae: float,
    combined_cv_mae: float,
    repeated_summary: dict,
) -> dict:
    """Combine the original-split result, the CV result, and the repeated-holdout result
    into one explicit conclusion, rather than treating any single criterion as decisive.

    `overall_conclusion` is only ever "combined_likely_better" or
    "context_only_likely_sufficient" when the original split AND cross-validation agree on
    the direction AND the repeated-holdout evaluation decisively (>=60%/<=40% of splits)
    supports that same direction. Any disagreement between criteria -- exactly what happens
    on the primary analysis of this dataset -- yields "inconclusive".
    """
    combined_wins_original_split = combined_test_mae < context_test_mae
    combined_wins_cv = combined_cv_mae < context_cv_mae
    frac_combined_better = repeated_summary["paired_difference_context_minus_combined"]["fraction_splits_combined_better"]

    criteria_agree = combined_wins_original_split == combined_wins_cv
    if not criteria_agree:
        overall_conclusion = "inconclusive"
    elif combined_wins_cv and frac_combined_better >= 0.6:
        overall_conclusion = "combined_likely_better"
    elif (not combined_wins_cv) and frac_combined_better <= 0.4:
        overall_conclusion = "context_only_likely_sufficient"
    else:
        overall_conclusion = "inconclusive"

    return {
        "original_held_out_split": {
            "context_only_best_test_MAE": context_test_mae,
            "combined_best_test_MAE": combined_test_mae,
            "combined_improves_on_context_original_split": bool(combined_wins_original_split),
        },
        "participant_aware_cv": {
            "context_only_best_cv_MAE_mean": context_cv_mae,
            "combined_best_cv_MAE_mean": combined_cv_mae,
            "combined_improves_on_context_cv": bool(combined_wins_cv),
        },
        "repeated_participant_holdout": repeated_summary,
        "overall_conclusion": overall_conclusion,
        "summary": _SUMMARY_TEXT[overall_conclusion],
    }


def sensitivity_excluding_baseline(long_df: pd.DataFrame, n_splits: int | None = None) -> dict:
    """Sensitivity analysis: repeat the main feature-set comparison and the repeated
    participant-level holdout evaluation using only exposure levels 1-9 (baseline level 0
    excluded).

    Level 0 has a strong SUDS floor effect (most participants report 0 fear) and every
    baseline-corrected physiological delta is trivially 0 there, which could make
    prediction artificially easy and inflate apparent performance. This is a secondary,
    clearly-labeled check -- it does not replace the primary analysis, which uses all
    levels 0-9.
    """
    subset = long_df[long_df["level"] > 0].reset_index(drop=True)

    feature_set_metrics = compare_feature_sets(subset)

    repeated_df = repeated_participant_holdout(subset, n_splits=n_splits)
    repeated_summary = summarize_repeated_holdout(repeated_df)

    context_result = feature_set_metrics["context_only"]
    combined_result = feature_set_metrics["combined"]
    physiology_value_assessment = build_physiology_value_assessment(
        context_result[context_result["best_model"]]["held_out_test"]["MAE"],
        combined_result[combined_result["best_model"]]["held_out_test"]["MAE"],
        context_result[context_result["best_model"]]["cv"]["MAE_mean"],
        combined_result[combined_result["best_model"]]["cv"]["MAE_mean"],
        repeated_summary,
    )

    metrics = {
        "description": (
            "Sensitivity analysis restricted to exposure levels 1-9 (baseline level 0 "
            "excluded), since level 0 has a strong SUDS floor effect and all "
            "baseline-corrected physiological deltas are trivially 0 there, which could "
            "make prediction artificially easy. This does not replace the primary "
            "analysis, which uses all levels 0-9."
        ),
        "n_observations": int(len(subset)),
        "levels_included": sorted(subset["level"].unique().tolist()),
        "feature_sets": feature_set_metrics,
        "physiology_value_assessment": physiology_value_assessment,
    }
    return {"metrics": metrics, "repeated_df": repeated_df}
