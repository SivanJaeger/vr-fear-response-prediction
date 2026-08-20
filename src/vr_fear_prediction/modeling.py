"""Train and evaluate context-only, physiology-only, and combined models.

For every feature set, compares a DummyRegressor baseline against regularized linear
regression (Ridge) and a RandomForestRegressor, using participant-aware GroupKFold cross-
validation for model/hyperparameter comparison and a held-out group of participants for a
final check. All preprocessing (imputation, scaling, one-hot encoding) is fit inside a
scikit-learn Pipeline so it only ever sees training folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vr_fear_prediction import evaluation, visualization
from vr_fear_prediction.data import build_long_dataset
from vr_fear_prediction.features import FEATURE_SETS, build_feature_matrix
from vr_fear_prediction.splitting import (
    RANDOM_SEED,
    assert_no_participant_overlap,
    held_out_participant_split,
    participant_group_kfold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

N_CV_FOLDS = 5
CV_SCORING = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Median-impute + standardize numeric columns; most-frequent-impute + one-hot
    encode categorical columns. Fit only ever happens on training folds (inside a
    Pipeline / cross-validation call), never on the full dataset up front.
    """
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers)


def _model_registry() -> dict[str, tuple[object, dict]]:
    return {
        "dummy_mean": (DummyRegressor(strategy="mean"), {}),
        "ridge": (Ridge(random_state=RANDOM_SEED), {"model__alpha": [0.1, 1.0, 10.0, 50.0]}),
        "random_forest": (
            RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=300),
            {"model__max_depth": [3, 5, None], "model__min_samples_leaf": [1, 3]},
        ),
    }


def _cv_summary(cv_results: dict, best_index: int) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in CV_SCORING:
        mean = cv_results[f"mean_test_{metric}"][best_index]
        std = cv_results[f"std_test_{metric}"][best_index]
        if metric in ("MAE", "RMSE"):
            mean = -mean  # sklearn reports negated error scores
        summary[metric] = {"mean": float(mean), "std": float(std)}
    return summary


def fit_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    model_name: str,
    estimator,
    param_grid: dict,
) -> tuple[Pipeline, dict]:
    """Select hyperparameters (if any) via participant-aware GroupKFold CV on the training
    set, then evaluate the selected pipeline once on the held-out test participants.
    """
    pipeline = Pipeline([("preprocess", build_preprocessor(numeric_cols, categorical_cols)), ("model", estimator)])
    n_splits = min(N_CV_FOLDS, groups_train.nunique())
    cv = participant_group_kfold(n_splits=n_splits)

    if param_grid:
        search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=CV_SCORING, refit="MAE")
        search.fit(X_train, y_train, groups=groups_train)
        best_pipeline = search.best_estimator_
        cv_summary = _cv_summary(search.cv_results_, search.best_index_)
        best_params = {k: v for k, v in search.best_params_.items()}
    else:
        from sklearn.model_selection import cross_validate

        cv_results = cross_validate(
            pipeline, X_train, y_train, groups=groups_train, cv=cv, scoring=CV_SCORING, return_estimator=False
        )
        cv_summary = {
            metric: {
                "mean": float(-np.mean(cv_results[f"test_{metric}"])) if metric in ("MAE", "RMSE") else float(np.mean(cv_results[f"test_{metric}"])),
                "std": float(np.std(cv_results[f"test_{metric}"])),
            }
            for metric in CV_SCORING
        }
        best_pipeline = pipeline.fit(X_train, y_train)
        best_params = {}

    y_pred_test = best_pipeline.predict(X_test)
    test_metrics = evaluation.regression_metrics(y_test.to_numpy(), y_pred_test)

    result = {
        "cv": {f"{metric}_mean": v["mean"] for metric, v in cv_summary.items()}
        | {f"{metric}_std": v["std"] for metric, v in cv_summary.items()},
        "held_out_test": test_metrics,
        "best_params": best_params,
        "n_train_observations": int(len(X_train)),
        "n_test_observations": int(len(X_test)),
    }
    return best_pipeline, result


def compare_feature_sets(long_df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_SEED) -> dict:
    """Run the context-only / physiology-only / combined x Dummy/Ridge/RandomForest
    comparison once on the given long-format dataframe.

    Extracted as its own function so the exact same comparison logic can be reused for
    the primary analysis (all exposure levels) and the baseline-exclusion sensitivity
    analysis (levels 1-9 only, see `vr_fear_prediction.robustness`) without duplication.
    """
    feature_set_metrics: dict = {}

    for feature_set in FEATURE_SETS:
        X, y, groups = build_feature_matrix(long_df, feature_set)
        numeric_cols, categorical_cols = FEATURE_SETS[feature_set]["numeric"], FEATURE_SETS[feature_set]["categorical"]

        train_idx, test_idx = held_out_participant_split(X, y, groups, test_size=test_size, random_state=seed)
        assert_no_participant_overlap(groups, train_idx, test_idx)

        X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        feature_set_metrics[feature_set] = {}
        for model_name, (estimator, param_grid) in _model_registry().items():
            _pipeline, result = fit_and_evaluate(
                X_train, y_train, groups_train, X_test, y_test, numeric_cols, categorical_cols, model_name, estimator, param_grid
            )
            feature_set_metrics[feature_set][model_name] = result

        best_model_name = min(
            feature_set_metrics[feature_set], key=lambda m: feature_set_metrics[feature_set][m]["cv"]["MAE_mean"]
        )
        feature_set_metrics[feature_set]["best_model"] = best_model_name

    return feature_set_metrics


def run_full_pipeline() -> dict:
    """Train/evaluate context-only, physiology-only, and combined models, run the
    repeated participant-holdout and baseline-exclusion robustness checks, and write
    reports/metrics.json and the central figures.
    """
    # Local import: avoids a module-load-time circular import (robustness.py imports
    # build_preprocessor/compare_feature_sets from this module at its top level).
    from vr_fear_prediction import robustness

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    long_df, validation_report = build_long_dataset()

    metrics: dict = {
        "dataset": {
            "n_participants": validation_report.n_participants,
            "n_observations": len(long_df),
            "levels": sorted(long_df["level"].unique().tolist()),
            "target": "suds",
            "target_range": [0, 10],
        },
        "feature_sets": {},
        "random_seed": RANDOM_SEED,
        "cv_folds": N_CV_FOLDS,
    }

    metrics["feature_sets"] = compare_feature_sets(long_df)

    overall_best_feature_set = min(
        FEATURE_SETS, key=lambda fs: metrics["feature_sets"][fs][metrics["feature_sets"][fs]["best_model"]]["cv"]["MAE_mean"]
    )
    overall_best_model = metrics["feature_sets"][overall_best_feature_set]["best_model"]
    metrics["overall_best"] = {"feature_set": overall_best_feature_set, "model": overall_best_model}

    # Does physiology add value beyond context? Judged from three angles at once (single
    # held-out split, participant-aware CV, and a repeated-holdout evaluation) rather than
    # trusting whichever single number looks best -- see robustness.py.
    context_result = metrics["feature_sets"]["context_only"]
    combined_result = metrics["feature_sets"]["combined"]

    primary_repeated_df = robustness.repeated_participant_holdout(long_df)
    primary_repeated_df.to_csv(REPORTS_DIR / "repeated_holdout_results.csv", index=False)
    primary_repeated_summary = robustness.summarize_repeated_holdout(primary_repeated_df)

    metrics["physiology_value_assessment"] = robustness.build_physiology_value_assessment(
        context_result[context_result["best_model"]]["held_out_test"]["MAE"],
        combined_result[combined_result["best_model"]]["held_out_test"]["MAE"],
        context_result[context_result["best_model"]]["cv"]["MAE_mean"],
        combined_result[combined_result["best_model"]]["cv"]["MAE_mean"],
        primary_repeated_summary,
    )

    # Sensitivity analysis: does excluding the level-0 floor effect change the picture?
    sensitivity = robustness.sensitivity_excluding_baseline(long_df)
    sensitivity_repeated_df = sensitivity["repeated_df"]
    sensitivity_repeated_df.to_csv(REPORTS_DIR / "repeated_holdout_results_excluding_baseline.csv", index=False)
    metrics["sensitivity_excluding_baseline"] = sensitivity["metrics"]

    primary_conclusion = metrics["physiology_value_assessment"]["overall_conclusion"]
    sensitivity_conclusion = metrics["sensitivity_excluding_baseline"]["physiology_value_assessment"]["overall_conclusion"]
    primary_frac = metrics["physiology_value_assessment"]["repeated_participant_holdout"][
        "paired_difference_context_minus_combined"
    ]["fraction_splits_combined_better"]
    sensitivity_frac = metrics["sensitivity_excluding_baseline"]["physiology_value_assessment"][
        "repeated_participant_holdout"
    ]["paired_difference_context_minus_combined"]["fraction_splits_combined_better"]
    frac_delta = sensitivity_frac - primary_frac
    metrics["sensitivity_excluding_baseline"]["conclusion_materially_changed_vs_primary"] = bool(
        (primary_conclusion != sensitivity_conclusion) or (abs(frac_delta) > 0.15)
    )
    metrics["sensitivity_excluding_baseline"]["fraction_combined_better_delta_vs_primary"] = float(frac_delta)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    visualization.plot_suds_and_physio_by_level(long_df)
    visualization.plot_dissociation(long_df)
    visualization.plot_paired_mae_difference(
        primary_repeated_df,
        "Stability of the context-vs-combined comparison across 50 held-out participant splits",
        "paired_mae_difference_repeated_holdout.png",
    )
    visualization.plot_baseline_sensitivity_comparison(
        primary_repeated_df, sensitivity_repeated_df, "baseline_sensitivity_comparison.png"
    )

    return metrics


def main() -> None:
    metrics = run_full_pipeline()
    print(json.dumps(metrics["overall_best"], indent=2))
    print(f"Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
