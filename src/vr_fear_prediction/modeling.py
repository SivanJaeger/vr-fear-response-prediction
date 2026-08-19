"""Train and evaluate context-only, physiology-only, and combined models.

For every feature set, compares a DummyRegressor baseline against regularized linear
regression (Ridge) and a RandomForestRegressor, using participant-aware GroupKFold cross-
validation for model/hyperparameter comparison and a held-out group of participants for a
final generalization check. All preprocessing (imputation, scaling, one-hot encoding) is
fit inside a scikit-learn Pipeline so it only ever sees training folds.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
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
MODELS_DIR = PROJECT_ROOT / "models"
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


def run_full_pipeline() -> dict:
    """Train/evaluate context-only, physiology-only, and combined models and write
    reports/metrics.json + figures + the selected fitted pipeline.
    """
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

    fitted_pipelines: dict[str, dict[str, Pipeline]] = {}
    held_out_predictions: dict[str, dict] = {}

    for feature_set in FEATURE_SETS:
        X, y, groups = build_feature_matrix(long_df, feature_set)
        numeric_cols, categorical_cols = FEATURE_SETS[feature_set]["numeric"], FEATURE_SETS[feature_set]["categorical"]

        train_idx, test_idx = held_out_participant_split(X, y, groups)
        assert_no_participant_overlap(groups, train_idx, test_idx)

        X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        metrics["feature_sets"][feature_set] = {}
        fitted_pipelines[feature_set] = {}

        for model_name, (estimator, param_grid) in _model_registry().items():
            pipeline, result = fit_and_evaluate(
                X_train, y_train, groups_train, X_test, y_test, numeric_cols, categorical_cols, model_name, estimator, param_grid
            )
            metrics["feature_sets"][feature_set][model_name] = result
            fitted_pipelines[feature_set][model_name] = pipeline

        best_model_name = min(
            metrics["feature_sets"][feature_set], key=lambda m: metrics["feature_sets"][feature_set][m]["cv"]["MAE_mean"]
        )
        metrics["feature_sets"][feature_set]["best_model"] = best_model_name
        held_out_predictions[feature_set] = {
            "y_true": y_test.to_numpy(),
            "y_pred": fitted_pipelines[feature_set][best_model_name].predict(X_test),
            "pipeline": fitted_pipelines[feature_set][best_model_name],
            "X_test": X_test,
            "y_test": y_test,
        }

    overall_best_feature_set = min(
        FEATURE_SETS, key=lambda fs: metrics["feature_sets"][fs][metrics["feature_sets"][fs]["best_model"]]["cv"]["MAE_mean"]
    )
    overall_best_model = metrics["feature_sets"][overall_best_feature_set]["best_model"]
    metrics["overall_best"] = {"feature_set": overall_best_feature_set, "model": overall_best_model}

    context_best_mae = metrics["feature_sets"]["context_only"][metrics["feature_sets"]["context_only"]["best_model"]]["held_out_test"]["MAE"]
    combined_best_mae = metrics["feature_sets"]["combined"][metrics["feature_sets"]["combined"]["best_model"]]["held_out_test"]["MAE"]
    metrics["physiology_adds_value"] = {
        "context_only_best_test_MAE": context_best_mae,
        "combined_best_test_MAE": combined_best_mae,
        "combined_improves_on_context": bool(combined_best_mae < context_best_mae),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_pipeline_overall = held_out_predictions[overall_best_feature_set]["pipeline"]
    joblib.dump(best_pipeline_overall, MODELS_DIR / "best_pipeline.joblib")
    saved_pipeline_path = MODELS_DIR / "best_pipeline.joblib"
    try:
        metrics["saved_pipeline"] = str(saved_pipeline_path.relative_to(PROJECT_ROOT))
    except ValueError:
        metrics["saved_pipeline"] = str(saved_pipeline_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    ridge_coefs = evaluation.standardized_ridge_coefficients(fitted_pipelines["combined"]["ridge"])
    ridge_coefs.to_csv(REPORTS_DIR / "ridge_coefficients_combined.csv", index=False)

    _generate_figures(long_df, metrics, held_out_predictions, overall_best_feature_set)

    return metrics


def _generate_figures(long_df: pd.DataFrame, metrics: dict, held_out_predictions: dict, best_feature_set: str) -> None:
    visualization.plot_suds_and_physio_by_level(long_df)
    visualization.plot_dissociation(long_df)
    visualization.plot_feature_set_comparison(metrics)

    # Diagnostics are generated both for the CV-selected overall-best feature set and for
    # "combined" specifically, since combined is the feature set most relevant to the
    # primary research question (does physiology add value beyond context alone) even
    # when it is not the CV-selected winner overall (see README "A note on model
    # selection instability").
    feature_sets_to_plot = {best_feature_set, "combined"}
    for feature_set in feature_sets_to_plot:
        result = held_out_predictions[feature_set]
        suffix = "" if feature_set == best_feature_set else f"_{feature_set}"
        visualization.plot_predicted_vs_observed(
            result["y_true"],
            result["y_pred"],
            f"Predicted vs. observed SUDS ({feature_set}, held-out participants)",
            f"predicted_vs_observed{suffix}.png",
        )
        visualization.plot_residuals(
            result["y_true"], result["y_pred"], f"Residuals ({feature_set}, held-out participants)", f"residuals{suffix}.png"
        )
        importance_df = evaluation.permutation_importance_on_holdout(result["pipeline"], result["X_test"], result["y_test"])
        visualization.plot_permutation_importance(
            importance_df, f"Permutation importance ({feature_set}, held-out participants)", f"permutation_importance{suffix}.png"
        )
        importance_df.to_csv(REPORTS_DIR / f"permutation_importance{suffix}.csv", index=False)


def main() -> None:
    metrics = run_full_pipeline()
    print(json.dumps(metrics["overall_best"], indent=2))
    print(f"Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
