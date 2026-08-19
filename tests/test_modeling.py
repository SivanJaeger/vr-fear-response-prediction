import json
from pathlib import Path

import pandas as pd
import pytest

from vr_fear_prediction import modeling, robustness, visualization
from vr_fear_prediction.data import build_long_dataset
from vr_fear_prediction.features import build_feature_matrix
from vr_fear_prediction.splitting import held_out_participant_split

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "summary_fixture.csv"


@pytest.fixture
def long_df():
    df, _ = build_long_dataset(FIXTURE_PATH)
    return df


def test_build_preprocessor_transforms_mixed_columns(long_df):
    X, _, _ = build_feature_matrix(long_df, "combined")
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = modeling.build_preprocessor(numeric_cols, categorical_cols)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] >= len(numeric_cols)  # one-hot expands categorical columns


def test_fit_and_evaluate_dummy_model_produces_expected_result_shape(long_df):
    X, y, groups = build_feature_matrix(long_df, "context_only")
    train_idx, test_idx = held_out_participant_split(X, y, groups, test_size=0.34)
    X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    from sklearn.dummy import DummyRegressor

    pipeline, result = modeling.fit_and_evaluate(
        X_train, y_train, groups_train, X_test, y_test,
        *_context_cols(), "dummy_mean", DummyRegressor(strategy="mean"), {},
    )
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(result.keys()) >= {"cv", "held_out_test", "n_train_observations", "n_test_observations"}
    assert set(result["held_out_test"].keys()) == {"MAE", "RMSE", "R2"}
    assert set(result["cv"].keys()) == {"MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std"}


def test_fit_and_evaluate_ridge_with_grid_search_selects_params(long_df):
    X, y, groups = build_feature_matrix(long_df, "context_only")
    train_idx, test_idx = held_out_participant_split(X, y, groups, test_size=0.34)
    X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    from sklearn.linear_model import Ridge

    pipeline, result = modeling.fit_and_evaluate(
        X_train, y_train, groups_train, X_test, y_test,
        *_context_cols(), "ridge", Ridge(random_state=42), {"model__alpha": [0.1, 10.0]},
    )
    assert "model__alpha" in result["best_params"]
    assert result["best_params"]["model__alpha"] in (0.1, 10.0)
    assert hasattr(pipeline.named_steps["model"], "coef_")


def _context_cols():
    from vr_fear_prediction.features import get_feature_columns

    return get_feature_columns("context_only")


def test_run_full_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(modeling, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(modeling, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(modeling, "METRICS_PATH", tmp_path / "reports" / "metrics.json")
    monkeypatch.setattr(visualization, "FIGURES_DIR", tmp_path / "reports" / "figures")
    # Keep the integration test fast: the repeated-holdout logic itself is covered in
    # depth by test_robustness.py, so a handful of splits is enough here to check wiring.
    monkeypatch.setattr(robustness, "DEFAULT_N_SPLITS", 5)

    import vr_fear_prediction.data as data_module

    monkeypatch.setattr(data_module, "RAW_SUMMARY_PATH", FIXTURE_PATH)

    metrics = modeling.run_full_pipeline()

    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "models" / "best_pipeline.joblib").exists()
    on_disk = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert on_disk["dataset"]["n_participants"] == 6
    for feature_set in ("context_only", "physiology_only", "combined"):
        assert feature_set in metrics["feature_sets"]
        assert "best_model" in metrics["feature_sets"][feature_set]
    assert "overall_best" in metrics
    assert "physiology_value_assessment" in metrics
    assert metrics["physiology_value_assessment"]["overall_conclusion"] in (
        "inconclusive",
        "combined_likely_better",
        "context_only_likely_sufficient",
    )
    assert "sensitivity_excluding_baseline" in metrics
    assert metrics["sensitivity_excluding_baseline"]["levels_included"] == list(range(1, 10))
    assert (tmp_path / "reports" / "repeated_holdout_results.csv").exists()
    assert (tmp_path / "reports" / "repeated_holdout_results_excluding_baseline.csv").exists()
