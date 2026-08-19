"""Regression metrics, interpretation, and simple repeated-measures correlation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def permutation_importance_on_holdout(
    pipeline: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, n_repeats: int = 50, random_state: int = RANDOM_SEED
) -> pd.DataFrame:
    """Permutation importance computed on held-out participants only (never on train)."""
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=random_state, scoring="neg_mean_absolute_error"
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def standardized_ridge_coefficients(pipeline: BaseEstimator) -> pd.DataFrame:
    """Extract standardized coefficients from a fitted Pipeline(preprocess, Ridge(...)).

    Valid because the preprocessing step already standardizes numeric features and
    one-hot encodes categoricals, so raw Ridge coefficients are already on a standardized
    scale for numeric inputs (one-hot dummy coefficients are on a 0/1 scale by nature).
    """
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .reset_index(drop=True)
    )


def repeated_measures_correlation(df: pd.DataFrame, participant_col: str, x_col: str, y_col: str) -> dict[str, float]:
    """Approximate repeated-measures correlation (Bakdash & Marusich, 2017).

    Removes each participant's mean from both variables before correlating, so the
    result reflects the within-participant association between x and y rather than being
    driven by between-participant differences (e.g. some participants generally reporting
    higher fear and having generally higher heart rate, regardless of exposure stage).

    This is a simplified version: the original method also adjusts degrees of freedom via
    an ANCOVA-style correction. We report the demeaned Pearson r and note the
    approximation in the notebook.
    """
    from scipy.stats import pearsonr

    centered = df.copy()
    centered[x_col] = centered.groupby(participant_col)[x_col].transform(lambda s: s - s.mean())
    centered[y_col] = centered.groupby(participant_col)[y_col].transform(lambda s: s - s.mean())
    valid = centered[[x_col, y_col]].dropna()
    r, p = pearsonr(valid[x_col], valid[y_col])
    return {"r": float(r), "p_value": float(p), "n_observations": int(len(valid))}
