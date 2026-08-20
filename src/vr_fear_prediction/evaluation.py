"""Regression metrics and a repeated-measures correlation helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def repeated_measures_correlation(df: pd.DataFrame, participant_col: str, x_col: str, y_col: str) -> dict[str, float]:
    """Approximate repeated-measures correlation (Bakdash & Marusich, 2017).

    Removes each participant's mean from both variables before correlating, so the
    result reflects the within-participant association rather than being driven by
    between-participant differences (e.g. some participants generally reporting higher
    fear and having generally higher heart rate, regardless of exposure stage).
    """
    from scipy.stats import pearsonr

    centered = df.copy()
    centered[x_col] = centered.groupby(participant_col)[x_col].transform(lambda s: s - s.mean())
    centered[y_col] = centered.groupby(participant_col)[y_col].transform(lambda s: s - s.mean())
    valid = centered[[x_col, y_col]].dropna()
    r, p = pearsonr(valid[x_col], valid[y_col])
    return {"r": float(r), "p_value": float(p), "n_observations": int(len(valid))}
