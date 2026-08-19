"""Participant-aware train/test splitting and cross-validation.

Every participant in this dataset contributes 10 rows (one per exposure level). A random
row-level split would put the same participant's rows in both train and test, letting the
model implicitly memorize participant identity/baseline rather than learning a
generalizable physiology-fear relationship. All splitting in this project must therefore
group by participant_id.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RANDOM_SEED = 42


def held_out_participant_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) such that no participant appears in both sets."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    assert_no_participant_overlap(groups, train_idx, test_idx)
    return train_idx, test_idx


def participant_group_kfold(n_splits: int = 5) -> GroupKFold:
    """A GroupKFold configured for this project's default number of CV folds."""
    return GroupKFold(n_splits=n_splits)


def assert_no_participant_overlap(groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Raise if any participant id appears in both the train and test index sets."""
    groups_arr = groups.to_numpy() if hasattr(groups, "to_numpy") else np.asarray(groups)
    train_participants = set(groups_arr[train_idx])
    test_participants = set(groups_arr[test_idx])
    overlap = train_participants & test_participants
    if overlap:
        raise AssertionError(f"Participant leakage detected: {overlap} present in both train and test sets")
