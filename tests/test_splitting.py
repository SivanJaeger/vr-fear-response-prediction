import numpy as np
import pandas as pd
import pytest

from vr_fear_prediction.splitting import (
    assert_no_participant_overlap,
    held_out_participant_split,
    participant_group_kfold,
)


@pytest.fixture
def grouped_data():
    n_participants = 10
    rows_per_participant = 10
    rng = np.random.default_rng(0)
    groups = np.repeat([f"P{i:02d}" for i in range(n_participants)], rows_per_participant)
    X = pd.DataFrame({"feature": rng.normal(size=len(groups))})
    y = pd.Series(rng.normal(size=len(groups)))
    groups = pd.Series(groups)
    return X, y, groups


def test_held_out_participant_split_has_no_overlap(grouped_data):
    X, y, groups = grouped_data
    train_idx, test_idx = held_out_participant_split(X, y, groups, test_size=0.3)
    train_participants = set(groups.iloc[train_idx])
    test_participants = set(groups.iloc[test_idx])
    assert train_participants.isdisjoint(test_participants)


def test_held_out_participant_split_covers_all_rows(grouped_data):
    X, y, groups = grouped_data
    train_idx, test_idx = held_out_participant_split(X, y, groups, test_size=0.3)
    assert len(train_idx) + len(test_idx) == len(X)
    assert set(train_idx) | set(test_idx) == set(range(len(X)))


def test_assert_no_participant_overlap_raises_on_overlap():
    groups = pd.Series(["A", "A", "B", "B", "C", "C"])
    train_idx = np.array([0, 1, 2])  # A, A, B
    test_idx = np.array([3, 4, 5])  # B, C, C -- B leaks into both sets
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_participant_overlap(groups, train_idx, test_idx)


def test_assert_no_participant_overlap_passes_when_clean():
    groups = pd.Series(["A", "A", "B", "B", "C", "C"])
    train_idx = np.array([0, 1, 2, 3])  # A, A, B, B
    test_idx = np.array([4, 5])  # C, C
    assert_no_participant_overlap(groups, train_idx, test_idx) is None


def test_participant_group_kfold_splits_have_no_overlap(grouped_data):
    X, y, groups = grouped_data
    cv = participant_group_kfold(n_splits=5)
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        assert_no_participant_overlap(groups, train_idx, test_idx)


def test_participant_group_kfold_uses_all_participants_as_test_exactly_once(grouped_data):
    X, y, groups = grouped_data
    cv = participant_group_kfold(n_splits=5)
    seen_test_participants = set()
    for _, test_idx in cv.split(X, y, groups=groups):
        seen_test_participants |= set(groups.iloc[test_idx])
    assert seen_test_participants == set(groups.unique())
