from pathlib import Path

import pytest

from vr_fear_prediction.data import PARTICIPANT_COLUMN, TARGET_COLUMN, build_long_dataset
from vr_fear_prediction.features import FEATURE_SETS, build_feature_matrix, get_feature_columns

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "summary_fixture.csv"


@pytest.fixture
def long_df():
    df, _ = build_long_dataset(FIXTURE_PATH)
    return df


@pytest.mark.parametrize("feature_set", list(FEATURE_SETS))
def test_build_feature_matrix_excludes_target(long_df, feature_set):
    X, y, groups = build_feature_matrix(long_df, feature_set)
    assert TARGET_COLUMN not in X.columns
    assert PARTICIPANT_COLUMN not in X.columns
    assert len(X) == len(y) == len(groups)


def test_build_feature_matrix_unknown_feature_set_raises(long_df):
    with pytest.raises(ValueError, match="Unknown feature set"):
        build_feature_matrix(long_df, "not_a_real_feature_set")


def test_context_only_excludes_physiology_columns(long_df):
    X, _, _ = build_feature_matrix(long_df, "context_only")
    physio_only_numeric, _ = get_feature_columns("physiology_only")
    assert not any(col in X.columns for col in physio_only_numeric)


def test_physiology_only_excludes_context_columns(long_df):
    X, _, _ = build_feature_matrix(long_df, "physiology_only")
    context_numeric, context_categorical = get_feature_columns("context_only")
    assert not any(col in X.columns for col in context_numeric + context_categorical)


def test_combined_is_union_of_context_and_physiology(long_df):
    X_combined, _, _ = build_feature_matrix(long_df, "combined")
    X_context, _, _ = build_feature_matrix(long_df, "context_only")
    X_physio, _, _ = build_feature_matrix(long_df, "physiology_only")
    assert set(X_context.columns) | set(X_physio.columns) == set(X_combined.columns)


def test_feature_sets_are_disjoint_between_context_and_physiology():
    context_numeric, context_categorical = get_feature_columns("context_only")
    physio_numeric, _ = get_feature_columns("physiology_only")
    overlap = (set(context_numeric) | set(context_categorical)) & set(physio_numeric)
    assert overlap == set()
