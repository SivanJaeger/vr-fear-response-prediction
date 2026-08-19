from pathlib import Path

import pandas as pd
import pytest

from vr_fear_prediction.data import (
    LEAKAGE_EXCLUDED_COLUMNS,
    LEVELS,
    PARTICIPANT_COLUMN,
    TARGET_COLUMN,
    build_long_dataset,
    load_raw_summary,
    reshape_wide_to_long,
    validate_raw_summary,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "summary_fixture.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return load_raw_summary(FIXTURE_PATH)


def test_load_raw_summary_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw_summary(tmp_path / "does_not_exist.csv")


def test_validate_raw_summary_passes_on_clean_fixture(raw_df):
    report = validate_raw_summary(raw_df)
    assert report.is_valid
    assert report.missing_columns == []
    assert report.duplicate_participants == []
    assert report.out_of_range_suds == []


def test_validate_raw_summary_detects_duplicate_participants(raw_df):
    duplicated = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    report = validate_raw_summary(duplicated)
    assert not report.is_valid
    assert "P01" in report.duplicate_participants


def test_validate_raw_summary_detects_out_of_range_suds(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "suds_3"] = 99
    report = validate_raw_summary(bad)
    assert not report.is_valid
    assert any(col == "suds_3" for _, col, _ in report.out_of_range_suds)


def test_validate_raw_summary_detects_missing_columns(raw_df):
    dropped = raw_df.drop(columns=["age"])
    report = validate_raw_summary(dropped)
    assert "age" in report.missing_columns
    with pytest.raises(ValueError, match="missing required columns"):
        report.raise_if_invalid()


def test_reshape_wide_to_long_row_count(raw_df):
    long_df = reshape_wide_to_long(raw_df)
    assert len(long_df) == raw_df["participant"].nunique() * len(LEVELS)


def test_reshape_wide_to_long_participant_id_normalized(raw_df):
    long_df = reshape_wide_to_long(raw_df)
    assert set(long_df[PARTICIPANT_COLUMN].unique()) == set(raw_df["participant"].unique())


def test_reshape_wide_to_long_target_matches_source_suds_columns(raw_df):
    long_df = reshape_wide_to_long(raw_df)
    for _, row in raw_df.iterrows():
        for lvl in LEVELS:
            expected = row[f"suds_{lvl}"]
            actual = long_df.loc[
                (long_df[PARTICIPANT_COLUMN] == f"P{int(row['participant'][1:]):02d}") & (long_df["level"] == lvl),
                TARGET_COLUMN,
            ].iloc[0]
            assert actual == expected


def test_reshape_wide_to_long_baseline_delta_is_zero_at_level_zero(raw_df):
    long_df = reshape_wide_to_long(raw_df)
    baseline_rows = long_df[long_df["level"] == 0]
    assert (baseline_rows["PPG_HR_delta_baseline"] == 0).all()


def test_reshape_wide_to_long_no_participant_has_duplicate_levels(raw_df):
    long_df = reshape_wide_to_long(raw_df)
    counts = long_df.groupby(PARTICIPANT_COLUMN)["level"].apply(lambda s: s.duplicated().sum())
    assert (counts == 0).all()


def test_build_long_dataset_end_to_end():
    long_df, report = build_long_dataset(FIXTURE_PATH)
    assert report.is_valid
    assert len(long_df) == report.n_participants * len(LEVELS)
    assert long_df[TARGET_COLUMN].between(0, 10).all()


def test_leakage_excluded_columns_are_not_in_long_dataset_by_default():
    long_df, _ = build_long_dataset(FIXTURE_PATH)
    leaked_present = [c for c in LEAKAGE_EXCLUDED_COLUMNS if c in long_df.columns]
    assert leaked_present == []
