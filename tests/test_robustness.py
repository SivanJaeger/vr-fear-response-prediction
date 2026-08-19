from pathlib import Path

import pandas as pd
import pytest

from vr_fear_prediction import robustness
from vr_fear_prediction.data import build_long_dataset
from vr_fear_prediction.features import build_feature_matrix

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "summary_fixture.csv"
SMALL_N_SPLITS = 5


@pytest.fixture
def long_df():
    df, _ = build_long_dataset(FIXTURE_PATH)
    return df


def test_repeated_holdout_has_expected_number_of_splits(long_df):
    df = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=500)
    assert len(df) == SMALL_N_SPLITS
    assert set(df["split_id"]) == set(range(SMALL_N_SPLITS))


def test_repeated_holdout_different_n_splits(long_df):
    df = robustness.repeated_participant_holdout(long_df, n_splits=8, test_size=0.34, base_seed=500)
    assert len(df) == 8


def test_repeated_holdout_completes_without_participant_leakage(long_df):
    """repeated_participant_holdout calls assert_no_participant_overlap on every outer
    split internally; if any split leaked a participant into both train and test, this
    call would raise AssertionError and the test would fail before returning a result."""
    df = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=500)
    assert len(df) == SMALL_N_SPLITS
    n_participants = build_feature_matrix(long_df, "context_only")[2].nunique()
    assert (df["n_train_participants"] + df["n_test_participants"] == n_participants).all()


def test_repeated_holdout_uses_identical_groups_for_both_feature_sets(long_df):
    """context_only and combined are built from the same long_df with the same row order,
    so the single (train_idx, test_idx) pair computed per split is applied to identical
    participants for both feature sets by construction. This is the precondition that
    makes the paired MAE difference a fair, same-participants comparison."""
    _, y_context, groups_context = build_feature_matrix(long_df, "context_only")
    _, y_combined, groups_combined = build_feature_matrix(long_df, "combined")
    pd.testing.assert_series_equal(groups_context, groups_combined)
    pd.testing.assert_series_equal(y_context, y_combined)


def test_repeated_holdout_is_reproducible_with_fixed_seeds(long_df):
    df1 = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=777)
    df2 = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=777)
    pd.testing.assert_frame_equal(df1, df2)


def test_repeated_holdout_different_base_seed_can_change_results(long_df):
    df1 = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=1)
    df2 = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=999)
    # Different seeds draw different held-out participants at least some of the time.
    assert not df1["seed"].equals(df2["seed"])


def test_repeated_holdout_output_columns(long_df):
    df = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=500)
    expected = {
        "split_id", "seed", "n_train_participants", "n_test_participants",
        "context_only_MAE", "combined_MAE", "diff_context_minus_combined",
        "context_only_best_alpha", "combined_best_alpha",
    }
    assert expected.issubset(df.columns)
    # diff is defined as context_only_MAE - combined_MAE
    diff_recomputed = df["context_only_MAE"] - df["combined_MAE"]
    pd.testing.assert_series_equal(diff_recomputed, df["diff_context_minus_combined"], check_names=False)


def test_summarize_repeated_holdout_structure(long_df):
    df = robustness.repeated_participant_holdout(long_df, n_splits=SMALL_N_SPLITS, test_size=0.34, base_seed=500)
    summary = robustness.summarize_repeated_holdout(df)
    assert summary["n_splits"] == SMALL_N_SPLITS
    assert 0.0 <= summary["paired_difference_context_minus_combined"]["fraction_splits_combined_better"] <= 1.0
    assert summary["paired_difference_context_minus_combined"]["ci_2.5pct"] <= summary["paired_difference_context_minus_combined"]["ci_97.5pct"]
    for feature_set in ("context_only", "combined"):
        assert {"MAE_mean", "MAE_median", "MAE_std"} <= set(summary[feature_set].keys())


def _fake_repeated_summary(fraction_combined_better: float) -> dict:
    return {
        "n_splits": 50,
        "context_only": {"MAE_mean": 2.0, "MAE_median": 2.0, "MAE_std": 0.5},
        "combined": {"MAE_mean": 2.0, "MAE_median": 2.0, "MAE_std": 0.5},
        "paired_difference_context_minus_combined": {
            "mean": 0.0, "median": 0.0, "std": 0.5,
            "ci_2.5pct": -1.0, "ci_97.5pct": 1.0,
            "fraction_splits_combined_better": fraction_combined_better,
        },
    }


def test_physiology_value_assessment_inconclusive_when_split_and_cv_disagree():
    # Combined wins the original split (0.89 < 1.31) but loses under CV (2.09 > 1.82) --
    # this reproduces the real disagreement this project found on the full dataset.
    result = robustness.build_physiology_value_assessment(
        context_test_mae=1.31, combined_test_mae=0.89,
        context_cv_mae=1.82, combined_cv_mae=2.09,
        repeated_summary=_fake_repeated_summary(0.54),
    )
    assert result["overall_conclusion"] == "inconclusive"
    assert result["original_held_out_split"]["combined_improves_on_context_original_split"] is True
    assert result["participant_aware_cv"]["combined_improves_on_context_cv"] is False


def test_physiology_value_assessment_combined_likely_better_when_all_criteria_agree():
    result = robustness.build_physiology_value_assessment(
        context_test_mae=2.0, combined_test_mae=1.0,
        context_cv_mae=2.0, combined_cv_mae=1.0,
        repeated_summary=_fake_repeated_summary(0.8),
    )
    assert result["overall_conclusion"] == "combined_likely_better"


def test_physiology_value_assessment_context_only_sufficient_when_all_criteria_agree():
    result = robustness.build_physiology_value_assessment(
        context_test_mae=1.0, combined_test_mae=2.0,
        context_cv_mae=1.0, combined_cv_mae=2.0,
        repeated_summary=_fake_repeated_summary(0.1),
    )
    assert result["overall_conclusion"] == "context_only_likely_sufficient"


def test_physiology_value_assessment_inconclusive_when_repeated_holdout_is_borderline():
    # Split and CV agree combined is better, but the repeated-holdout fraction is not
    # decisive (near 50%) -- still inconclusive rather than declaring a winner.
    result = robustness.build_physiology_value_assessment(
        context_test_mae=2.0, combined_test_mae=1.0,
        context_cv_mae=2.0, combined_cv_mae=1.0,
        repeated_summary=_fake_repeated_summary(0.52),
    )
    assert result["overall_conclusion"] == "inconclusive"


def test_sensitivity_excluding_baseline_restricts_to_levels_1_to_9(long_df):
    result = robustness.sensitivity_excluding_baseline(long_df, n_splits=SMALL_N_SPLITS)
    metrics = result["metrics"]
    assert metrics["levels_included"] == list(range(1, 10))
    assert 0 not in metrics["levels_included"]
    assert metrics["n_observations"] == len(long_df[long_df["level"] > 0])
    for feature_set in ("context_only", "physiology_only", "combined"):
        assert feature_set in metrics["feature_sets"]
    assert "physiology_value_assessment" in metrics
    assert len(result["repeated_df"]) == SMALL_N_SPLITS
