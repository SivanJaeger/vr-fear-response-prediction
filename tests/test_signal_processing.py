import numpy as np
import pandas as pd
import pytest

from vr_fear_prediction.signal_processing import (
    MIN_BEATS_FOR_HRV,
    MIN_WINDOW_SECONDS_FOR_HRV,
    SignalWindow,
    baseline_correct,
    extract_cardiac_features,
    extract_eda_features,
    extract_level_window,
    extract_respiration_rate,
)

SAMPLING_RATE = 100


def _timestamps(n_samples: int, fs: float = SAMPLING_RATE) -> np.ndarray:
    dt_us = 1e6 / fs
    return np.arange(n_samples) * dt_us


@pytest.fixture
def long_eda_window():
    import neurokit2 as nk

    duration_s = 60
    signal = nk.eda_simulate(duration=duration_s, sampling_rate=SAMPLING_RATE, scr_number=5, random_state=0)
    return SignalWindow(timestamps_us=_timestamps(len(signal)), eda=signal)


@pytest.fixture
def long_ppg_window():
    import neurokit2 as nk

    duration_s = 60
    signal = nk.ppg_simulate(duration=duration_s, sampling_rate=SAMPLING_RATE, heart_rate=75, random_state=0)
    return SignalWindow(timestamps_us=_timestamps(len(signal)), ppg=signal)


@pytest.fixture
def short_ppg_window():
    import neurokit2 as nk

    duration_s = 5
    signal = nk.ppg_simulate(duration=duration_s, sampling_rate=SAMPLING_RATE, heart_rate=75, random_state=0)
    return SignalWindow(timestamps_us=_timestamps(len(signal)), ppg=signal)


@pytest.fixture
def long_resp_window():
    import neurokit2 as nk

    duration_s = 30
    signal = nk.rsp_simulate(duration=duration_s, sampling_rate=SAMPLING_RATE, respiratory_rate=15, random_state=0)
    return SignalWindow(timestamps_us=_timestamps(len(signal)), resp=signal)


def test_signal_window_duration_and_sampling_rate():
    window = SignalWindow(timestamps_us=_timestamps(1000))
    assert window.duration_seconds == pytest.approx(9.99, abs=0.01)
    assert window.sampling_rate_hz == pytest.approx(SAMPLING_RATE, rel=0.01)


def test_signal_window_handles_too_few_samples():
    window = SignalWindow(timestamps_us=np.array([1000.0]))
    assert window.duration_seconds == 0.0
    assert np.isnan(window.sampling_rate_hz)


def test_extract_level_window_selects_correct_rows():
    df = pd.DataFrame(
        {
            "Timestamp": _timestamps(30),
            "Level": [0] * 10 + [1] * 10 + [2] * 10,
            "EDA": np.arange(30, dtype=float),
            "PPG": np.arange(30, dtype=float),
            "RESP": np.arange(30, dtype=float),
        }
    )
    window = extract_level_window(df, level=1)
    assert len(window.timestamps_us) == 10
    assert window.eda.tolist() == list(range(10, 20))


def test_extract_level_window_raises_on_missing_level():
    df = pd.DataFrame({"Timestamp": _timestamps(5), "Level": [0] * 5, "EDA": np.zeros(5), "PPG": np.zeros(5), "RESP": np.zeros(5)})
    with pytest.raises(ValueError, match="No rows found"):
        extract_level_window(df, level=7)


def test_extract_eda_features_on_simulated_signal(long_eda_window):
    features = extract_eda_features(long_eda_window)
    assert features["eda_scr_count"] >= 1
    assert not np.isnan(features["eda_tonic_mean"])
    assert features["eda_phasic_auc"] >= 0


def test_extract_eda_features_empty_on_too_short_window():
    window = SignalWindow(timestamps_us=_timestamps(2), eda=np.array([1.0, 2.0]))
    features = extract_eda_features(window)
    assert np.isnan(features["eda_tonic_mean"])


def test_extract_cardiac_features_hr_in_plausible_range(long_ppg_window):
    features = extract_cardiac_features(long_ppg_window)
    assert 50 <= features["cardiac_hr_mean"] <= 110
    assert features["cardiac_n_beats"] > 0


def test_extract_cardiac_features_rmssd_computed_for_long_enough_window(long_ppg_window):
    assert long_ppg_window.duration_seconds >= MIN_WINDOW_SECONDS_FOR_HRV
    features = extract_cardiac_features(long_ppg_window)
    assert features["cardiac_n_beats"] >= MIN_BEATS_FOR_HRV
    assert not np.isnan(features["cardiac_rmssd"])
    assert features["cardiac_rmssd"] > 0


def test_extract_cardiac_features_rmssd_withheld_for_short_window(short_ppg_window):
    assert short_ppg_window.duration_seconds < MIN_WINDOW_SECONDS_FOR_HRV
    features = extract_cardiac_features(short_ppg_window)
    assert np.isnan(features["cardiac_rmssd"])
    # HR mean can still be estimated even when the window is too short for HRV
    assert not np.isnan(features["cardiac_hr_mean"])


def test_extract_respiration_rate_in_plausible_range(long_resp_window):
    rate = extract_respiration_rate(long_resp_window)
    assert 5 <= rate <= 30


def test_extract_respiration_rate_nan_when_missing():
    window = SignalWindow(timestamps_us=_timestamps(10))
    assert np.isnan(extract_respiration_rate(window))


def test_baseline_correct():
    assert baseline_correct(80.0, 70.0) == 10.0
    assert baseline_correct(70.0, 70.0) == 0.0
