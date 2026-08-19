"""Raw physiological signal processing (EDA, PPG, respiration) using NeuroKit2.

The main modeling pipeline (see data.py / features.py) uses the dataset authors'
precomputed per-level physiological features from summary.csv, which is the honest choice
given that the raw per-participant recordings are ~9 GB in total (150 MB-1.2 GB per
participant) and cannot reasonably be processed in full for a small student project or CI.

This module instead demonstrates a from-scratch feature-extraction pipeline over raw
signal windows, used in the exploration notebook to (a) show real physiological
signal-processing skills and (b) sanity-check the precomputed features against an
independent extraction for one participant. It operates on arbitrary (timestamps, signal)
windows, so it is exercised in tests with small synthetic fixtures rather than the full
dataset.

Column semantics (from data/raw/pX_description.txt):
- Timestamp: Unix time in microseconds
- EDA, PPG, RESP: raw sensor signal values (arbitrary sensor units, not pre-scaled)
- Level: identifier of the active VR exposure level (0 = baseline, 1-9 = exposure tasks)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Below this many seconds, RMSSD and other short-term HRV estimates are not considered
# reliable (established HRV guidance recommends at least ~60s of data; even short-window
# HRV work rarely trusts windows under this length). We refuse to compute HRV metrics on
# shorter windows rather than silently return a noisy number.
MIN_WINDOW_SECONDS_FOR_HRV = 30.0
MIN_BEATS_FOR_HRV = 15


@dataclass
class SignalWindow:
    """A single time-aligned physiological recording segment."""

    timestamps_us: np.ndarray  # Unix microseconds
    eda: np.ndarray | None = None
    ppg: np.ndarray | None = None
    resp: np.ndarray | None = None

    @property
    def duration_seconds(self) -> float:
        if len(self.timestamps_us) < 2:
            return 0.0
        return float(self.timestamps_us[-1] - self.timestamps_us[0]) / 1e6

    @property
    def sampling_rate_hz(self) -> float:
        """Estimated sampling rate from median inter-sample interval."""
        if len(self.timestamps_us) < 2:
            return float("nan")
        diffs = np.diff(self.timestamps_us.astype(np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return float("nan")
        return 1e6 / np.median(diffs)


def load_raw_participant_signal(path) -> pd.DataFrame:
    """Load one participant's raw recording (semicolon-delimited, UTF-8 BOM, two
    'Timestamp' columns where the first is authoritative Unix-microsecond time).
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    # The raw files have a duplicated 'Timestamp' column name (Unix us, then a relative
    # seconds counter); pandas disambiguates the second occurrence as 'Timestamp.1'.
    return df


def extract_level_window(df: pd.DataFrame, level: int) -> SignalWindow:
    """Extract a contiguous SignalWindow for one VR exposure level from a raw recording."""
    subset = df[df["Level"] == level]
    if subset.empty:
        raise ValueError(f"No rows found for level {level}")
    return SignalWindow(
        timestamps_us=subset["Timestamp"].to_numpy(dtype=np.float64),
        eda=subset["EDA"].to_numpy(dtype=np.float64) if "EDA" in subset else None,
        ppg=subset["PPG"].to_numpy(dtype=np.float64) if "PPG" in subset else None,
        resp=subset["RESP"].to_numpy(dtype=np.float64) if "RESP" in subset else None,
    )


def extract_eda_features(window: SignalWindow) -> dict[str, float]:
    """Tonic/phasic EDA features via NeuroKit2's standard EDA processing pipeline.

    Returns NaN-filled features (rather than raising) if the window is too short to
    process, since exploratory notebook use may hit edge-case windows.
    """
    import neurokit2 as nk

    fs = window.sampling_rate_hz
    if window.eda is None or len(window.eda) < 4 or not np.isfinite(fs) or fs <= 0:
        return _empty_eda_features()

    try:
        signals, info = nk.eda_process(window.eda, sampling_rate=fs)
    except Exception:
        return _empty_eda_features()

    tonic = signals["EDA_Tonic"].to_numpy()
    phasic = signals["EDA_Phasic"].to_numpy()
    scr_peaks = info.get("SCR_Peaks", np.array([]))
    scr_amplitude = signals["SCR_Amplitude"].to_numpy() if "SCR_Amplitude" in signals else np.array([])
    amplitudes_at_peaks = scr_amplitude[scr_peaks] if len(scr_peaks) else np.array([])

    return {
        "eda_tonic_mean": float(np.mean(tonic)),
        "eda_tonic_std": float(np.std(tonic)),
        "eda_scr_count": float(len(scr_peaks)),
        "eda_scr_amplitude_mean": float(np.mean(amplitudes_at_peaks)) if len(amplitudes_at_peaks) else 0.0,
        "eda_scr_amplitude_max": float(np.max(amplitudes_at_peaks)) if len(amplitudes_at_peaks) else 0.0,
        "eda_phasic_auc": float(np.trapezoid(np.clip(phasic, 0, None))),
    }


def _empty_eda_features() -> dict[str, float]:
    return {
        "eda_tonic_mean": float("nan"),
        "eda_tonic_std": float("nan"),
        "eda_scr_count": float("nan"),
        "eda_scr_amplitude_mean": float("nan"),
        "eda_scr_amplitude_max": float("nan"),
        "eda_phasic_auc": float("nan"),
    }


def extract_cardiac_features(window: SignalWindow) -> dict[str, float]:
    """Heart-rate and HRV features from a raw PPG window via NeuroKit2 peak detection.

    RMSSD is only computed when the window is long enough and contains enough detected
    beats to be a defensible estimate (see MIN_WINDOW_SECONDS_FOR_HRV); otherwise it is
    returned as NaN with the reason implicit in the accompanying duration/beat count.
    """
    import neurokit2 as nk

    fs = window.sampling_rate_hz
    result: dict[str, float] = {
        "cardiac_hr_mean": float("nan"),
        "cardiac_hr_std": float("nan"),
        "cardiac_rmssd": float("nan"),
        "cardiac_n_beats": 0.0,
        "cardiac_window_seconds": window.duration_seconds,
    }
    if window.ppg is None or len(window.ppg) < 4 or not np.isfinite(fs) or fs <= 0:
        return result

    try:
        cleaned = nk.ppg_clean(window.ppg, sampling_rate=fs)
        _, info = nk.ppg_peaks(cleaned, sampling_rate=fs)
        peaks = info["PPG_Peaks"]
    except Exception:
        return result

    result["cardiac_n_beats"] = float(len(peaks))
    if len(peaks) < 2:
        return result

    ibi_ms = np.diff(peaks) / fs * 1000.0
    hr_bpm = 60000.0 / ibi_ms
    result["cardiac_hr_mean"] = float(np.mean(hr_bpm))
    result["cardiac_hr_std"] = float(np.std(hr_bpm))

    if window.duration_seconds >= MIN_WINDOW_SECONDS_FOR_HRV and len(peaks) >= MIN_BEATS_FOR_HRV:
        successive_diffs = np.diff(ibi_ms)
        result["cardiac_rmssd"] = float(np.sqrt(np.mean(successive_diffs**2)))

    return result


def extract_respiration_rate(window: SignalWindow) -> float:
    """Breaths-per-minute from a raw respiration window via NeuroKit2 peak detection."""
    import neurokit2 as nk

    fs = window.sampling_rate_hz
    if window.resp is None or len(window.resp) < 4 or not np.isfinite(fs) or fs <= 0:
        return float("nan")
    try:
        cleaned = nk.rsp_clean(window.resp, sampling_rate=fs)
        _, info = nk.rsp_peaks(cleaned, sampling_rate=fs)
        n_breaths = len(info["RSP_Peaks"])
    except Exception:
        return float("nan")
    if window.duration_seconds <= 0:
        return float("nan")
    return float(n_breaths / window.duration_seconds * 60.0)


def baseline_correct(value: float, baseline_value: float) -> float:
    """Simple baseline correction: change relative to the participant's own level-0 value."""
    return value - baseline_value


def extract_all_features(window: SignalWindow) -> dict[str, float]:
    """Convenience wrapper combining EDA, cardiac, and respiration feature extraction."""
    features = {}
    features.update(extract_eda_features(window))
    features.update(extract_cardiac_features(window))
    features["resp_rate_bpm"] = extract_respiration_rate(window)
    return features
