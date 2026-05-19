"""
Regression tests for PPG heart-rate processing.

testData/shimmer.csv contains a recording with two distinct amplitude regimes:
  - t = 0– 60 s  : high-amplitude burst (sensor being positioned, AC ~ 2800 a.u.)
  - t = 60–540 s : low-amplitude stable segment (sensor settled, AC ~  15 a.u.)
  - t = 540–630 s: second high-amplitude burst (sensor reposition)
  - t = 630–end  : low-amplitude stable segment

Without amplitude normalisation NeuroKit2's Elgendi algorithm sets its peak
threshold from the global mean of the beat-window moving average.  The burst
region dominates that mean and the threshold ends up ~30 000× too high for the
stable segments, causing 0 peaks to be detected there.  The tests below guard
against that regression.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_DATA = Path(__file__).parent.parent / "testData" / "shimmer.csv"


def _load_and_run(tmp_path: Path):
    """Run the full PPG processing stack on testData/shimmer.csv.

    Returns
    -------
    peak_times : np.ndarray
        Peak positions in seconds (256 Hz domain).
    duration_s : float
        Recording duration in seconds.
    """
    from nucleuskit_pipeline.shimmer.processor.resampler import (
        resample_to_grid,
        normalise_timestamps_to_seconds,
    )
    from nucleuskit_pipeline.shimmer.processor.heart import (
        SAMPLING_RATE,
        _UPSAMPLE_RATE,
        apply_ppg_artifact_rejection,
        _detect_ppg_peaks,
    )

    df  = pd.read_csv(_TEST_DATA, header=None)
    ts  = normalise_timestamps_to_seconds(df[0].values.astype(float))
    ppg = df[5].values.astype(float)

    ts_g, ppg_g, gap_mask = resample_to_grid(ts, ppg, "ppg", str(tmp_path))
    ppg_clean, rej = apply_ppg_artifact_rejection(ppg_g, ts_g, str(tmp_path))
    combined_mask = gap_mask | rej

    _, peak_idx_256, _, _, _ = _detect_ppg_peaks(ppg_clean, gap_mask=combined_mask)

    peak_times = peak_idx_256 / _UPSAMPLE_RATE
    duration_s = len(ppg_g) / SAMPLING_RATE
    return peak_times, duration_s


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ppg_results(tmp_path_factory):
    """Run the pipeline once; share results across all tests in this module."""
    tmp = tmp_path_factory.mktemp("ppg_out")
    peak_times, duration_s = _load_and_run(tmp)
    return peak_times, duration_s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TEST_DATA.exists(), reason="testData/shimmer.csv not found")
class TestPPGPeakDetection:
    """Peak detection must work in every amplitude regime of the test recording."""

    # Minimum acceptable beat-detection rate per region.
    # The burst regions are harder (motion artefacts, sensor settle) so we
    # accept a lower threshold there.
    _MIN_COVERAGE_STABLE = 0.75   # 75 % of expected beats in stable segments
    _MIN_COVERAGE_BURST  = 0.50   # 50 % of expected beats in burst segments

    # Expected heart rate used to compute the expected number of beats.
    _ASSUMED_HR_BPM = 66.0

    def _expected(self, duration_s: float) -> int:
        return max(1, int(duration_s * self._ASSUMED_HR_BPM / 60.0))

    def _count(self, peak_times: np.ndarray, lo: float, hi: float) -> int:
        return int(np.sum((peak_times >= lo) & (peak_times < hi)))

    # ------------------------------------------------------------------

    def test_total_peaks_reasonable(self, ppg_results):
        """Total peak count must be at least 80 % of expected beats."""
        peak_times, duration_s = ppg_results
        exp = self._expected(duration_s)
        assert len(peak_times) >= 0.80 * exp, (
            f"Only {len(peak_times)} peaks found; expected ~{exp}"
        )

    def test_stable_region_1_peaks_detected(self, ppg_results):
        """Stable region 1 (t=60–540 s, low AC amplitude) must yield peaks."""
        peak_times, _ = ppg_results
        lo, hi  = 60.0, 540.0
        n       = self._count(peak_times, lo, hi)
        exp     = self._expected(hi - lo)
        assert n >= self._MIN_COVERAGE_STABLE * exp, (
            f"Stable region 1: only {n} peaks detected (expected ~{exp}). "
            "Rolling amplitude normalisation may not be applied before NK2."
        )

    def test_stable_region_2_peaks_detected(self, ppg_results):
        """Stable region 2 (t=630 s onwards) must yield peaks."""
        peak_times, duration_s = ppg_results
        lo, hi  = 630.0, duration_s
        n       = self._count(peak_times, lo, hi)
        exp     = self._expected(hi - lo)
        assert n >= self._MIN_COVERAGE_STABLE * exp, (
            f"Stable region 2: only {n} peaks detected (expected ~{exp}). "
            "Rolling amplitude normalisation may not be applied before NK2."
        )

    def test_burst_region_1_peaks_detected(self, ppg_results):
        """Burst region 1 (t=0–60 s, high AC amplitude) must yield peaks."""
        peak_times, _ = ppg_results
        lo, hi  = 0.0, 60.0
        n       = self._count(peak_times, lo, hi)
        exp     = self._expected(hi - lo)
        assert n >= self._MIN_COVERAGE_BURST * exp, (
            f"Burst region 1: only {n}/{exp} peaks detected."
        )

    def test_burst_region_2_peaks_detected(self, ppg_results):
        """Burst region 2 (t=540–630 s, high AC amplitude) must yield peaks."""
        peak_times, _ = ppg_results
        lo, hi  = 540.0, 630.0
        n       = self._count(peak_times, lo, hi)
        exp     = self._expected(hi - lo)
        assert n >= self._MIN_COVERAGE_BURST * exp, (
            f"Burst region 2: only {n}/{exp} peaks detected."
        )

    def test_ibi_distribution_physiological(self, ppg_results):
        """At least 95 % of detected IBIs must be within physiological bounds."""
        peak_times, _ = ppg_results
        ibis        = np.diff(peak_times) * 1000.0   # ms
        n_total     = len(ibis)
        n_valid     = int(np.sum((ibis > 300) & (ibis < 2000)))
        assert n_total > 0, "No IBIs computable — too few peaks."
        assert n_valid / n_total >= 0.95, (
            f"Only {n_valid}/{n_total} IBIs are within physiological bounds "
            "(300–2000 ms). Peak detection may be producing many false detections."
        )

    def test_mean_heart_rate_plausible(self, ppg_results):
        """Mean heart rate from valid IBIs must be between 40 and 120 bpm."""
        peak_times, _ = ppg_results
        ibis  = np.diff(peak_times) * 1000.0
        valid = ibis[(ibis > 300) & (ibis < 2000)]
        assert len(valid) > 0, "No valid IBIs — cannot compute mean HR."
        mean_bpm = 60_000.0 / valid.mean()
        assert 40.0 <= mean_bpm <= 120.0, (
            f"Mean HR = {mean_bpm:.1f} bpm is outside the expected 40–120 bpm range."
        )
