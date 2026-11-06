"""Qt-free unit tests for the pure-Python helpers in playback_helpers.

These tests exercise:

1. ``session_time_range`` — union of result-file timestamps across all
   available metric CSVs, including files that start with leading NULL rows.
2. Zero-column invariant — ``ordered_playback_columns`` keeps every column
   regardless of whether its values happen to be all-zero or all-NaN.
3. ``smooth_y`` — NaN positions in the input must remain NaN in the output
   regardless of the smoothing window width.

No QApplication or Qt widget is instantiated.  The helpers live in
``playback_helpers.py`` which deliberately has no Qt/pyqtgraph imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nucleuskit_toolkit.ui.pages.playback_helpers import (
    COGNITION_COLUMNS,
    ordered_playback_columns,
    session_time_range,
    smooth_y,
)
from reak_core.models import EMOTION_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(path: str, timestamps: list[float], columns: dict[str, list]) -> None:
    """Write a CSV with a Timestamp column plus the provided value columns."""
    data: dict = {"Timestamp": timestamps}
    data.update(columns)
    pd.DataFrame(data).to_csv(path, index=False, na_rep="NULL")


# ---------------------------------------------------------------------------
# 1.  session_time_range
# ---------------------------------------------------------------------------


class TestSessionTimeRange:
    """session_time_range returns the union of all result-file timestamp spans."""

    def test_single_file_basic(self, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        _write_csv(
            str(results / "Emotions.csv"),
            timestamps=[0.0, 0.5, 1.0, 1.5],
            columns={c: [float("nan")] * 4 for c in EMOTION_COLUMNS},
        )
        span = session_time_range(str(tmp_path))
        assert span is not None
        lo, hi = span
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(1.5)

    def test_leading_null_rows_still_counted(self, tmp_path):
        """Even when all metric values are NULL the timestamps must be included."""
        results = tmp_path / "results"
        results.mkdir()
        # Timestamps 0..4.5 s — the first 5 rows are all-NULL (common for
        # emotion files where the recording starts before the first valid EXG
        # window).
        n = 10
        vals = [float("nan")] * 5 + [0.3, 0.7, 0.5, 0.4, 0.6]
        _write_csv(
            str(results / "Emotions.csv"),
            timestamps=[i * 0.5 for i in range(n)],
            columns={c: vals for c in EMOTION_COLUMNS},
        )
        span = session_time_range(str(tmp_path))
        assert span is not None
        lo, hi = span
        # t_min must be 0.0, not the first finite-valued timestamp (2.0 s)
        assert lo == pytest.approx(0.0), "leading-null timestamps must not be clipped"
        assert hi == pytest.approx(4.5)

    def test_union_across_multiple_files(self, tmp_path):
        """The returned span is the union of all available result files."""
        results = tmp_path / "results"
        results.mkdir()
        # Emotions: 0 .. 4 s
        _write_csv(
            str(results / "Emotions.csv"),
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            columns={c: [float("nan")] * 9 for c in EMOTION_COLUMNS},
        )
        # HeartDynamics: 0 .. 5 s (longer — HRV window covers more time)
        _write_csv(
            str(results / "HeartDynamics.csv"),
            timestamps=[i * 0.5 for i in range(11)],
            columns={
                "mean_hr": [float("nan")] * 11,
                "mean_nn": [float("nan")] * 11,
                "sdnn":    [float("nan")] * 11,
                "rmssd":   [float("nan")] * 11,
                "pnn50":   [float("nan")] * 11,
                "cvsd":    [float("nan")] * 11,
                "cvnn":    [float("nan")] * 11,
                "n_beats": list(range(11)),
            },
        )
        span = session_time_range(str(tmp_path))
        assert span is not None
        lo, hi = span
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(5.0), "span must cover the longest file"

    def test_missing_file_ignored(self, tmp_path):
        """Files that do not exist are silently skipped."""
        results = tmp_path / "results"
        results.mkdir()
        _write_csv(
            str(results / "Arousal.csv"),
            timestamps=[1.0, 1.5, 2.0],
            columns={"TonicEDA": [0.1, 0.2, 0.3], "PhasicEDA": [0.0, 0.1, 0.0]},
        )
        # Only Arousal.csv is present; the rest are missing.
        span = session_time_range(str(tmp_path))
        assert span is not None
        lo, hi = span
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(2.0)

    def test_no_files_returns_none(self, tmp_path):
        (tmp_path / "results").mkdir()
        assert session_time_range(str(tmp_path)) is None

    def test_empty_results_dir_returns_none(self, tmp_path):
        assert session_time_range(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# 2.  Zero-column invariant
# ---------------------------------------------------------------------------


class TestZeroColumns:
    """Columns whose values are all-zero must NOT be filtered from plot_cols."""

    def test_all_emotion_columns_kept_when_some_are_all_zero(self):
        """ordered_playback_columns preserves all-zero emotion columns."""
        # Simulate the case where only Neutral ever fires (one-hot 2.6.0):
        # every other emotion column is all 0.
        result = ordered_playback_columns(EMOTION_COLUMNS)
        assert set(result) == set(EMOTION_COLUMNS), (
            "all emotion columns must appear in plot_cols even when they are all-zero"
        )

    def test_neutral_is_first(self):
        result = ordered_playback_columns(EMOTION_COLUMNS)
        assert result[0] == "Neutral"

    def test_all_zero_column_is_finite_for_y_range(self):
        """A column of zeros is finite, so it is included in any isfinite-based
        Y-range calculation and must not be excluded from the plot."""
        vals = np.zeros(20, dtype=np.float64)
        assert np.isfinite(vals).all(), "zeros must be finite (not excluded by isfinite filter)"
        assert vals.max() == 0.0

    def test_all_nan_column_kept_in_order(self):
        """An all-NaN column still shows up in ordered_playback_columns."""
        # If the session had "Happiness" but all its values were NaN, the column
        # name IS in EMOTION_COLUMNS so it must appear in the result.
        cols = ["Neutral", "Happiness"]
        result = ordered_playback_columns(cols)
        assert "Neutral" in result
        assert "Happiness" in result

    def test_cognition_columns_allowlist(self):
        """COGNITION_COLUMNS contains at least 'Engagement'."""
        assert "Engagement" in COGNITION_COLUMNS


# ---------------------------------------------------------------------------
# 3.  smooth_y — NaN positions preserved
# ---------------------------------------------------------------------------


class TestSmoothY:
    """smooth_y must never fill positions that were NaN in the input."""

    def test_no_smoothing_returns_input_unchanged(self):
        y = np.array([float("nan"), float("nan"), 0.5, 0.8, float("nan")])
        result = smooth_y(y, k=1)
        np.testing.assert_array_equal(np.isnan(result), np.isnan(y))
        assert result[2] == pytest.approx(0.5)

    def test_leading_nan_stays_nan_after_smoothing(self):
        """Leading NaN rows must not be filled even when k is large."""
        y = np.array([float("nan")] * 5 + [0.3, 0.5, 0.7, 0.6, 0.4])
        result = smooth_y(y, k=5)
        # The first 5 positions were NaN — they must stay NaN
        for i in range(5):
            assert np.isnan(result[i]), f"position {i} should remain NaN"
        # Finite positions should produce a smoothed (finite) value
        for i in range(5, len(y)):
            assert np.isfinite(result[i]), f"position {i} should be finite after smoothing"

    def test_trailing_nan_stays_nan_after_smoothing(self):
        y = np.array([0.3, 0.5, 0.7, float("nan"), float("nan")])
        result = smooth_y(y, k=3)
        assert np.isfinite(result[0])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_interior_nan_stays_nan_after_smoothing(self):
        y = np.array([0.2, 0.4, float("nan"), 0.6, 0.8])
        result = smooth_y(y, k=3)
        assert np.isnan(result[2])
        # Neighbours of the gap get their rolling mean from available samples
        assert np.isfinite(result[1])
        assert np.isfinite(result[3])

    def test_all_nan_stays_all_nan(self):
        y = np.full(8, float("nan"))
        result = smooth_y(y, k=5)
        assert np.isnan(result).all()

    def test_no_nan_smoothing_is_centered_mean(self):
        """Without NaN, the output should be a standard centered rolling mean."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = smooth_y(y, k=3)
        expected = pd.Series(y).rolling(3, center=True, min_periods=1).mean().to_numpy()
        np.testing.assert_allclose(result, expected)
