"""
Emotion pipeline tests for classical-emotion 2.5.0.

Coverage:
- StreamingWindowSource yields (8, 500) windows with the correct hop
- Null windows (>10 % NaN samples) are excluded from calibration and predict
- Pipeline probabilities land on EMOTION_COLUMNS and sum to 1 for valid windows
- ClassicalEmotionAdapter can load the real pipeline.joblib and return
  EmotionWindowResult objects with the expected structure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import (
    EMOTION_COLUMNS,
    RMS_COLUMNS,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.streaming import (
    StreamingWindowSource,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.model import (
    EmotionModel,
    EmotionWindowResult,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.pipeline import run_window_pipeline
from nucleuskit_toolkit.hermes.processor.emotions_processor.models.classical_emotion import (
    ClassicalEmotionAdapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_CHANNELS = 8
_SAMPLING_RATE = 250
_WINDOW_SAMPLES = 500   # 2.0 s at 250 Hz
_STEP_SAMPLES = 125     # 0.5 s at 250 Hz


def _make_eeg(n_seconds: float = 6.0, nan_rows: slice | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, eeg_data).  Optionally NaN-out a row slice."""
    n = int(n_seconds * _SAMPLING_RATE)
    eeg = _RNG.normal(scale=5.0, size=(n, _N_CHANNELS)).astype(np.float64)
    if nan_rows is not None:
        eeg[nan_rows, :] = np.nan
    ts = np.arange(n, dtype=np.float64) / _SAMPLING_RATE
    return ts, eeg


class _FixedModel(EmotionModel):
    """Minimal mock model: returns equal probabilities for every window."""

    def __init__(self) -> None:
        self._calibrated = False
        self._context_shape: tuple | None = None

    @classmethod
    def load(cls, release_dir=None, **kwargs) -> "_FixedModel":
        return cls()

    def reset(self) -> None:
        self._calibrated = False
        self._context_shape = None

    def set_subject_context(self, windows: np.ndarray) -> None:
        self._calibrated = True
        self._context_shape = windows.shape

    def predict(self, samples: np.ndarray) -> EmotionWindowResult:
        n = len(EMOTION_COLUMNS)
        prob = {c: 1.0 / n for c in EMOTION_COLUMNS}
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2, axis=1))
        return EmotionWindowResult(
            probabilities=prob,
            label=EMOTION_COLUMNS[0],
            confidence=1.0 / n,
            channel_rms=rms,
            model_features=np.zeros(25),
        )

    @property
    def feature_columns(self) -> list[str]:
        return [f"f{i}" for i in range(25)]

    @property
    def emotion_labels(self) -> list[str]:
        return list(EMOTION_COLUMNS)


# ---------------------------------------------------------------------------
# StreamingWindowSource tests
# ---------------------------------------------------------------------------

class TestStreamingWindowSource:
    """Tests for the batch-filtfilt windowing source."""

    def test_window_shape(self):
        """Each valid window must be (n_channels, 500)."""
        _, eeg = _make_eeg(6.0)
        source = StreamingWindowSource(eeg)
        valid = [w for w in source if not w.is_invalid]
        assert valid, "Expected at least one valid window"
        for w in valid:
            assert w.samples is not None
            assert w.samples.shape == (_N_CHANNELS, _WINDOW_SAMPLES), (
                f"Expected ({_N_CHANNELS}, {_WINDOW_SAMPLES}), got {w.samples.shape}"
            )

    def test_hop_produces_correct_window_count(self):
        """A 6 s recording at 250 Hz should yield exactly the right number of windows."""
        n_seconds = 6.0
        n_samples = int(n_seconds * _SAMPLING_RATE)
        _, eeg = _make_eeg(n_seconds)
        source = StreamingWindowSource(eeg)
        windows = list(source)
        # start positions: 0, 125, 250, …, while start+500 <= 1500
        expected = max(0, (n_samples - _WINDOW_SAMPLES) // _STEP_SAMPLES + 1)
        assert len(windows) == expected, (
            f"Expected {expected} windows, got {len(windows)}"
        )

    def test_null_window_flagged(self):
        """A window where >10 % samples were NaN must be flagged is_invalid."""
        n_seconds = 6.0
        n_samples = int(n_seconds * _SAMPLING_RATE)
        # NaN the first 100 samples of the first window (100/500 = 20 % > 10 %).
        _, eeg = _make_eeg(n_seconds, nan_rows=slice(0, 100))
        source = StreamingWindowSource(eeg)
        windows = list(source)
        assert windows[0].is_invalid, "First window should be invalid (>10 % NaN)"
        assert windows[0].samples is None

    def test_valid_window_has_no_nan(self):
        """Valid window samples must be finite (NaN-fill before filter)."""
        _, eeg = _make_eeg(6.0)
        source = StreamingWindowSource(eeg)
        valid = [w for w in source if not w.is_invalid]
        assert valid
        for w in valid:
            assert np.isfinite(w.samples).all(), "Valid window samples must be finite"

    def test_timestamps_with_hardware_clock(self):
        """Window timestamps should be median of the hardware timestamps in the window."""
        ts, eeg = _make_eeg(6.0)
        source = StreamingWindowSource(eeg, timestamps=ts)
        windows = list(source)
        assert windows, "Expected windows"
        w0 = windows[0]
        expected_ts = float(np.median(ts[:_WINDOW_SAMPLES]))
        assert abs(w0.timestamp - expected_ts) < 1e-6

    def test_timestamps_without_hardware_clock(self):
        """Without hardware timestamps, window time is the sample-count midpoint."""
        _, eeg = _make_eeg(6.0)
        source = StreamingWindowSource(eeg, timestamps=None)
        windows = list(source)
        assert windows
        # mid of the first window: (0 + 500/2) / 250 = 1.0 s
        expected_ts = (_WINDOW_SAMPLES / 2) / _SAMPLING_RATE
        assert abs(windows[0].timestamp - expected_ts) < 1e-6


# ---------------------------------------------------------------------------
# Pipeline tests (using mock model)
# ---------------------------------------------------------------------------

class TestRunWindowPipeline:
    """Tests for the two-pass pipeline using a lightweight mock model."""

    def _run(self, n_seconds: float = 6.0, **source_kwargs):
        ts, eeg = _make_eeg(n_seconds)
        source = StreamingWindowSource(eeg, timestamps=ts, **source_kwargs)
        model = _FixedModel()
        result = run_window_pipeline(source, model)
        return result, model

    def test_produces_emotion_rows(self):
        result, _ = self._run()
        assert len(result.emotion_rows) > 0, "Pipeline must produce emotion rows"

    def test_emotion_probabilities_sum_to_one(self):
        """For every valid window, emotion probabilities must sum to 1."""
        result, _ = self._run()
        for row in result.emotion_rows:
            if not np.isnan(row[1]):
                s = sum(row[1:])
                assert abs(s - 1.0) < 1e-9, f"Probabilities sum to {s}, expected 1.0"

    def test_emotion_columns_all_present(self):
        """Every valid emotion row must include all EMOTION_COLUMNS."""
        result, _ = self._run()
        assert len(result.emotion_rows[0]) == 1 + len(EMOTION_COLUMNS)

    def test_rms_rows_shape(self):
        """rms_rows must have Timestamp + 8 channels per window."""
        result, _ = self._run()
        assert len(result.rms_rows) > 0
        assert result.rms_rows[0].shape == (1 + _N_CHANNELS,)

    def test_calibration_called_before_predict(self):
        """set_subject_context must be called with valid windows before predict."""
        ts, eeg = _make_eeg(6.0)
        source = StreamingWindowSource(eeg, timestamps=ts)
        model = _FixedModel()
        run_window_pipeline(source, model)
        assert model._calibrated, "set_subject_context was not called"
        assert model._context_shape is not None
        assert model._context_shape[1] == _N_CHANNELS
        assert model._context_shape[2] == _WINDOW_SAMPLES

    def test_null_windows_excluded_from_calibration(self):
        """Invalid windows must not appear in the set_subject_context call."""
        n_seconds = 6.0
        n_samples = int(n_seconds * _SAMPLING_RATE)
        # NaN the first window (first 100 samples → 20 % NaN in window 0).
        ts, eeg = _make_eeg(n_seconds, nan_rows=slice(0, 100))
        source = StreamingWindowSource(eeg, timestamps=ts)
        windows = list(source)
        n_valid = sum(1 for w in windows if not w.is_invalid)

        model = _FixedModel()
        source2 = StreamingWindowSource(eeg, timestamps=ts)
        run_window_pipeline(source2, model)

        assert model._context_shape is not None
        assert model._context_shape[0] == n_valid, (
            f"Calibration received {model._context_shape[0]} windows, "
            f"expected {n_valid} (valid only)"
        )

    def test_null_window_row_is_nan(self):
        """Invalid windows must produce NaN rows in emotion_rows."""
        n_seconds = 6.0
        _, eeg = _make_eeg(n_seconds, nan_rows=slice(0, 100))
        source = StreamingWindowSource(eeg)
        model = _FixedModel()
        result = run_window_pipeline(source, model)
        # First window is invalid.
        first = result.emotion_rows[0]
        assert np.isnan(first[1:]).all(), "Invalid window must yield NaN probabilities"

    def test_model_reset_called(self):
        """Pipeline must call model.reset() before collecting windows."""
        ts, eeg = _make_eeg(6.0)

        class TrackingModel(_FixedModel):
            def __init__(self):
                super().__init__()
                self.reset_count = 0

            def reset(self):
                super().reset()
                self.reset_count += 1

        model = TrackingModel()
        source = StreamingWindowSource(eeg, timestamps=ts)
        run_window_pipeline(source, model)
        assert model.reset_count == 1, "reset() must be called exactly once"

    def test_feature_columns_propagated(self):
        """PipelineResult.feature_columns must match model.feature_columns."""
        result, model = self._run()
        assert result.feature_columns == model.feature_columns


# ---------------------------------------------------------------------------
# ClassicalEmotionAdapter smoke tests (real model load)
# ---------------------------------------------------------------------------

RELEASE_DIR = (
    Path(__file__).parent.parent
    / "nucleuskit_toolkit"
    / "hermes"
    / "processor"
    / "emotions_processor"
    / "models"
    / "classical-emotion-2.5.0"
)


@pytest.mark.skipif(
    not (RELEASE_DIR / "pipeline.joblib").exists(),
    reason="classical-emotion-2.5.0 pipeline.joblib not found",
)
class TestClassicalEmotionAdapter:
    """Integration tests that load the real pipeline.joblib."""

    @pytest.fixture(scope="class")
    def adapter(self):
        return ClassicalEmotionAdapter.load()

    def test_load_succeeds(self, adapter):
        assert adapter is not None

    def test_emotion_labels_match_emotion_columns(self, adapter):
        assert set(adapter.emotion_labels) == set(EMOTION_COLUMNS)

    def test_feature_columns_length(self, adapter):
        assert len(adapter.feature_columns) == 25

    def test_predict_after_calibration(self, adapter):
        """predict must return a valid EmotionWindowResult after calibration."""
        rng = np.random.default_rng(7)
        n_windows = 10
        windows = rng.normal(scale=5.0, size=(n_windows, _N_CHANNELS, _WINDOW_SAMPLES)).astype(np.float64)

        adapter.reset()
        adapter.set_subject_context(windows)

        result = adapter.predict(windows[0])

        assert isinstance(result, EmotionWindowResult)
        assert result.label in EMOTION_COLUMNS
        assert result.channel_rms.shape == (_N_CHANNELS,)
        assert result.model_features.shape == (25,)

        prob_sum = sum(result.probabilities.values())
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities sum to {prob_sum}"
        assert set(result.probabilities.keys()) == set(EMOTION_COLUMNS)

    def test_full_pipeline_with_real_model(self):
        """End-to-end: StreamingWindowSource → pipeline → real model."""
        rng = np.random.default_rng(99)
        n_seconds = 6.0
        n_samples = int(n_seconds * _SAMPLING_RATE)
        eeg = rng.normal(scale=5.0, size=(n_samples, _N_CHANNELS)).astype(np.float64)
        ts = np.arange(n_samples, dtype=np.float64) / _SAMPLING_RATE

        model = ClassicalEmotionAdapter.load()
        source = StreamingWindowSource(eeg, timestamps=ts)
        result = run_window_pipeline(source, model)

        assert len(result.emotion_rows) > 0
        # At least some valid windows.
        valid = [r for r in result.emotion_rows if not np.isnan(r[1])]
        assert valid, "Expected at least one valid emotion window"
        for row in valid:
            s = sum(row[1:])
            assert abs(s - 1.0) < 1e-6, f"Probabilities sum to {s}"
