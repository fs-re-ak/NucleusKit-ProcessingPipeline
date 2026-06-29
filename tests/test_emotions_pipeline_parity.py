"""
Parity test: StreamingWindowSource and RmsCsvWindowSource must produce
identical RMS values when the streaming source first generates the RMS CSV
that the CSV source then replays.

This ensures the unified pipeline produces bit-for-bit consistent emotion
outputs regardless of which path is selected by session.py.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nucleuskit_pipeline.hermes.processor.emotions_processor.interface.streaming import (
    RmsCsvWindowSource,
    StreamingWindowSource,
)
from nucleuskit_pipeline.hermes.processor.emotions_processor.models import get_model
from nucleuskit_pipeline.hermes.processor.emotions_processor.pipeline import run_window_pipeline
from nucleuskit_pipeline.hermes.processor.emotions_processor.constants import RMS_COLUMNS


_RNG = np.random.default_rng(0)
_N_SAMPLES = 750       # 3 seconds at 250 Hz → 5 full windows (step=125)
_N_CHANNELS = 8
_SAMPLING_RATE = 250


def _make_synthetic_eeg(n_samples: int = _N_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, eeg_data) with no NaN samples."""
    eeg = _RNG.normal(loc=0.0, scale=5.0, size=(n_samples, _N_CHANNELS)).astype(float)
    timestamps = np.arange(n_samples, dtype=float) / _SAMPLING_RATE
    return timestamps, eeg


def _rms_rows_to_csv(rms_rows: list, tmp_dir: Path) -> str:
    """Serialise pipeline rms_rows to a temp CSV and return its path."""
    csv_path = str(tmp_dir / "rmsSignals.csv")
    df = pd.DataFrame(np.asarray(rms_rows), columns=RMS_COLUMNS)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def streaming_result():
    """Run the pipeline via StreamingWindowSource once, return result + model."""
    model = get_model("v12", cooldown_windows=0)
    timestamps, eeg = _make_synthetic_eeg()
    source = StreamingWindowSource(eeg, timestamps=timestamps)
    result = run_window_pipeline(source, model)
    return result, model


def test_streaming_produces_windows(streaming_result):
    result, _ = streaming_result
    assert len(result.emotion_rows) > 0, "StreamingWindowSource produced no windows"


def test_rms_csv_parity(streaming_result, tmp_path):
    """RmsCsvWindowSource replaying the streaming-produced RMS must yield the
    same emotion probabilities (within floating-point tolerance)."""
    streaming_res, model = streaming_result

    csv_path = _rms_rows_to_csv(streaming_res.rms_rows, tmp_path)

    csv_result = run_window_pipeline(RmsCsvWindowSource(csv_path), model)

    assert len(csv_result.emotion_rows) == len(streaming_res.emotion_rows), (
        f"Row count mismatch: streaming={len(streaming_res.emotion_rows)}, "
        f"csv={len(csv_result.emotion_rows)}"
    )

    streaming_arr = np.array(
        [[v if v is not None else np.nan for v in row[1:]] for row in streaming_res.emotion_rows],
        dtype=float,
    )
    csv_arr = np.array(
        [[v if v is not None else np.nan for v in row[1:]] for row in csv_result.emotion_rows],
        dtype=float,
    )

    np.testing.assert_allclose(
        streaming_arr, csv_arr, rtol=1e-6, atol=1e-9,
        err_msg="Emotion probabilities differ between streaming and RMS-CSV paths",
    )


def test_rms_csv_parity_model_features(streaming_result, tmp_path):
    """Model input feature vectors must also be identical across both paths."""
    streaming_res, model = streaming_result

    csv_path = _rms_rows_to_csv(streaming_res.rms_rows, tmp_path)
    csv_result = run_window_pipeline(RmsCsvWindowSource(csv_path), model)

    assert len(csv_result.model_input_rows) == len(streaming_res.model_input_rows), (
        "model_input_rows count mismatch"
    )

    for i, (s_row, c_row) in enumerate(
        zip(streaming_res.model_input_rows, csv_result.model_input_rows)
    ):
        s_feats = np.array(s_row[1:-2], dtype=float)
        c_feats = np.array(c_row[1:-2], dtype=float)
        np.testing.assert_allclose(
            s_feats, c_feats, rtol=1e-6, atol=1e-9,
            err_msg=f"Feature vector mismatch at window {i}",
        )
        assert s_row[-2] == c_row[-2], f"PredictedLabel mismatch at window {i}"
