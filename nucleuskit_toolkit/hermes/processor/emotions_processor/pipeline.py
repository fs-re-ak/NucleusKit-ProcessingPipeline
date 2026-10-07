"""
Unified window pipeline.

:func:`run_window_pipeline` is the single loop for all emotion classification.
It accepts any :class:`~.interface.windows.WindowSource` and any
:class:`~.interface.model.EmotionModel`, so the caller (``session.py``) only
needs to build the right source and model.

Two-pass design
---------------
1. Collect all windows from *source* into memory.
2. Stack the valid windows and call :meth:`~.interface.model.EmotionModel.set_subject_context`
   once so the model can compute per-subject normalisation stats.
3. Classify every window and accumulate CSV rows.

This approach is required by the ``classical-emotion`` feature set
(``extended_nm_a``), which z-scores RMS and waveform-length features against
the full-session distribution before classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import EMOTION_COLUMNS
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.model import EmotionModel
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.windows import WindowSource


@dataclass
class PipelineResult:
    """Collected rows ready for CSV serialisation."""

    emotion_rows: list = field(default_factory=list)
    """One row per window: ``[timestamp, *per_emotion_probabilities]``."""

    rms_rows: list = field(default_factory=list)
    """One row per window: ``[timestamp, *per_channel_rms]``.
    Valid windows carry the raw RMS computed from the filtered samples;
    invalid windows carry NaN."""

    model_input_rows: list = field(default_factory=list)
    """One row per **valid** window: ``[timestamp, *feature_vector, label, confidence]``."""

    feature_columns: list[str] = field(default_factory=list)
    """Feature column names matching ``model_input_rows`` columns (after Timestamp)."""


def run_window_pipeline(
    source: WindowSource,
    model: EmotionModel,
) -> PipelineResult:
    """
    Collect all windows from *source*, calibrate *model*, then classify.

    Parameters
    ----------
    source:
        Any :class:`~.interface.windows.WindowSource`.
    model:
        Any :class:`~.interface.model.EmotionModel` implementation.

    Returns
    -------
    PipelineResult
    """
    model.reset()

    # Pass 1 — materialise the source so we can calibrate before classifying.
    all_windows = list(source)

    # Calibrate on the valid subset (required for per-subject z-score models).
    valid_samples = [w.samples for w in all_windows if not w.is_invalid]
    if valid_samples:
        context = np.stack(valid_samples)  # (n_valid, n_channels, n_samples)
        model.set_subject_context(context)

    # Pass 2 — classify and accumulate rows.
    result = PipelineResult(feature_columns=model.feature_columns)
    n_channels = 8  # rms_rows always has one column per Hermes channel

    for window in all_windows:
        if window.is_invalid:
            result.emotion_rows.append(
                [window.timestamp] + [np.nan] * len(EMOTION_COLUMNS)
            )
            result.rms_rows.append(
                np.concatenate([[window.timestamp], np.full(n_channels, np.nan)])
            )
        else:
            win_result = model.predict(window.samples)
            result.emotion_rows.append(
                [window.timestamp]
                + [float(win_result.probabilities.get(c, 0.0)) for c in EMOTION_COLUMNS]
            )
            result.rms_rows.append(
                np.concatenate([[window.timestamp], win_result.channel_rms])
            )
            result.model_input_rows.append(
                [window.timestamp]
                + win_result.model_features.tolist()
                + [win_result.label, float(win_result.confidence)]
            )

    return result
