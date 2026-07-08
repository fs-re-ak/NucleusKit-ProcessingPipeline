"""
Unified window pipeline.

:func:`run_window_pipeline` is the single loop for all emotion classification
paths (raw EMG streaming and RMS-CSV replay).  It accepts any
:class:`~.interface.windows.WindowSource` and any
:class:`~.interface.model.EmotionModel`, so the caller (``session.py``) only
needs to pick the right source and model.
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
    """One row per valid window: ``[timestamp, *per_channel_rms]``."""

    model_input_rows: list = field(default_factory=list)
    """One row per valid window: ``[timestamp, *feature_vector, label, confidence]``."""

    feature_columns: list[str] = field(default_factory=list)
    """Feature column names matching ``model_input_rows`` columns (after Timestamp)."""


def run_window_pipeline(
    source: WindowSource,
    model: EmotionModel,
) -> PipelineResult:
    """
    Iterate *source*, classify each valid window with *model*, and return
    collected rows.

    Invalid windows (hardware disconnects) are emitted as NaN rows in
    ``emotion_rows`` and ``rms_rows`` but are excluded from
    ``model_input_rows`` (diagnostic file only).

    Parameters
    ----------
    source:
        Any :class:`~.interface.windows.WindowSource` (streaming or CSV).
    model:
        Any :class:`~.interface.model.EmotionModel` implementation.

    Returns
    -------
    PipelineResult
    """
    model.reset()
    result = PipelineResult(feature_columns=model.feature_columns)
    n_channels = len(result.feature_columns) - 1  # feature_columns = n_channels RMS + AVG_RMS

    for window in source:
        if window.is_invalid:
            result.emotion_rows.append([window.timestamp] + [np.nan] * len(EMOTION_COLUMNS))
            result.rms_rows.append(
                np.concatenate([[window.timestamp], np.full(n_channels, np.nan)])
            )
        else:
            win_result = model.infer_from_rms(window.channel_rms)
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
