"""
computeEmotions — session-level orchestration.

Decides which window source to use (streaming from raw EMG or replay from an
existing ``rmsSignals.csv``), runs the unified pipeline, and writes the output
CSVs.  Incremental semantics are preserved: existing output files are never
overwritten unless they are missing.
"""

from __future__ import annotations

import os
import traceback
from os import path

import numpy as np
import pandas as pd

from nucleuskit_pipeline.hermes.processor.data_interface import loadEXG
from nucleuskit_pipeline.hermes.processor.cognition_processor import _simple_resample
from nucleuskit_pipeline.logging_utils import printInfo, printError

from nucleuskit_pipeline.hermes.processor.emotions_processor.constants import (
    DISCONNECT_VALUE,
    EMOTION_COLUMNS,
    RMS_COLUMNS,
)
from nucleuskit_pipeline.hermes.processor.emotions_processor.interface.streaming import (
    RmsCsvWindowSource,
    StreamingWindowSource,
)
from nucleuskit_pipeline.hermes.processor.emotions_processor.models import get_model
from nucleuskit_pipeline.hermes.processor.emotions_processor.pipeline import run_window_pipeline
from nucleuskit_pipeline.hermes.processor.emotions_processor.report import generate_report


def _default_weights_dir() -> str:
    return path.normpath(
        path.join(
            path.dirname(path.abspath(__file__)),
            "models", "v12", "weights",
        )
    )


def _invalidate_disconnected_samples(eeg_data: np.ndarray) -> None:
    """Replace hardware-saturated (disconnected) samples with NaN in-place."""
    eeg_data[np.isclose(abs(eeg_data), DISCONNECT_VALUE, atol=1)] = np.nan


def computeEmotions(
    recpath: str,
    apply_nlms: bool = True,
    model_name: str | None = None,
) -> None:
    """
    Compute emotions from EMG data and write results to *recpath*.

    Incremental outputs under ``results/`` and ``features/emotions/``:

    - ``Emotions.csv`` and ``emotionClassifierInputs.csv`` are treated as a
      pair: if either is missing both are recomputed.
    - ``rmsSignals.csv`` is computed from raw EXG the first time and then
      never overwritten.  If it already exists when the emotion pair needs
      recomputing, inference runs from the RMS file (no raw EXG load).

    Parameters
    ----------
    recpath:
        Path to the recording directory.
    apply_nlms:
        Apply causal NLMS adaptive decorrelation before windowing (streaming
        path only; ignored when replaying from ``rmsSignals.csv``).
    model_name:
        Registry key selecting the emotion model (default: ``"v12"``).
    """
    printInfo("[emotionsProcessor] Computing Emotions")

    emotions_path = path.join(recpath, "results", "Emotions.csv")
    model_out = path.join(recpath, "features", "emotions", "emotionClassifierInputs.csv")
    features_out = path.join(recpath, "features", "emotions", "rmsSignals.csv")

    have_pair = path.isfile(emotions_path) and path.isfile(model_out)
    have_rms = path.isfile(features_out)
    need_pair = not have_pair
    need_rms = not have_rms

    if not need_pair and not need_rms:
        printInfo(
            "[emotionsProcessor] Emotions, classifier inputs, and RMS features already present — skipping"
        )
        return

    if need_pair:
        printInfo("[emotionsProcessor] Will (re)compute Emotions.csv and emotionClassifierInputs.csv")
    if need_rms:
        printInfo("[emotionsProcessor] Will compute rmsSignals.csv")
    elif have_rms:
        printInfo("[emotionsProcessor] rmsSignals.csv exists — will not overwrite")

    clf_dir = _default_weights_dir()
    if not path.isdir(clf_dir):
        printError(f"[emotionsProcessor] Classifier weights directory not found: {clf_dir}")
        return

    model = get_model(model_name, weights_dir=clf_dir, cooldown_windows=0)
    use_hw_timestamps = False

    try:
        printInfo(f"[emotionsProcessor] Session: {recpath.split(os.sep)[-1]}")

        if need_pair and have_rms:
            # Fast path: re-derive emotion probabilities from existing RMS features.
            printInfo(
                "[emotionsProcessor] rmsSignals.csv present — computing emotions from RMS file "
                "(skipping raw EXG)"
            )
            source = RmsCsvWindowSource(features_out)
        else:
            # Full path: load raw EXG, filter, window, and compute RMS.
            result = loadEXG(recpath, re_reference=False)
            if result is None:
                printError("[emotionsProcessor] loadEXG returned None — cannot proceed")
                return
            timestamps, eeg_data = result
            if eeg_data is None:
                printError("[emotionsProcessor] eeg_data is None — cannot proceed")
                return

            if eeg_data.ndim != 2 or eeg_data.shape[1] != 8:
                printError(
                    f"[emotionsProcessor] Expected EMG with 8 channels, got shape {eeg_data.shape}"
                )
                return

            printInfo(f"[emotionsProcessor] EMG loaded: shape={eeg_data.shape}")

            if apply_nlms:
                from nucleuskit_pipeline.hermes.realtime.nlms_filter import CausalNLMSFilter
                printInfo("[emotionsProcessor] Applying NLMS adaptive decorrelation...")
                _nlms = CausalNLMSFilter(n_channels=8, fs=250.0)
                eeg_data = _nlms.push_batch(eeg_data)
                printInfo("[emotionsProcessor] NLMS decorrelation complete")

            _invalidate_disconnected_samples(eeg_data)
            use_hw_timestamps = timestamps is not None
            source = StreamingWindowSource(eeg_data, timestamps=timestamps)

        pipeline_result = run_window_pipeline(source, model)

        if not pipeline_result.emotion_rows:
            printError(
                "[emotionsProcessor] No emotion windows produced — recording may be too short "
                "for a full 1.0 s window at 250 Hz."
            )
            return

        printInfo(f"[emotionsProcessor] Collected {len(pipeline_result.emotion_rows)} emotion windows")

        os.makedirs(path.join(recpath, "results"), exist_ok=True)
        os.makedirs(path.join(recpath, "features", "emotions"), exist_ok=True)

        if need_pair:
            emo_df = pd.DataFrame(
                pipeline_result.emotion_rows,
                columns=["Timestamp"] + EMOTION_COLUMNS,
            )
            if use_hw_timestamps:
                emo_df = _simple_resample(emo_df, target_interval=0.5)

            printInfo(f"[emotionsProcessor] Writing {emotions_path}")
            emo_df.to_csv(emotions_path, na_rep="NULL", index=False)

            if pipeline_result.model_input_rows:
                mcols = (
                    ["Timestamp"]
                    + pipeline_result.feature_columns
                    + ["PredictedLabel", "PredictedConfidence"]
                )
                model_df = pd.DataFrame(pipeline_result.model_input_rows, columns=mcols)
                printInfo(f"[emotionsProcessor] Writing {model_out}")
                model_df.to_csv(model_out, index=False)
            else:
                printError(
                    "[emotionsProcessor] No model input rows for emotionClassifierInputs.csv — "
                    "pair output may be incomplete"
                )

        if need_rms and pipeline_result.rms_rows:
            rms_df = pd.DataFrame(
                np.asarray(pipeline_result.rms_rows),
                columns=RMS_COLUMNS,
            )
            printInfo(f"[emotionsProcessor] Writing {features_out}")
            rms_df.to_csv(features_out, index=False)
        elif need_rms and not pipeline_result.rms_rows:
            printError("[emotionsProcessor] rmsSignals.csv was needed but no RMS rows were collected")

        printInfo("[emotionsProcessor] Emotion computation completed")
        generate_report(recpath)

    except Exception as e:
        printError(f"[emotionsProcessor] Unhandled error: {e}")
        printError(f"[emotionsProcessor] Traceback:\n{traceback.format_exc()}")
