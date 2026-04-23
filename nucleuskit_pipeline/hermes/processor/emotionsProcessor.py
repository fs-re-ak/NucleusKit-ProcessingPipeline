"""
Emotions Processor

Computes emotional states from EMG signals using the bundled streaming
two-stage classifier (RMS features + artefact gate + LDA/KNN). When
``rmsSignals.csv`` already exists and emotion outputs must be (re)computed,
inference uses that file instead of raw EXG.

Feature traceability (under ``features/emotions/``, one row per window, timestamps
aligned with ``results/Emotions.csv``):

- ``rmsSignals.csv`` — raw per-channel window RMS (same basis as the L2 step).
- ``emotionClassifierInputs.csv`` — exact 1-D vector passed to the two-stage model
  (L2-normalised channel RMS + ``AVG_RMS``), with ``PredictedLabel`` and
  ``PredictedConfidence`` matching :meth:`TwoStageClassifier.predict` for that window.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

from collections import deque
from os import path
import os
import traceback

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from nucleuskit_pipeline.hermes.devices.HermesDataInterface import loadEXG
from nucleuskit_pipeline.hermes.emotion.inference import TwoStageClassifier
from nucleuskit_pipeline.hermes.emotion.realtime_classifier import StreamingEMGClassifier
from nucleuskit_pipeline.hermes.processorDevelopment.channel_fixer_release.channel_fixer.rms_columns import (
    normalize_rms_dataframe,
)
from nucleuskit_pipeline.hermes.processor.emotionsReportGenerator import generate_report
from nucleuskit_pipeline.logging_utils import printInfo, printError

DISCONNECT_VALUE = 187500.0

# Fraction of samples in a window that must be hardware-invalid (NaN) before
# the whole window is marked null.  Keeps brief jitter from producing spurious
# nulls while still catching true electrode disconnects (~100% NaN).
NULL_WINDOW_NAN_THRESHOLD = 0.10

# HermesConstants storage order -> classifier feature_columns.json order
# (AF8, AF7, CHEEK_R, CHEEK_L, EAR_R, AFz, BROW_L, NOSE)

# Downstream consumers expect this column order (see webcamPipeline, session review)
EMOTION_COLUMNS = [
    "Neutral",
    "Happiness",
    "Anger",
    "Surprise",
    "Contempt",
    "Disgust",
    "Fear",
    "Sadness",
]

RMS_COLUMNS = [
    "Timestamp",
    "AF8",
    "AF7",
    "CHEEK_R",
    "CHEEK_L",
    "EAR_R",
    "AFz",
    "BROW_L",
    "NOSE",
]


def _classifier_dir():
    return path.normpath(
        path.join(path.dirname(path.abspath(__file__)), "..", "emotion", "classifier")
    )


def _model_features_from_channel_rms(rms: np.ndarray) -> np.ndarray:
    """Match :meth:`StreamingEMGClassifier._classify`: L2 RMS + AVG_RMS."""
    rms = np.asarray(rms, dtype=float).reshape(8,)
    avg_rms = float(np.mean(rms))
    norm_rms = normalize(rms.reshape(1, -1)).squeeze()
    return np.append(norm_rms, avg_rms)


def _compute_emotions_from_rms_csv(
    recpath: str,
    emotions_path: str,
    model_out: str,
    features_out: str,
    clf_dir: str,
) -> None:
    """
    Recompute ``Emotions.csv`` and ``emotionClassifierInputs.csv`` from existing
    ``rmsSignals.csv`` (per-window channel RMS), without loading raw EXG.
    """
    printInfo(f"[emotionsProcessor] Loading RMS windows from {features_out}")
    df = normalize_rms_dataframe(pd.read_csv(features_out))
    channel_order = RMS_COLUMNS[1:]

    clf = TwoStageClassifier.load(clf_dir, cooldown_windows=0)
    clf.reset_artefact_state()

    emotion_rows: list[list] = []
    model_input_rows: list[list] = []

    for _, row in df.iterrows():
        ts = float(row["Timestamp"])
        rms = row[list(channel_order)].to_numpy(dtype=float)

        # Null RMS rows represent windows with lost samples — propagate as null.
        if np.isnan(rms).any():
            emotion_rows.append([ts] + [np.nan] * len(EMOTION_COLUMNS))
            model_input_rows.append([ts] + [np.nan] * len(clf.feature_columns) + [None, np.nan])
            continue

        x = _model_features_from_channel_rms(rms)
        proba, label, confidence = clf.infer(x)
        emotion_rows.append([ts] + [float(proba.get(c, 0.0)) for c in EMOTION_COLUMNS])
        model_input_rows.append(
            [ts]
            + x.tolist()
            + [label, float(confidence or 0.0)]
        )

    if not emotion_rows:
        printError("[emotionsProcessor] rmsSignals.csv produced no rows — cannot write emotions")
        return

    printInfo(f"[emotionsProcessor] Inferred {len(emotion_rows)} emotion windows from RMS file")

    os.makedirs(path.join(recpath, "results"), exist_ok=True)
    os.makedirs(path.join(recpath, "features", "emotions"), exist_ok=True)

    emo_df = pd.DataFrame(
        emotion_rows,
        columns=["Timestamp"] + EMOTION_COLUMNS,
    )
    printInfo(f"[emotionsProcessor] Writing {emotions_path}")
    emo_df.to_csv(emotions_path, na_rep="NULL", index=False)

    mcols = (
        ["Timestamp"]
        + clf.feature_columns
        + ["PredictedLabel", "PredictedConfidence"]
    )
    model_df = pd.DataFrame(model_input_rows, columns=mcols)
    printInfo(f"[emotionsProcessor] Writing {model_out}")
    model_df.to_csv(model_out, index=False)

    printInfo("[emotionsProcessor] Emotion computation completed (RMS-driven path)")
    generate_report(recpath)


def computeEmotions(recpath):
    """
    Compute emotions from EMG data using the bundled classifier.

    Incremental outputs under ``results/`` and ``features/emotions/``:
    - ``Emotions.csv`` and ``emotionClassifierInputs.csv`` are treated as a pair:
      if either is missing, both are recomputed.
    - If ``rmsSignals.csv`` already exists when that pair must be (re)computed,
      inference uses the RMS file only (no raw EXG load). Timestamps and
      per-window channel RMS match the file.
    - ``rmsSignals.csv`` is not overwritten when it already exists; if it is
      missing, it is filled from the streaming classifier over raw EXG as before.

    Args:
        recpath: Path to the recording directory
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
        printInfo(
            "[emotionsProcessor] Will (re)compute Emotions.csv and emotionClassifierInputs.csv"
        )
    if need_rms:
        printInfo("[emotionsProcessor] Will compute rmsSignals.csv")
    elif have_rms:
        printInfo("[emotionsProcessor] rmsSignals.csv exists — will not overwrite")

    clf_dir = _classifier_dir()
    if not path.isdir(clf_dir):
        printError(f"[emotionsProcessor] Classifier directory not found: {clf_dir}")
        return

    if need_pair and have_rms:
        printInfo(
            "[emotionsProcessor] rmsSignals.csv present — computing emotions from RMS file "
            "(skipping raw EXG)"
        )
        try:
            printInfo(f"[emotionsProcessor] Session: {recpath.split(os.path.sep)[-1]}")
            _compute_emotions_from_rms_csv(
                recpath, emotions_path, model_out, features_out, clf_dir
            )
        except Exception as e:
            printError(f"[emotionsProcessor] Unhandled error: {e}")
            printError(f"[emotionsProcessor] Traceback:\n{traceback.format_exc()}")
        return

    try:
        printInfo(f"[emotionsProcessor] Session: {recpath.split(os.path.sep)[-1]}")

        result = loadEXG(recpath, re_reference=False)
        if result is None:
            printError("[emotionsProcessor] loadEXG returned None — cannot proceed")
            return
        _timestamps, eeg_data = result
        if eeg_data is None:
            printError("[emotionsProcessor] eeg_data is None — cannot proceed")
            return

        if eeg_data.ndim != 2 or eeg_data.shape[1] != 8:
            printError(
                f"[emotionsProcessor] Expected EMG with 8 channels, got shape {eeg_data.shape}"
            )
            return

        printInfo(f"[emotionsProcessor] EMG loaded: shape={eeg_data.shape}")

        _invalidate_invalid_eeg_samples(eeg_data)

        clf = StreamingEMGClassifier(
            clf_dir=clf_dir,
            window_sec=1.0,
            step_sec=0.5,
            sampling_rate=250,
            cooldown_windows=0,
        )
        clf.reset()

        emotion_rows = []
        rms_rows = []
        model_input_rows = []

        # Track whether each sample in the current window buffer had any NaN
        # channel (invalid/disconnected hardware sample) before nan_to_num.
        nan_flags: deque = deque(maxlen=clf.window_samples)

        for sample_idx in range(eeg_data.shape[0]):
            row = eeg_data[sample_idx, :]
            nan_flags.append(bool(np.isnan(row).any()))
            row = np.nan_to_num(row, nan=0.0)
            proba = clf.push_sample(row)
            if proba is None:
                continue

            # Center of the analysis window.  clf.time_sec is the end of the
            # window (total_samples / sampling_rate); subtracting half the
            # window duration gives the exact center on a clean 0.5 s grid.
            ts = clf.time_sec - clf.window_sec / 2

            window_has_lost_samples = (sum(nan_flags) / len(nan_flags)) > NULL_WINDOW_NAN_THRESHOLD

            if window_has_lost_samples:
                # Window contained hardware-invalid (disconnected) samples —
                # emit a null row preserving the original timestamp.
                emotion_rows.append([ts] + [np.nan] * len(EMOTION_COLUMNS))
                rms_rows.append(np.concatenate([[ts], np.full(len(RMS_COLUMNS) - 1, np.nan)]))
                # model_input_rows: skip null windows (diagnostic file only)
            else:
                emotion_rows.append([ts] + [float(proba.get(c, 0.0)) for c in EMOTION_COLUMNS])

                rms = clf.last_channel_rms
                if rms is not None:
                    rms_rows.append(np.concatenate([[ts], rms]))

                xf = clf.last_model_features
                if xf is not None and clf.last_predicted_label is not None:
                    model_input_rows.append(
                        [ts]
                        + xf.tolist()
                        + [clf.last_predicted_label, float(clf.last_predicted_confidence or 0.0)]
                    )

        if not emotion_rows:
            printError(
                "[emotionsProcessor] No emotion windows produced — recording may be too short "
                "for a full 1.0 s window at 250 Hz."
            )
            return

        printInfo(f"[emotionsProcessor] Collected {len(emotion_rows)} emotion windows")

        os.makedirs(path.join(recpath, "results"), exist_ok=True)
        os.makedirs(path.join(recpath, "features", "emotions"), exist_ok=True)

        if need_pair:
            emo_df = pd.DataFrame(
                emotion_rows,
                columns=["Timestamp"] + EMOTION_COLUMNS,
            )
            printInfo(f"[emotionsProcessor] Writing {emotions_path}")
            emo_df.to_csv(emotions_path, na_rep="NULL", index=False)

            if model_input_rows:
                mcols = (
                    ["Timestamp"]
                    + clf.model_feature_columns
                    + ["PredictedLabel", "PredictedConfidence"]
                )
                model_df = pd.DataFrame(model_input_rows, columns=mcols)
                printInfo(f"[emotionsProcessor] Writing {model_out}")
                model_df.to_csv(model_out, index=False)
            else:
                printError(
                    "[emotionsProcessor] No model input rows for emotionClassifierInputs.csv — "
                    "pair output may be incomplete"
                )

        if need_rms and rms_rows:
            rms_df = pd.DataFrame(np.asarray(rms_rows), columns=RMS_COLUMNS)
            printInfo(f"[emotionsProcessor] Writing {features_out}")
            rms_df.to_csv(features_out, index=False)
        elif need_rms and not rms_rows:
            printError("[emotionsProcessor] rmsSignals.csv was needed but no RMS rows were collected")

        printInfo("[emotionsProcessor] Emotion computation completed")
        generate_report(recpath)

    except Exception as e:
        printError(f"[emotionsProcessor] Unhandled error: {e}")
        printError(f"[emotionsProcessor] Traceback:\n{traceback.format_exc()}")


def _invalidate_invalid_eeg_samples(eeg_data):
    """Invalidate disconnected or invalid EEG samples by setting them to NaN."""
    eeg_data[np.isclose(abs(eeg_data), DISCONNECT_VALUE, atol=1)] = np.nan
    eeg_data[np.isclose(abs(eeg_data), 0, atol=1)] = np.nan
