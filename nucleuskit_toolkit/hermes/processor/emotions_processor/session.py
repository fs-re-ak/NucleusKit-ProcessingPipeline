"""
computeEmotions — session-level orchestration.

Loads raw EXG, applies hardware-disconnect invalidation, runs the unified
window pipeline with the selected emotion model, and writes the output CSVs.

Incremental semantics: existing output files are reused only when
``features/emotions/model_info.json`` names the same release as the current
request.  Legacy files whose ``registry_key`` is ``classical-emotion`` (or
that have no ``model_info.json``) are treated as 2.5.0.

NLMS decorrelation is **not** applied to the EXG before emotion inference.
The ``apply_nlms`` parameter is accepted for backward compatibility with
existing callers but is silently ignored by this step (cognition processing
is unaffected).
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from os import path
from pathlib import Path

import numpy as np
import pandas as pd

from nucleuskit_toolkit.hermes.constants import OUTPUT_RMS_COLUMNS
from nucleuskit_toolkit.hermes.processor.cognition_processor import _simple_resample
from nucleuskit_toolkit.hermes.processor.data_interface import loadEXG
from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import (
    DISCONNECT_VALUE,
    EMOTION_COLUMNS,
    RMS_COLUMNS,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.streaming import (
    StreamingWindowSource,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.models import (
    DEFAULT_MODEL,
    canonicalize_model_key,
    get_model,
    load_experiment_id,
    resolve_model,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.pipeline import run_window_pipeline
from nucleuskit_toolkit.hermes.processor.emotions_processor.report import generate_report
from nucleuskit_toolkit.logging_utils import printError, printInfo


def _invalidate_disconnected_samples(eeg_data: np.ndarray) -> None:
    """Replace hardware-saturated (disconnected) samples with NaN in-place."""
    eeg_data[np.isclose(abs(eeg_data), DISCONNECT_VALUE, atol=1)] = np.nan


def outputs_match_requested_model(emotions_dir: str, model_name: str | None) -> bool:
    """True when existing ``model_info.json`` describes the requested release.

    Missing ``model_info.json`` is treated as classical-emotion 2.5.0 so
    re-running the default model does not redo work.  Unreadable JSON is a
    mismatch (recompute).
    """
    try:
        requested_info = resolve_model(model_name)
        requested_key = requested_info.key
        requested_version = requested_info.version
    except ValueError:
        requested_key = canonicalize_model_key(model_name)
        requested_version = ""

    info_path = path.join(emotions_dir, "model_info.json")
    if not path.isfile(info_path):
        stored_key = canonicalize_model_key(DEFAULT_MODEL, "2.5.0")
        stored_version = "2.5.0"
    else:
        try:
            data = json.loads(Path(info_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        stored_key = canonicalize_model_key(
            data.get("registry_key") or DEFAULT_MODEL,
            str(data.get("model_version") or ""),
        )
        stored_version = str(data.get("model_version") or "")

    if stored_key != requested_key:
        return False
    if stored_version and requested_version and stored_version != requested_version:
        return False
    return True


def _write_model_info(emotions_dir: str, model, registry_key: str | None) -> None:
    """Write ``model_info.json`` to *emotions_dir* recording which model produced the outputs.

    Parameters
    ----------
    emotions_dir:
        Path to ``features/emotions/`` for the current recording.
    model:
        The :class:`~.interface.model.EmotionModel` instance that was used.
    registry_key:
        The registry key that was passed to :func:`~.models.get_model`
        (``None`` means the default was used).
    """
    try:
        resolved = resolve_model(registry_key)
        canonical_key = resolved.key
    except ValueError:
        canonical_key = canonicalize_model_key(registry_key)

    experiment_id: str | None = None
    release_dir = getattr(model, "release_dir", None)
    if release_dir is not None:
        experiment_id = load_experiment_id(release_dir)

    info: dict = {
        "model_name": model.model_name or canonical_key,
        "model_version": model.model_version or "",
        "registry_key": canonical_key,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    if experiment_id:
        info["experiment_id"] = experiment_id

    out_path = path.join(emotions_dir, "model_info.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=4)
    printInfo(f"[emotionsProcessor] Writing {out_path}")


def _outputs_use_rms_names(model_out: str, features_out: str) -> bool:
    """True when existing emotion CSVs already use the HEAD_*_RMS column names."""
    required = set(OUTPUT_RMS_COLUMNS)
    for csv_path in (model_out, features_out):
        if not path.isfile(csv_path):
            continue
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except (OSError, pd.errors.EmptyDataError, ValueError):
            return False
        if not required.issubset(columns):
            return False
    return True


def _attach_l2_rms_columns(model_df: pd.DataFrame, channel_rms: list) -> None:
    """Add L2-normalised per-channel RMS columns expected by the emotions report."""
    if not channel_rms or len(channel_rms) != len(model_df):
        printError(
            "[emotionsProcessor] Could not attach "
            + ", ".join(OUTPUT_RMS_COLUMNS)
            + " — RMS rows do not match classifier windows"
        )
        return
    rms = np.vstack(channel_rms)
    norms = np.linalg.norm(rms, axis=1, keepdims=True)
    unit = np.divide(rms, np.maximum(norms, 1e-12))
    for i, col in enumerate(OUTPUT_RMS_COLUMNS):
        model_df[col] = unit[:, i]


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
    - ``rmsSignals.csv`` records the raw per-channel RMS of each 2.0 s window
      and is recomputed together with the emotion pair.
    - Existing files are reused only when they were produced by the same
      release as *model_name*.

    Parameters
    ----------
    recpath:
        Path to the recording directory.
    apply_nlms:
        Accepted for backward compatibility; **ignored** by the emotion step.
        Emotion models must receive bandpass-only EXG (plus the release's
        notch, if any).  NLMS decorrelation is still applied to cognition
        processing.
    model_name:
        Registry key selecting the emotion model (directory name, or the
        ``classical-emotion`` alias for 2.5.0).
    """
    printInfo("[emotionsProcessor] Computing Emotions")

    emotions_path = path.join(recpath, "results", "Emotions.csv")
    model_out = path.join(recpath, "features", "emotions", "emotionClassifierInputs.csv")
    features_out = path.join(recpath, "features", "emotions", "rmsSignals.csv")
    emotions_dir = path.join(recpath, "features", "emotions")

    have_pair = path.isfile(emotions_path) and path.isfile(model_out)
    have_rms = path.isfile(features_out)
    need_pair = not have_pair
    need_rms = not have_rms

    if have_pair or have_rms:
        if not outputs_match_requested_model(emotions_dir, model_name):
            need_pair = True
            need_rms = True
            printInfo(
                "[emotionsProcessor] Existing emotion outputs were produced by a "
                "different model — recomputing"
            )
        elif not _outputs_use_rms_names(model_out, features_out):
            need_pair = True
            need_rms = True
            printInfo(
                "[emotionsProcessor] Existing emotion features use older channel names — recomputing"
            )

    if not need_pair and not need_rms:
        printInfo(
            "[emotionsProcessor] Emotions, classifier inputs, and RMS features already present — skipping"
        )
        return

    if need_pair:
        printInfo("[emotionsProcessor] Will (re)compute Emotions.csv and emotionClassifierInputs.csv")
    if need_rms:
        printInfo("[emotionsProcessor] Will compute rmsSignals.csv")

    printInfo(
        "[emotionsProcessor] Channel map: ch1–ch8 → " + ", ".join(OUTPUT_RMS_COLUMNS)
    )

    model = get_model(model_name)
    spec = model.preprocess_spec
    printInfo(
        f"[emotionsProcessor] Using emotion model {model.model_name} {model.model_version}"
    )
    notch_msg = f", notch {spec.notch_hz} Hz" if spec.notch_hz else ", no notch"
    printInfo(
        f"[emotionsProcessor] Preprocess: bandpass {spec.bandpass_low}–{spec.bandpass_high} Hz{notch_msg}"
    )
    if spec.used_fallback:
        printInfo(
            "[emotionsProcessor] Release config.json does not declare bandpass_hz; "
            "using 15–40 Hz, no notch"
        )

    use_hw_timestamps = False

    try:
        printInfo(f"[emotionsProcessor] Session: {recpath.split(os.sep)[-1]}")

        # Always load raw EXG — RMS-replay is not supported with this model
        # because subject calibration requires the full filtered recording.
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

        # Invalidate hardware-disconnect samples before filtering.
        _invalidate_disconnected_samples(eeg_data)

        use_hw_timestamps = timestamps is not None
        source = StreamingWindowSource(
            eeg_data,
            timestamps=timestamps,
            bandpass_low=spec.bandpass_low,
            bandpass_high=spec.bandpass_high,
            notch_hz=spec.notch_hz,
        )

        pipeline_result = run_window_pipeline(source, model)

        if not pipeline_result.emotion_rows:
            printError(
                "[emotionsProcessor] No emotion windows produced — recording may be too short "
                "for a full 2.0 s window at 250 Hz."
            )
            return

        printInfo(f"[emotionsProcessor] Collected {len(pipeline_result.emotion_rows)} emotion windows")

        os.makedirs(path.join(recpath, "results"), exist_ok=True)
        os.makedirs(emotions_dir, exist_ok=True)

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
                _attach_l2_rms_columns(model_df, pipeline_result.valid_channel_rms)
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

        _write_model_info(emotions_dir, model, model_name)

        printInfo("[emotionsProcessor] Emotion computation completed")
        generate_report(recpath)

    except Exception as e:
        printError(f"[emotionsProcessor] Unhandled error: {e}")
        printError(f"[emotionsProcessor] Traceback:\n{traceback.format_exc()}")
