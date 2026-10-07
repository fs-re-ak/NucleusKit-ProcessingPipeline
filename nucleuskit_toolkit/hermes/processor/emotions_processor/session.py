"""
computeEmotions — session-level orchestration.

Loads raw EXG, applies hardware-disconnect invalidation, runs the unified
window pipeline with the classical-emotion 2.5.0 model, and writes the
output CSVs.  Incremental semantics are preserved: existing output files are
never overwritten unless they are missing.

NLMS decorrelation is **not** applied to the EXG before emotion inference.
The classical-emotion 2.5.0 model was trained on bandpass-only windows;
NLMS costs ~14 percentage-points of LOSO accuracy.  The ``apply_nlms``
parameter is accepted for backward compatibility with existing callers but
is silently ignored by this step (cognition processing is unaffected).
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from os import path

import numpy as np
import pandas as pd

from nucleuskit_toolkit.hermes.processor.data_interface import loadEXG
from nucleuskit_toolkit.hermes.processor.cognition_processor import _simple_resample
from nucleuskit_toolkit.logging_utils import printInfo, printError

from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import (
    DISCONNECT_VALUE,
    EMOTION_COLUMNS,
    RMS_COLUMNS,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.streaming import (
    StreamingWindowSource,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.models import get_model
from nucleuskit_toolkit.hermes.processor.emotions_processor.pipeline import run_window_pipeline
from nucleuskit_toolkit.hermes.processor.emotions_processor.report import generate_report


def _invalidate_disconnected_samples(eeg_data: np.ndarray) -> None:
    """Replace hardware-saturated (disconnected) samples with NaN in-place."""
    eeg_data[np.isclose(abs(eeg_data), DISCONNECT_VALUE, atol=1)] = np.nan


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
    from nucleuskit_toolkit.hermes.processor.emotions_processor.models import DEFAULT_MODEL

    # Try to read training provenance from the manifest bundled with the release.
    experiment_id: str | None = None
    try:
        from nucleuskit_toolkit.hermes.processor.emotions_processor.models.classical_emotion.model import (
            _RELEASE_DIR,
        )
        manifest_path = _RELEASE_DIR / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            experiment_id = manifest.get("training", {}).get("experiment_id")
    except Exception:
        pass

    info: dict = {
        "model_name": model.model_name or (registry_key or DEFAULT_MODEL),
        "model_version": model.model_version or "",
        "registry_key": registry_key or DEFAULT_MODEL,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    if experiment_id:
        info["experiment_id"] = experiment_id

    out_path = path.join(emotions_dir, "model_info.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=4)
    printInfo(f"[emotionsProcessor] Writing {out_path}")


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

    Parameters
    ----------
    recpath:
        Path to the recording directory.
    apply_nlms:
        Accepted for backward compatibility; **ignored** by the emotion step.
        The classical-emotion 2.5.0 model must receive bandpass-only EXG.
        NLMS decorrelation is still applied to cognition processing.
    model_name:
        Registry key selecting the emotion model (default: ``"classical-emotion"``).
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

    model = get_model(model_name)
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

        printInfo(f"[emotionsProcessor] EMG loaded: shape={eeg_data.shape}")

        # Invalidate hardware-disconnect samples before filtering.
        _invalidate_disconnected_samples(eeg_data)

        use_hw_timestamps = timestamps is not None
        source = StreamingWindowSource(eeg_data, timestamps=timestamps)

        pipeline_result = run_window_pipeline(source, model)

        if not pipeline_result.emotion_rows:
            printError(
                "[emotionsProcessor] No emotion windows produced — recording may be too short "
                "for a full 2.0 s window at 250 Hz."
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

        _write_model_info(
            path.join(recpath, "features", "emotions"),
            model,
            model_name,
        )

        printInfo("[emotionsProcessor] Emotion computation completed")
        generate_report(recpath)

    except Exception as e:
        printError(f"[emotionsProcessor] Unhandled error: {e}")
        printError(f"[emotionsProcessor] Traceback:\n{traceback.format_exc()}")
