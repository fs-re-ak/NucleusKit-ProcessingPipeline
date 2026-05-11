"""
PPG Fixer: standalone post-hoc artifact rejection for an already-processed session.

This module is kept for backward-compatibility and developer use.  The artifact
rejection logic (squared z-score outlier detection, margin expansion, figure
generation) now lives in ``heart.py`` and is executed automatically as part of
the standard ``computeHeartDynamics`` pipeline.  Use ``fix_ppg_session`` only
when you need to re-apply the rejection to a session that was processed by an
older pipeline version.

Steps performed by ``fix_ppg_session``
---------------------------------------
1. Load the already-resampled PPG from ``features/ppg/ppg_resampled.csv``.
2. Apply squared z-score artifact rejection (via ``apply_ppg_artifact_rejection``).
3. Overwrite ``features/ppg/ppg_resampled.csv`` with the cleaned signal.
4. Delete the cached ``results/HeartDynamics.csv`` so it is recomputed.
5. Re-run peak detection, diagnostic figure, sliding HRV and save a fresh
   ``results/HeartDynamics.csv``.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Spring 2026
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from nucleuskit_pipeline.logging_utils import printError, printInfo, printWarning

from nucleuskit_pipeline.shimmer.processor.heart import (
    SAMPLING_RATE,
    _UPSAMPLE_RATE,
    MISSING_BEAT_MAX_FILL_S,
    MISSING_BEAT_RATIO,
    apply_ppg_artifact_rejection,
    _detect_ppg_peaks,
    _fill_missing_beats,
    _save_ppg_figure,
    _sliding_hrv_dataframe,
)


def fix_ppg_session(rec_path: str) -> None:
    """Re-run PPG artifact rejection on an already-processed session.

    Parameters
    ----------
    rec_path:
        Path to a session directory that already has
        ``features/ppg/ppg_resampled.csv`` (produced by the normal offline
        pipeline run).

    Raises
    ------
    FileNotFoundError
        If ``features/ppg/ppg_resampled.csv`` is not found.
    """
    ppg_features_dir = os.path.join(rec_path, "features", "ppg")
    resampled_csv    = os.path.join(ppg_features_dir, "ppg_resampled.csv")

    if not os.path.isfile(resampled_csv):
        raise FileNotFoundError(
            f"[ppgFixer] Resampled PPG not found: {resampled_csv}\n"
            "Run offline processing for this session first."
        )

    printInfo(f"[ppgFixer] Loading resampled PPG from {resampled_csv}")
    df         = pd.read_csv(resampled_csv)
    timestamps = df["Timestamp"].values.astype(float)
    ppg_raw    = df["Value"].values.astype(float)

    # Artifact rejection
    ppg_fixed, _ = apply_ppg_artifact_rejection(ppg_raw, timestamps, ppg_features_dir)

    # Overwrite resampled CSV with the cleaned signal
    df["Value"] = ppg_fixed
    df.to_csv(resampled_csv, index=False)
    printInfo(f"[ppgFixer] Saved fixed PPG → {resampled_csv}")

    # Invalidate cached HeartDynamics.csv
    heart_dynamics_path = os.path.join(rec_path, "results", "HeartDynamics.csv")
    if os.path.isfile(heart_dynamics_path):
        os.remove(heart_dynamics_path)
        printInfo("[ppgFixer] Removed cached HeartDynamics.csv")

    if len(ppg_fixed) < 250:
        printWarning("[ppgFixer] Remaining signal too short — skipping peak detection.")
        return

    gap_mask = np.isnan(ppg_fixed)

    try:
        ppg_clean_256, peak_idx_256, peak_idx_orig, t_upsampled, gap_mask_256 = _detect_ppg_peaks(
            ppg_fixed, gap_mask=gap_mask
        )
    except Exception as exc:
        printError(f"[ppgFixer] Peak detection failed: {exc}")
        return

    peak_idx_orig, _, n_inserted_orig = _fill_missing_beats(
        peak_idx_orig, fs=SAMPLING_RATE, gap_mask=gap_mask,
    )
    peak_idx_256, synth_mask_256, _ = _fill_missing_beats(
        peak_idx_256, fs=_UPSAMPLE_RATE, gap_mask=gap_mask_256,
    )
    printInfo(
        f"[ppgFixer] Inserted {n_inserted_orig} synthetic peaks to fill "
        f"missed beats (adaptive threshold = {MISSING_BEAT_RATIO}x recent "
        f"median IBI, max gap = {MISSING_BEAT_MAX_FILL_S:.0f} s)"
    )

    _save_ppg_figure(ppg_clean_256, peak_idx_256, t_upsampled, ppg_features_dir,
                     synth_mask=synth_mask_256)

    try:
        hrv_df = _sliding_hrv_dataframe(peak_idx_orig)
    except Exception as exc:
        printError(f"[ppgFixer] HRV computation failed: {exc}")
        return

    if hrv_df.empty:
        printWarning("[ppgFixer] No valid HRV windows after artifact rejection.")
        return

    os.makedirs(os.path.join(rec_path, "results"), exist_ok=True)
    hrv_df.to_csv(heart_dynamics_path, index=False)
    printInfo(f"[ppgFixer] Heart dynamics recomputed → {heart_dynamics_path}")
