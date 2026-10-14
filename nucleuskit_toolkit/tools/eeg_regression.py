"""
EMG-regression variant of the cognition pipeline (Tools > EEG regression page).

Exploratory Nucleus-Kit tool, not part of re-ak-core: it removes the EMG noise shared with the
reference channels (CHEEK_R, CHEEK_L, BROW_L, NOSE) from the EEG by ridge regression before
re-ak-core's cognition steps (filtering, artefact rejection, power bands, Engagement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from reak_core.io.exg import CHANNEL_NAMES
from sklearn.linear_model import Ridge

from nucleuskit_toolkit.logging_utils import printInfo, printWarning

REFERENCE_CHANNELS = ("CHEEK_R", "CHEEK_L", "BROW_L", "NOSE")
"""EMG reference channels. AFz is left out: it is an EEG channel and would remove brain signal."""


def reference_channels(exg: np.ndarray) -> pd.DataFrame | None:
    """The reference channels of a marked EXG matrix (hardware order, invalid samples already NaN)."""
    if exg is None or exg.ndim != 2 or exg.shape[1] < len(CHANNEL_NAMES):
        printWarning(f"[eegRegressionTool] Expected {len(CHANNEL_NAMES)} EXG channels to extract the references")
        return None
    return pd.DataFrame({name: exg[:, CHANNEL_NAMES.index(name)] for name in REFERENCE_CHANNELS})


def regress_noise_channels(eeg_df, ref_df):
    """
    Remove shared EMG noise from EEG channels via ridge regression.

    For each EEG channel, a Ridge regressor is fitted against all reference
    channels using only the samples where both the target and every reference
    are finite (non-NaN).  The fitted model's prediction is then subtracted
    from the full signal; samples that are NaN in the reference matrix are
    replaced with zeros before prediction (those samples are already flagged
    by ``hardware_invalid`` and will be excluded downstream).

    Reference channels are expected to be filtered with the same bandpass as
    ``eeg_df`` so that the regression operates on the same frequency content.

    Args:
        eeg_df: Bandpass-filtered EEG DataFrame (samples × EEG channels).
        ref_df: Bandpass-filtered reference DataFrame (samples × reference
            channels, e.g. CHEEK_R, CHEEK_L, BROW_L, NOSE).

    Returns:
        Cleaned EEG DataFrame of the same shape as ``eeg_df``.
    """
    printInfo("[cognitionProcessor] Applying regression-based noise removal")

    cleaned = eeg_df.copy()
    X_raw = ref_df.to_numpy(dtype=float)
    X_safe = np.nan_to_num(X_raw, nan=0.0)  # for prediction on gap samples

    for ch in eeg_df.columns:
        y = eeg_df[ch].to_numpy(dtype=float)

        # Use only rows where target AND all references are finite
        valid = np.isfinite(y) & np.all(np.isfinite(X_raw), axis=1)
        n_valid = int(valid.sum())

        if n_valid < X_raw.shape[1] + 10:
            printWarning(
                f"[cognitionProcessor] regress_noise_channels: channel '{ch}' "
                f"has only {n_valid} valid samples (need > {X_raw.shape[1] + 9}) "
                f"— skipping regression for this channel"
            )
            continue

        reg = Ridge(alpha=1.0, fit_intercept=True)
        reg.fit(X_raw[valid], y[valid])

        predicted = reg.predict(X_safe)
        residual = y - predicted

        # Variance reduction as a quality indicator
        var_orig = float(np.nanvar(y[valid]))
        var_res  = float(np.nanvar(residual[valid]))
        pct_reduction = round(100.0 * (1.0 - var_res / (var_orig + 1e-12)), 1)

        printInfo(
            f"[cognitionProcessor] regress_noise_channels: '{ch}' fitted on "
            f"{n_valid} samples — variance reduction {pct_reduction}%"
        )

        cleaned[ch] = residual

    return cleaned
