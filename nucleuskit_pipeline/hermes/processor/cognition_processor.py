"""
Cognition Processor

Computes cognitive metrics (engagement, frontal asymmetry) from EEG data,
including signal preprocessing, power band extraction, and index computation.

Feature traceability: per-window EEG band powers are written under the session's
``features/cognition/`` folder (``powerBands.csv``), separate from other processors.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os
import traceback
import pandas as pd
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch
from nucleuskit_pipeline.hermes.processor.data_interface import HermesDataInterface
from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError

# NOTE: np.seterr(all='raise') was previously set here. Removed because it
# converts benign floating-point events (e.g. 0/0 in normalisation) into
# exceptions that silently escape the pipeline's broad except-as-warning
# handler. Errors are now reported explicitly at each failure point instead.


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def fill_nans_with_interpolation(series):
    """Fill NaNs in a Series by linear interpolation."""
    return series.interpolate(method='linear')


def bandpass_filter(df, sf, lowcut=0.5, highcut=45, notch_freq=60):
    """
    Apply bandpass and notch filtering to EEG data.

    Args:
        df: DataFrame with one column per EEG channel
        sf: Sampling frequency in Hz

    Returns:
        Filtered DataFrame with the same columns.
    """
    nyq = 0.5 * sf

    if lowcut >= highcut or highcut >= nyq:
        raise ValueError(f"[cognitionProcessor] Invalid cutoff frequencies: lowcut={lowcut}, highcut={highcut}, nyq={nyq}")

    if notch_freq >= nyq:
        raise ValueError(f"[cognitionProcessor] Notch frequency {notch_freq} must be below Nyquist {nyq}")

    df = df.copy()

    for col in df.columns:
        series = df[col]

        if series.isna().any() or np.isinf(series.to_numpy()).any():
            printWarning(f"[cognitionProcessor] Column '{col}' contains NaNs or Infs — interpolating")

            if pd.isna(series.iloc[0]):
                first_valid = series.dropna().iloc[0] if not series.dropna().empty else 0
                series.iloc[0] = first_valid

            if pd.isna(series.iloc[-1]):
                last_valid = series.dropna().iloc[-1] if not series.dropna().empty else 0
                series.iloc[-1] = last_valid

            series = fill_nans_with_interpolation(series)
            df[col] = series

    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')

    if len(df) < (3 * max(len(a), len(b))):
        printWarning(f"[cognitionProcessor] Data too short for filtering ({len(df)} samples). Returning unfiltered data.")
        return df

    arr = filtfilt(b, a, df.to_numpy(), axis=0)

    b_notch, a_notch = iirnotch(notch_freq / nyq, 30)
    arr = filtfilt(b_notch, a_notch, arr, axis=0)

    return pd.DataFrame(arr, columns=df.columns)


def remove_statistical_outliers(df, threshold=3):
    """
    Detect and remove statistical outliers using z-score.

    Computed column-by-column to avoid allocating multiple full-size
    intermediate DataFrames at once (which can OOM-crash the process
    on large recordings via a C-level allocation failure that bypasses
    Python's exception handling).

    Args:
        df: DataFrame with one column per EEG channel

    Returns:
        Tuple (cleaned_df, mask) where cleaned_df has outlier rows removed and
        index reset, and mask is the boolean array indicating which of the
        *input* rows were kept (used by callers to propagate parallel arrays).
    """
    # Build a boolean keep-mask one column at a time (O(1) extra memory per column)
    mask = np.ones(len(df), dtype=bool)

    for col in df.columns:
        col_data = df[col].to_numpy()
        col_mean = np.nanmean(col_data)
        col_std  = np.nanstd(col_data)

        if col_std == 0 or np.isnan(col_std):
            printWarning(f"[cognitionProcessor] Column '{col}' has zero/NaN std — skipping outlier removal for this channel")
            continue

        z = np.abs((col_data - col_mean) / col_std)
        mask &= z < threshold

    cleaned = df[mask].reset_index(drop=True)

    removed = len(df) - len(cleaned)
    if removed > 0:
        printInfo(f"[cognitionProcessor] Removed {removed} outlier rows ({removed / len(df) * 100:.1f}%)")

    return cleaned, mask


def preprocess_eeg(df, sf=HermesDataInterface.SAMPLING_RATE,
                   timestamps=None, hardware_invalid=None):
    """
    Preprocess EEG data: filter and remove statistical outliers.

    Args:
        df: DataFrame with one column per EEG channel
        sf: Sampling frequency in Hz
        timestamps: Optional 1-D array of per-sample timestamps (seconds).
            Passed through with the same row-wise filtering applied to df.
        hardware_invalid: Optional boolean 1-D array marking samples that
            were NaN in the original recording (hardware disconnects).
            Passed through with the same row-wise filtering applied to df.

    Returns:
        Tuple (cleaned_df, timestamps, hardware_invalid).
        cleaned_df is None on failure (timestamps and hardware_invalid
        are also None in that case).
    """
    printInfo("[cognitionProcessor] Preprocessing EEG data")

    try:
        filtered = bandpass_filter(df, sf)
    except BaseException as e:
        printError(f"[cognitionProcessor] bandpass_filter failed: {type(e).__name__}: {e}")
        printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None, None, None

    try:
        cleaned, mask = remove_statistical_outliers(filtered)
    except BaseException as e:
        printError(f"[cognitionProcessor] remove_statistical_outliers failed: {type(e).__name__}: {e}")
        printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None, None, None

    if cleaned.empty:
        printError("[cognitionProcessor] preprocess_eeg: DataFrame is empty after outlier removal — "
                   "all rows were flagged as outliers. Check signal quality.")
        return None, None, None

    if timestamps is not None:
        timestamps = timestamps[mask]
    if hardware_invalid is not None:
        hardware_invalid = hardware_invalid[mask]

    printInfo(f"[cognitionProcessor] Preprocessing done: {len(cleaned)} samples remaining")
    return cleaned, timestamps, hardware_invalid


# ---------------------------------------------------------------------------
# Power band extraction
# ---------------------------------------------------------------------------

def compute_eeg_power_bands(df, sf=HermesDataInterface.SAMPLING_RATE, window_duration=2,
                            bands_definitions=[[0, 4], [4, 8], [8, 13], [13, 22], [30, 50]],
                            overlap=0.75, timestamps=None, hardware_invalid=None):
    """
    Compute EEG power in different frequency bands using Welch's method.

    Args:
        df: Preprocessed EEG DataFrame (samples × channels)
        sf: Sampling frequency (default from HermesDataInterface)
        window_duration: Duration of analysis window in seconds
        bands_definitions: List of [low, high] frequency ranges for each band
        overlap: Fraction of window overlap (0–1)
        timestamps: Optional 1-D array of per-sample timestamps (seconds).
            When provided, the Timestamp of each window is set to the midpoint
            of the first and last sample's original timestamps rather than being
            derived from the post-preprocessing array index.
        hardware_invalid: Optional boolean 1-D array (same length as df).
            Windows where any sample is marked True are emitted as NaN rows
            instead of running Welch, preserving timeline alignment.

    Returns:
        DataFrame with columns [Timestamp, channel, band, power], or None on failure.
        Windows with lost samples have NaN in the power column.
    """
    printInfo("[cognitionProcessor] Computing EEG power bands")

    channel_names = df.columns.tolist()
    arr = df.to_numpy()
    n_samples = len(df)

    window_samples = int(window_duration * sf)
    step_samples = int(window_samples * (1 - overlap))
    n_windows = (n_samples - window_samples) // step_samples + 1

    if n_windows <= 0:
        printError(f"[cognitionProcessor] compute_eeg_power_bands: not enough data for a single window "
                   f"(samples={n_samples}, window={window_samples}). Recording may be too short.")
        return None

    printInfo(f"[cognitionProcessor] Computing power bands: {n_windows} windows over {len(channel_names)} channels")

    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    rows = []

    for win_idx in range(n_windows):
        start_idx = win_idx * step_samples
        end_idx = start_idx + window_samples

        # Timestamp at the center of the window, using original recording time
        # when available, otherwise fall back to sample-index / sf.
        if timestamps is not None:
            ts_window = timestamps[start_idx:end_idx]
            timestamp = float((ts_window[0] + ts_window[-1]) / 2.0)
        else:
            timestamp = (start_idx + window_samples / 2) / sf

        # Windows that contain hardware-invalid samples are emitted as NaN rows
        # so downstream consumers can identify and exclude them while the
        # timeline (timestamp column) remains intact.
        if hardware_invalid is not None and hardware_invalid[start_idx:end_idx].any():
            for ch_name in channel_names:
                for band_name in band_names:
                    rows.append({
                        'Timestamp': timestamp,
                        'channel': ch_name,
                        'band': band_name,
                        'power': np.nan,
                    })
            continue

        window_data = arr[start_idx:end_idx, :]

        for ch_idx, ch_name in enumerate(channel_names):
            try:
                freqs, psd = welch(window_data[:, ch_idx], fs=sf, nperseg=min(256, window_samples))
            except Exception as e:
                printWarning(f"[cognitionProcessor] Welch failed on window {win_idx} channel {ch_name}: {e}")
                continue

            for band_idx, (low, high) in enumerate(bands_definitions):
                freq_mask = (freqs >= low) & (freqs <= high)
                power = np.trapz(psd[freq_mask], freqs[freq_mask]) if freq_mask.any() else 0.0

                rows.append({
                    'Timestamp': timestamp,
                    'channel': ch_name,
                    'band': band_names[band_idx],
                    'power': power,
                })

    if not rows:
        printError("[cognitionProcessor] compute_eeg_power_bands: no rows produced — "
                   "Welch failed on every window/channel combination.")
        return None

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engagement index computation
# ---------------------------------------------------------------------------

def compute_engagement_indexes(df):
    """
    Compute per-channel engagement indexes and derive cognitive metrics.

    Engagement = Beta / (Alpha + Theta), computed per channel, then pivoted
    to wide format and used to derive:
        Intertemporal : engagement on the Temporal channel
        Lateralization: log(LeftHemi) - log(RightHemi)
        Frontal       : mean engagement of AF7 and AF8
        Engagement    : mean engagement across AF7, AF8 and Temporal

    Args:
        df: DataFrame with columns [Timestamp, channel, band, power]

    Returns:
        DataFrame with columns [Timestamp, Engagement, Intertemporal,
        Lateralization, Frontal], or None on failure.
    """
    printInfo("[cognitionProcessor] Computing engagement indexes")

    required_bands = {'alpha', 'beta', 'theta'}
    present_bands = set(df['band'].unique()) if 'band' in df.columns else set()
    missing_bands = required_bands - present_bands
    if missing_bands:
        printError(f"[cognitionProcessor] compute_engagement_indexes: missing bands {missing_bands} "
                   f"in power-bands DataFrame. Present: {present_bands}")
        return None

    try:
        df_pivot = df.pivot_table(
            index=['Timestamp', 'channel'],
            columns='band',
            values='power',
            aggfunc='mean'
        ).reset_index()
    except Exception as e:
        printError(f"[cognitionProcessor] pivot_table failed: {e}")
        printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None

    df_pivot['engagement'] = df_pivot['beta'] / (df_pivot['alpha'] + df_pivot['theta'] + 1e-8)

    try:
        eng_wide = df_pivot[['Timestamp', 'channel', 'engagement']].pivot_table(
            index='Timestamp', columns='channel', values='engagement'
        ).reset_index()
        eng_wide.columns.name = None
    except Exception as e:
        printError(f"[cognitionProcessor] channel pivot failed: {e}")
        printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None

    required_channels = {'AF7', 'AF8', 'Temporal', 'LeftHemi', 'RightHemi'}
    missing_channels = required_channels - set(eng_wide.columns)
    if missing_channels:
        printError(f"[cognitionProcessor] compute_engagement_indexes: missing channels {missing_channels} "
                   f"after pivot. Present: {set(eng_wide.columns)}")
        return None

    eng_wide['Intertemporal'] = eng_wide['Temporal']

    eng_wide['Lateralization'] = (
        np.log(np.clip(eng_wide['LeftHemi'], 1e-12, None)) -
        np.log(np.clip(eng_wide['RightHemi'], 1e-12, None))
    )

    eng_wide['Frontal'] = (eng_wide['AF7'] + eng_wide['AF8']) / 2

    eng_wide['Engagement'] = eng_wide[['AF7', 'AF8', 'Temporal']].mean(axis=1)

    return eng_wide[['Timestamp', 'Engagement', 'Intertemporal', 'Lateralization', 'Frontal']]


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _simple_resample(df, target_interval=0.5):
    """
    Resample a DataFrame to a fixed time interval using linear interpolation.

    NaN values in source columns are preserved: output points that fall
    entirely within a NaN gap (no valid source neighbour on both sides) are
    emitted as NaN rather than being silently interpolated through.

    Args:
        df: DataFrame with a Timestamp column
        target_interval: Target time interval in seconds

    Returns:
        Resampled DataFrame.
    """
    if 'Timestamp' not in df.columns:
        return df

    src_ts = df['Timestamp'].to_numpy(dtype=float)
    max_time = src_ts.max()
    new_timestamps = np.arange(0, max_time + target_interval, target_interval)

    resampled = {'Timestamp': new_timestamps}
    for col in df.columns:
        if col == 'Timestamp':
            continue

        y = df[col].to_numpy(dtype=float)
        valid = ~np.isnan(y)

        if not valid.any():
            resampled[col] = np.full(len(new_timestamps), np.nan)
            continue

        # Interpolate only between valid (non-NaN) source points.
        interp_values = np.interp(new_timestamps, src_ts[valid], y[valid],
                                  left=np.nan, right=np.nan)

        # Mark output points that fall entirely inside a NaN gap.  For each
        # query point, find the bracketing valid source points; if the gap
        # between them exceeds the expected step size (indicating a NaN source
        # row between them), force the output to NaN.
        valid_ts = src_ts[valid]
        # expected maximum distance between two adjacent valid source timestamps
        # (generous: two output steps to avoid false positives from minor jitter)
        expected_step = target_interval * 2

        # For each output point find the bracketing valid source indices.
        left_idx = np.searchsorted(valid_ts, new_timestamps, side='right') - 1
        right_idx = left_idx + 1

        # Points that are inside the valid range on both sides
        has_bracket = (left_idx >= 0) & (right_idx < len(valid_ts))
        # Gap between the two enclosing valid source points (vectorised)
        safe_l = np.clip(left_idx, 0, len(valid_ts) - 1)
        safe_r = np.clip(right_idx, 0, len(valid_ts) - 1)
        gap = valid_ts[safe_r] - valid_ts[safe_l]
        in_gap = has_bracket & (gap > expected_step)

        interp_values[in_gap] = np.nan
        resampled[col] = interp_values

    return pd.DataFrame(resampled)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def computeCognitiveIndexes(recpath):
    """
    Compute cognitive indexes from EEG data.

    Incremental outputs:
    - If ``results/Cognition.csv`` exists, the step is skipped.
    - If cognition must be rebuilt but ``features/cognition/powerBands.csv`` exists,
      band powers are loaded from disk and not recomputed.
    - ``powerBands.csv`` is only written when it is missing.

    Args:
        recpath: Path to the recording directory
    """
    printInfo("[cognitionProcessor] Computing Cognitive Indexes")

    cognitive_path = os.path.join(recpath, 'results', 'Cognition.csv')
    features_dir = os.path.join(recpath, "features", "cognition")
    powerbands_path = os.path.join(features_dir, "powerBands.csv")

    if os.path.isfile(cognitive_path):
        printInfo("[cognitionProcessor] Cognitive indexes already computed, using cached results")
        return

    try:
        powerbands = None

        if os.path.isfile(powerbands_path):
            printInfo(
                f"[cognitionProcessor] Loading cached power bands from {powerbands_path} "
                "(will not recompute powerBands.csv)"
            )
            powerbands = pd.read_csv(powerbands_path)
            required = {"Timestamp", "channel", "band", "power"}
            if not required.issubset(set(powerbands.columns)):
                printError(
                    f"[cognitionProcessor] Cached powerBands.csv missing columns "
                    f"{required - set(powerbands.columns)} — will recompute from EEG"
                )
                powerbands = None

        if powerbands is None:
            printInfo("[cognitionProcessor] Loading EEG data...")
            eegData = HermesDataInterface(recpath).getEEG()

            if eegData is None:
                printError("[cognitionProcessor] HermesDataInterface.getEEG returned None — no EEG data available")
                return

            printInfo(f"[cognitionProcessor] EEG loaded: {len(eegData)} samples, columns: {list(eegData.columns)}")

            sf = HermesDataInterface.SAMPLING_RATE
            # Original per-sample timestamps (seconds from recording start).
            original_timestamps = np.arange(len(eegData), dtype=float) / sf
            # Samples that were hardware-invalid (NaN) before any interpolation.
            hardware_invalid = eegData.isna().any(axis=1).to_numpy()

            printInfo("[cognitionProcessor] Preprocessing EEG...")
            eegData, original_timestamps, hardware_invalid = preprocess_eeg(
                eegData, sf=sf,
                timestamps=original_timestamps,
                hardware_invalid=hardware_invalid,
            )
            if eegData is None:
                printError("[cognitionProcessor] preprocess_eeg returned None — cannot proceed")
                return

            printInfo("[cognitionProcessor] Computing power bands...")
            powerbands = compute_eeg_power_bands(
                eegData,
                timestamps=original_timestamps,
                hardware_invalid=hardware_invalid,
            )
            if powerbands is None:
                printError("[cognitionProcessor] compute_eeg_power_bands returned None — cannot proceed")
                return

            os.makedirs(features_dir, exist_ok=True)
            printInfo(f"[cognitionProcessor] Saving power bands to {powerbands_path}...")
            powerbands.to_csv(powerbands_path, index=False)

        printInfo("[cognitionProcessor] Computing engagement indexes...")
        result = compute_engagement_indexes(powerbands)
        if result is None:
            printError("[cognitionProcessor] compute_engagement_indexes returned None — cannot proceed")
            return

        result = _simple_resample(result, target_interval=0.5)

        os.makedirs(os.path.dirname(cognitive_path), exist_ok=True)
        printInfo(f"[cognitionProcessor] Writing results to {cognitive_path}...")
        result.to_csv(cognitive_path, index=False)

        printInfo("[cognitionProcessor] Cognitive index computation completed")

    except Exception as e:
        printError(f"[cognitionProcessor] Unhandled error: {e}")
        printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")


def loadCognitiveIndexes(src):
    """
    Load cognitive index data from CSV file.

    Args:
        src: Path to recording directory

    Returns:
        pandas DataFrame containing cognitive indexes, or None if not found.
    """
    cognitive_path = os.path.join(src, "results", "Cognition.csv")

    if not os.path.isfile(cognitive_path):
        printWarning(f"[cognitionProcessor] Cognitive file not found: {cognitive_path}")
        return None

    return pd.read_csv(cognitive_path)
