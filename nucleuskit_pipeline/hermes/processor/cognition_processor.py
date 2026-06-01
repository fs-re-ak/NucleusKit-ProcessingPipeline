"""
Cognition Processor

Computes cognitive metrics from EEG data using the cleaned temporal pipeline:
artefact rejection on T9/T10, bilateral temporal band-power averages for primary
metrics (Engagement, Focus, CognitiveLoad), and frontal/hemispheric channels for
secondary metrics (Frontal, Lateralization).

Feature traceability: per-window EEG band powers are written under the session's
``features/cognition/`` folder alongside artefact statistics and temporal band
power averages.

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

# Gaps in the hardware timestamp stream that are at least this long (seconds)
# are treated as true recording breaks; all windows that straddle a break are
# emitted as NaN rows (matching the convention used by shimmerResampler).
GAP_THRESHOLD_S = 5.0

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
# Artefact rejection
# ---------------------------------------------------------------------------

def reject_temporal_artefacts(filtered_df, hardware_invalid, sf=HermesDataInterface.SAMPLING_RATE):
    """
    Reject 1-second non-overlapping epochs on T9 and T10 using four signal-quality
    criteria from the cleaned EEG pipeline spec (cleanEEG.py / inspectEEG.py).

    An epoch is flagged if *either* channel fails *any* of the following:

        1. NaN fraction in raw signal (from hardware_invalid) > 20 %
        2. Peak absolute amplitude in filtered signal > 75 µV
        3. Peak-to-peak in filtered signal > 60 µV
        4. P(30–45 Hz) / P(1–45 Hz) via Welch > 0.30  (EMG contamination)

    Args:
        filtered_df: Bandpass-filtered DataFrame; must contain 'T9' and 'T10' columns.
        hardware_invalid: Boolean 1-D array (same length as filtered_df) marking
            samples that were NaN before filtering (hardware disconnects or gaps).
        sf: Sampling frequency in Hz (default 250).

    Returns:
        Tuple (rejected_mask, artefact_stats):
            - rejected_mask: boolean numpy array, True for every sample belonging
              to a rejected 1-second epoch.
            - artefact_stats: single-row DataFrame with summary counts written to
              ``features/cognition/artefactStats.csv``.
    """
    for ch in ('T9', 'T10'):
        if ch not in filtered_df.columns:
            printWarning(f"[cognitionProcessor] reject_temporal_artefacts: '{ch}' not found — skipping artefact rejection")
            n = len(filtered_df)
            return np.zeros(n, dtype=bool), None

    n_samples   = len(filtered_df)
    epoch_len   = int(sf)          # 1 s = 250 samples at 250 Hz
    n_epochs    = n_samples // epoch_len

    rejected_mask = np.zeros(n_samples, dtype=bool)

    t9  = filtered_df['T9'].to_numpy()
    t10 = filtered_df['T10'].to_numpy()

    # Per-criterion rejection counters (first criterion that triggered rejection)
    criterion_counts = {'nan': 0, 'peak_amp': 0, 'ptp': 0, 'emg_ratio': 0}

    for ei in range(n_epochs):
        s, e = ei * epoch_len, (ei + 1) * epoch_len
        epoch_bad   = False
        cause       = None

        for ch_name, ch_data in (('T9', t9), ('T10', t10)):
            # Criterion 1: NaN fraction in raw signal
            nan_frac = float(hardware_invalid[s:e].mean())
            if nan_frac > 0.20:
                cause = 'nan'
                epoch_bad = True
                break

            sig = ch_data[s:e]

            # Criterion 2: Peak absolute amplitude
            if np.nanmax(np.abs(sig)) > 75.0:
                cause = 'peak_amp'
                epoch_bad = True
                break

            # Criterion 3: Peak-to-peak
            if (np.nanmax(sig) - np.nanmin(sig)) > 60.0:
                cause = 'ptp'
                epoch_bad = True
                break

            # Criterion 4: EMG power ratio P(30–45) / P(1–45)
            try:
                freqs, psd = welch(sig, fs=sf, nperseg=epoch_len)
                mask_emg   = (freqs >= 30) & (freqs <= 45)
                mask_total = (freqs >=  1) & (freqs <= 45)
                p_emg   = np.trapz(psd[mask_emg],   freqs[mask_emg])
                p_total = np.trapz(psd[mask_total],  freqs[mask_total])
                if (p_emg / (p_total + 1e-12)) > 0.30:
                    cause = 'emg_ratio'
                    epoch_bad = True
                    break
            except Exception as exc:
                printWarning(f"[cognitionProcessor] Welch failed on artefact epoch {ei} channel {ch_name}: {exc}")

        if epoch_bad:
            rejected_mask[s:e] = True
            criterion_counts[cause] += 1

    n_rejected_epochs  = int(rejected_mask[:n_epochs * epoch_len]
                              .reshape(n_epochs, epoch_len).any(axis=1).sum())
    pct_epochs         = round(100.0 * n_rejected_epochs / max(n_epochs, 1), 1)
    n_samples_flagged  = int(rejected_mask.sum())
    pct_samples        = round(100.0 * n_samples_flagged / max(n_samples, 1), 1)

    printInfo(
        f"[cognitionProcessor] Artefact rejection: {n_rejected_epochs}/{n_epochs} epochs rejected "
        f"({pct_epochs} % of epochs, {pct_samples} % of samples) — "
        f"nan={criterion_counts['nan']}, peak={criterion_counts['peak_amp']}, "
        f"ptp={criterion_counts['ptp']}, emg={criterion_counts['emg_ratio']}"
    )

    artefact_stats = pd.DataFrame([{
        'n_epochs_total':         n_epochs,
        'n_epochs_rejected':      n_rejected_epochs,
        'pct_epochs_rejected':    pct_epochs,
        'n_rejected_by_nan':      criterion_counts['nan'],
        'n_rejected_by_peak_amp': criterion_counts['peak_amp'],
        'n_rejected_by_ptp':      criterion_counts['ptp'],
        'n_rejected_by_emg_ratio': criterion_counts['emg_ratio'],
        'n_samples_flagged':      n_samples_flagged,
        'pct_samples_flagged':    pct_samples,
    }])

    return rejected_mask, artefact_stats


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

        # Representative timestamp for this window: median of the recorded
        # hardware timestamps within the window.  The median is more robust
        # to per-sample jitter than a simple midpoint and correctly tracks
        # any long-term drift between the hardware clock and nominal rate.
        # Fall back to sample-index / sf when no timestamps are available.
        if timestamps is not None:
            ts_window = timestamps[start_idx:end_idx]
            timestamp = float(np.median(ts_window))
        else:
            timestamp = (start_idx + window_samples / 2) / sf

        # Windows where more than 25 % of samples are flagged (hardware-invalid
        # or artefact-rejected) are emitted as NaN rows so the timeline stays
        # intact while bad data is clearly marked.
        if hardware_invalid is not None and hardware_invalid[start_idx:end_idx].mean() > 0.25:
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
# Cognitive index computation
# ---------------------------------------------------------------------------

def compute_cognitive_indexes(df):
    """
    Compute cognitive metrics from cleaned EEG power bands.

    Primary metrics — bilateral temporal average of T9 and T10:
        Engagement   : avg_beta / (avg_alpha + avg_theta)      [Pope et al. 1995]
        Focus        : avg_beta / avg_alpha
        CognitiveLoad: (avg_theta × avg_beta) / avg_alpha²     [Borghini et al. 2014]

    Secondary metrics — frontal channels AF7 / AF8 and hemispheric channels:
        Frontal      : mean(β/(α+θ) for AF7, β/(α+θ) for AF8)
        Lateralization: log(clip(LeftHemi_eng, 1e-12)) − log(clip(RightHemi_eng, 1e-12))

    Args:
        df: DataFrame with columns [Timestamp, channel, band, power].

    Returns:
        Tuple (result, temporal_bands):
            result         — DataFrame [Timestamp, Engagement, Focus, CognitiveLoad,
                             Frontal, Lateralization], or None on failure.
            temporal_bands — DataFrame with per-timestamp bilateral temporal band
                             averages and relative powers (for feature storage), or
                             None on failure.
    """
    printInfo("[cognitionProcessor] Computing cognitive indexes")

    required_bands = {'alpha', 'beta', 'theta', 'delta', 'gamma'}
    present_bands  = set(df['band'].unique()) if 'band' in df.columns else set()
    missing_bands  = required_bands - present_bands
    if missing_bands:
        printError(f"[cognitionProcessor] compute_cognitive_indexes: missing bands {missing_bands}. "
                   f"Present: {present_bands}")
        return None, None

    try:
        df_pivot = df.pivot_table(
            index=['Timestamp', 'channel'],
            columns='band',
            values='power',
            aggfunc='mean',
        ).reset_index()
        df_pivot.columns.name = None
    except Exception as e:
        printError(f"[cognitionProcessor] pivot_table failed: {e}")
        printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None, None

    required_channels = {'T9', 'T10', 'AF7', 'AF8', 'LeftHemi', 'RightHemi'}
    present_channels  = set(df_pivot['channel'].unique())
    missing_channels  = required_channels - present_channels
    if missing_channels:
        printError(f"[cognitionProcessor] compute_cognitive_indexes: missing channels {missing_channels}. "
                   f"Present: {present_channels}")
        return None, None

    try:
        # --- Bilateral temporal averages (T9 + T10) ---
        temporal_df  = df_pivot[df_pivot['channel'].isin(['T9', 'T10'])]
        temporal_avg = temporal_df.groupby('Timestamp')[
            ['alpha', 'beta', 'theta', 'delta', 'gamma']
        ].mean()

        avg_alpha = temporal_avg['alpha']
        avg_beta  = temporal_avg['beta']
        avg_theta = temporal_avg['theta']
        avg_delta = temporal_avg['delta']
        avg_gamma = temporal_avg['gamma']
        all_power = avg_beta + avg_alpha + avg_theta + avg_delta + avg_gamma

        engagement  = avg_beta / (avg_alpha + avg_theta + 1e-8)
        focus       = avg_beta / (avg_alpha + 1e-8)
        cog_load    = (avg_theta * avg_beta) / (avg_alpha ** 2 + 1e-8)

        # --- Frontal engagement (AF7 + AF8 average) ---
        frontal_df         = df_pivot[df_pivot['channel'].isin(['AF7', 'AF8'])].copy()
        frontal_df['eng']  = frontal_df['beta'] / (frontal_df['alpha'] + frontal_df['theta'] + 1e-8)
        frontal_avg        = frontal_df.groupby('Timestamp')['eng'].mean()

        # --- Hemispheric lateralization (LeftHemi vs RightHemi) ---
        hemi_df        = df_pivot[df_pivot['channel'].isin(['LeftHemi', 'RightHemi'])].copy()
        hemi_df['eng'] = hemi_df['beta'] / (hemi_df['alpha'] + hemi_df['theta'] + 1e-8)

        left_eng  = hemi_df[hemi_df['channel'] == 'LeftHemi'].set_index('Timestamp')['eng']
        right_eng = hemi_df[hemi_df['channel'] == 'RightHemi'].set_index('Timestamp')['eng']
        lat       = (np.log(np.clip(left_eng,  1e-12, None)) -
                     np.log(np.clip(right_eng, 1e-12, None)))

        # --- Assemble results ---
        ts = temporal_avg.index

        result = pd.DataFrame({
            'Timestamp':    ts,
            'Engagement':   engagement.values,
            'Focus':        focus.values,
            'CognitiveLoad': cog_load.values,
            'Frontal':      frontal_avg.reindex(ts).values,
            'Lateralization': lat.reindex(ts).values,
        })

        # Temporal band powers saved to features (not in Cognition.csv)
        temporal_bands = pd.DataFrame({
            'Timestamp': ts,
            'avg_beta':  avg_beta.values,
            'avg_alpha': avg_alpha.values,
            'avg_theta': avg_theta.values,
            'avg_delta': avg_delta.values,
            'avg_gamma': avg_gamma.values,
            'all_power': all_power.values,
            'rel_beta':  (avg_beta  / (all_power + 1e-12)).values,
            'rel_alpha': (avg_alpha / (all_power + 1e-12)).values,
            'rel_theta': (avg_theta / (all_power + 1e-12)).values,
            'rel_delta': (avg_delta / (all_power + 1e-12)).values,
            'rel_gamma': (avg_gamma / (all_power + 1e-12)).values,
        })

    except Exception as e:
        printError(f"[cognitionProcessor] compute_cognitive_indexes failed: {e}")
        printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
        return None, None

    return result, temporal_bands


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
    Compute cognitive indexes from EEG data using the cleaned temporal pipeline.

    Pipeline steps
    --------------
    1. Load raw EEG; T9/T10 channels derived via midpoint re-reference.
    2. Detect hardware timestamp gaps (≥ 5 s).
    3. Bandpass + 60 Hz notch filter (0.5–45 Hz).
    4. Artefact rejection on T9/T10 — 1 s epochs, four signal-quality criteria.
       Bad epochs → sample mask; > 25 % of a 2 s analysis window flagged → NaN row.
    5. Welch PSD on all channels (2 s windows, 75 % overlap).
    6. Compute Engagement / Focus / CognitiveLoad from bilateral T9+T10 averages;
       Frontal from AF7+AF8; Lateralization from LeftHemi/RightHemi.
    7. Resample to 2 Hz (0.5 s grid).

    Incremental caching
    -------------------
    - ``results/Cognition.csv`` exists → step skipped entirely.
    - ``features/cognition/powerBands.csv`` exists *and* contains T9/T10 channels
      → band powers loaded from cache; steps 1–5 are skipped.
    - Any other case → full recomputation.

    Outputs
    -------
    - ``results/Cognition.csv``                     — primary cognitive metrics
    - ``features/cognition/powerBands.csv``         — per-window band powers (long format)
    - ``features/cognition/artefactStats.csv``      — epoch-level rejection summary
    - ``features/cognition/temporalBandPowers.csv`` — bilateral temporal band averages

    Args:
        recpath: Path to the recording directory.
    """
    printInfo("[cognitionProcessor] Computing Cognitive Indexes")

    cognitive_path   = os.path.join(recpath, 'results', 'Cognition.csv')
    features_dir     = os.path.join(recpath, 'features', 'cognition')
    powerbands_path  = os.path.join(features_dir, 'powerBands.csv')
    artefact_path    = os.path.join(features_dir, 'artefactStats.csv')
    temporal_bp_path = os.path.join(features_dir, 'temporalBandPowers.csv')

    if os.path.isfile(cognitive_path):
        printInfo("[cognitionProcessor] Cognitive indexes already computed, using cached results")
        return

    try:
        powerbands = None

        if os.path.isfile(powerbands_path):
            printInfo(
                f"[cognitionProcessor] Loading cached power bands from {powerbands_path}"
            )
            powerbands = pd.read_csv(powerbands_path)
            required_cols = {'Timestamp', 'channel', 'band', 'power'}
            if not required_cols.issubset(set(powerbands.columns)):
                printError(
                    f"[cognitionProcessor] Cached powerBands.csv missing columns "
                    f"{required_cols - set(powerbands.columns)} — will recompute"
                )
                powerbands = None
            elif not {'T9', 'T10'}.issubset(set(powerbands['channel'].unique())):
                printInfo(
                    "[cognitionProcessor] Cached powerBands.csv uses old channel layout "
                    "(missing T9/T10) — recomputing from EEG"
                )
                powerbands = None
            else:
                printInfo("[cognitionProcessor] Cache is valid, skipping EEG reprocessing")

        if powerbands is None:
            printInfo("[cognitionProcessor] Loading EEG data...")
            original_timestamps, eegData = HermesDataInterface(recpath).getEEG()

            if eegData is None:
                printError("[cognitionProcessor] HermesDataInterface.getEEG returned None — no EEG data available")
                return

            printInfo(f"[cognitionProcessor] EEG loaded: {len(eegData)} samples, columns: {list(eegData.columns)}")

            sf = HermesDataInterface.SAMPLING_RATE

            if original_timestamps is None:
                printWarning(
                    "[cognitionProcessor] Hardware timestamps unavailable — "
                    "falling back to synthetic timestamps (sample index / fs). "
                    "Gap detection is disabled."
                )
                original_timestamps = np.arange(len(eegData), dtype=float) / sf
                gap_sample_mask = np.zeros(len(eegData), dtype=bool)
            else:
                dt = np.diff(original_timestamps)
                gap_indices = np.where(dt >= GAP_THRESHOLD_S)[0]
                gap_sample_mask = np.zeros(len(original_timestamps), dtype=bool)
                if len(gap_indices):
                    printWarning(
                        f"[cognitionProcessor] {len(gap_indices)} hardware timestamp gap(s) "
                        f">= {GAP_THRESHOLD_S:.0f} s detected — affected windows will be NaN."
                    )
                    gap_sample_mask[gap_indices + 1] = True

            # Capture hardware invalids BEFORE any interpolation so artefact
            # rejection can assess raw NaN fraction per epoch.
            hardware_invalid = eegData.isna().any(axis=1).to_numpy() | gap_sample_mask

            printInfo("[cognitionProcessor] Filtering EEG (bandpass + notch)...")
            try:
                filtered_eeg = bandpass_filter(eegData, sf)
            except BaseException as e:
                printError(f"[cognitionProcessor] bandpass_filter failed: {type(e).__name__}: {e}")
                printError(f"[cognitionProcessor] Traceback:\n{traceback.format_exc()}")
                return

            printInfo("[cognitionProcessor] Running temporal artefact rejection...")
            artefact_mask, artefact_stats = reject_temporal_artefacts(
                filtered_eeg, hardware_invalid, sf
            )

            # Combine hardware invalids with artefact-rejected samples.
            # The 25 % window threshold in compute_eeg_power_bands treats both equally.
            combined_invalid = hardware_invalid | artefact_mask

            printInfo("[cognitionProcessor] Computing power bands...")
            powerbands = compute_eeg_power_bands(
                filtered_eeg,
                timestamps=original_timestamps,
                hardware_invalid=combined_invalid,
            )
            if powerbands is None:
                printError("[cognitionProcessor] compute_eeg_power_bands returned None — cannot proceed")
                return

            os.makedirs(features_dir, exist_ok=True)
            printInfo(f"[cognitionProcessor] Saving power bands to {powerbands_path}...")
            powerbands.to_csv(powerbands_path, index=False)

            if artefact_stats is not None:
                artefact_stats.to_csv(artefact_path, index=False)
                printInfo(f"[cognitionProcessor] Artefact stats saved to {artefact_path}")

        printInfo("[cognitionProcessor] Computing cognitive indexes...")
        result, temporal_bands = compute_cognitive_indexes(powerbands)
        if result is None:
            printError("[cognitionProcessor] compute_cognitive_indexes returned None — cannot proceed")
            return

        if temporal_bands is not None:
            os.makedirs(features_dir, exist_ok=True)
            temporal_bands.to_csv(temporal_bp_path, index=False)
            printInfo(f"[cognitionProcessor] Temporal band powers saved to {temporal_bp_path}")

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
