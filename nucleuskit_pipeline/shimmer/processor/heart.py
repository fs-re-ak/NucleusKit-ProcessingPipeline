"""
Heart rate dynamics from Shimmer PPG.

Before peak detection the raw signal is resampled onto a strict 51.2 Hz
grid using the original hardware timestamps (see shimmerResampler).  The
grid signal is then passed through a two-stage artifact rejection pipeline
before being upsampled to 256 Hz for NeuroKit2 (Elgendi algorithm) peak
detection.  Peaks inside hardware-gap or rejected regions are discarded.

Artifact rejection — two-stage approach
----------------------------------------
Shimmer PPG amplitude can vary legitimately between recording segments
(e.g. when the sensor is repositioned on the wrist).  A single global
amplitude threshold incorrectly rejects valid high-amplitude segments or —
when the ``REJECT_MAX_PCT`` safety cap fires — leaves all artifacts
untouched.  The two-stage approach separates the two distinct failure modes:

Stage 1 — Spectral Quality Index (SQI)
    Each sliding window is scored by the fraction of its power that falls
    within the cardiac band (``SQI_CARDIAC_LO``–``SQI_CARDIAC_HI`` Hz).
    Windows below ``SQI_THRESHOLD`` are classified as non-PPG (broadband
    noise, motion baseline, ADC glitches, flat lines) and blanked to NaN
    regardless of amplitude.  This gate is amplitude-invariant: stable
    noise at any level is caught because it carries no cardiac-frequency
    power.

Stage 2 — Local MAD glitch detector
    Within windows that passed Stage 1, sudden amplitude spikes or clipping
    transients are caught by a rolling MAD-based z-score.  Using a local
    (rolling) window instead of a global statistic means that valid PPG at
    a different amplitude regime is judged against its own neighbourhood
    and is not incorrectly penalised.

Outputs saved to ``features/ppg/``:
    ``ppg_resampled.csv``  — artifact-cleaned signal at 51.2 Hz; NaN in gaps
    ``ppg_normalized.csv`` — rolling z-score of the cleaned signal (same grid);
        this is what NeuroKit2 peak detection actually operates on
    ``ppg_rejected.png``   — three-colour rejection map
        (black = valid, orange = SQI-rejected, red = MAD-rejected)
    ``ppg_overview.png``   — cleaned signal with detected peak markers

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation

from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError
from nucleuskit_pipeline.shimmer.processor.resampler import resample_to_grid, normalise_timestamps_to_seconds

# -------------------------------------------------------------------
# Signal constants
# -------------------------------------------------------------------
SAMPLING_RATE  = 51.2    # native Shimmer PPG sample rate (Hz)

# ------------------------------------------------------------------
# Stage 1 — Spectral Quality Index (SQI) constants
#
# Each sliding window of SQI_WINDOW_S seconds is scored by the
# fraction of its power that lies in the cardiac band.  Windows below
# SQI_THRESHOLD are classified as non-PPG and blanked before the MAD
# stage runs, preventing stable noise from being treated as valid PPG.
#
# SQI_WINDOW_S  : window length in seconds.  8 s gives 0.125 Hz
#   frequency resolution — well below the 0.5 Hz lower cardiac bound.
#   Shorter windows improve temporal localisation but reduce frequency
#   resolution.
# SQI_CARDIAC_LO/HI : cardiac band limits (Hz).  0.5–4 Hz covers the
#   fundamental (30–240 bpm) plus the second harmonic for most resting
#   and exercise heart rates.
# SQI_THRESHOLD : minimum cardiac-band power fraction for a window to
#   be accepted.  Lower = more permissive.  Raise toward 0.40 if you
#   observe stable noise leaking through; lower toward 0.20 if valid
#   PPG is being over-rejected (e.g. very high heart rates or arrhythmia).
# ------------------------------------------------------------------
SQI_WINDOW_S   = 8.0    # analysis window (s); 6–12 s are reasonable
SQI_CARDIAC_LO = 0.5    # cardiac band lower bound (Hz)
SQI_CARDIAC_HI = 4.0    # cardiac band upper bound (Hz)
SQI_THRESHOLD  = 0.30   # minimum cardiac-band power fraction

# ------------------------------------------------------------------
# Stage 2 — Local (rolling) MAD glitch detector constants
#
# We use the median and MAD rather than mean/std because the squared
# PPG is right-skewed; global statistics would flag normal systolic
# peaks.  Using a rolling window (REJECT_WINDOW_S) instead of a single
# global statistic ensures that valid PPG at a different amplitude
# level — e.g. after the sensor is repositioned — is judged against
# its own neighbourhood and is not incorrectly flagged.
#
# Z_THRESHOLD     : MAD-based |z| threshold; genuine glitches >> 4.
# REJECT_MARGIN_S : seconds blanked around each detected glitch centre.
# REJECT_MAX_PCT  : if Stage-2 rejection would exceed this fraction of
#   the signal after Stage 1, the MAD detector has likely found no real
#   glitch; rejection is skipped to preserve valid signal.
# REJECT_WINDOW_S : rolling window for local median / MAD (seconds).
#   30–120 s are all reasonable; shorter = more adaptive.
# ------------------------------------------------------------------
Z_THRESHOLD     = 4.0    # MAD-based |z| threshold; genuine glitches >> 4
REJECT_MARGIN_S = 5.0    # seconds to blank around each glitch centre
REJECT_MAX_PCT  = 15.0   # abort MAD rejection if it would exceed this % of signal
REJECT_WINDOW_S = 60.0   # rolling window for local median / MAD (s)

# Elgendi peak detection requires ≥100 Hz for reliable operation.
# Upsample to this rate before processing so the algorithm has adequate resolution.
_UPSAMPLE_RATE = 256.0

# ------------------------------------------------------------------
# Amplitude normalisation before NeuroKit2 peak detection
#
# NeuroKit2's Elgendi algorithm sets its peak-detection threshold using
# mean(ma_beat), a global average of the beat-window moving average.
# When the recording contains two amplitude regimes — e.g. a brief
# high-amplitude burst while the sensor is being positioned followed by
# a stable low-amplitude segment — the burst region dominates the global
# threshold and the algorithm misses all peaks in the stable segment.
#
# Rolling z-score normalisation (window = NORMALIZE_WINDOW_S seconds)
# equalises the local amplitude before the signal is upsampled and fed
# to NeuroKit2.  Each sample is centred and scaled relative to its own
# neighbourhood, making the Elgendi threshold independent of absolute
# amplitude.  The normalised signal is used only for peak detection;
# the original (non-normalised) signal is kept for all other purposes.
#
# NORMALIZE_WINDOW_S: rolling window in seconds.  Should be longer
#   than two complete cardiac cycles (~1.5 s at max heart rate) and
#   shorter than the expected amplitude-regime duration so it adapts
#   within each regime.  10 s (~10 beats at 60 bpm) is a good default.
# ------------------------------------------------------------------
NORMALIZE_WINDOW_S = 10.0   # rolling z-score window for NK2 input (s)

# Physiological IBI bounds (ms). IBIs outside this window are missed/spurious
# peaks and must be excluded before computing RMSSD.
_IBI_MIN_MS = 300.0    # ~200 bpm
_IBI_MAX_MS = 2000.0   # ~30 bpm

# Adaptive missing-beat filler constants (used by _fill_missing_beats).
MISSING_BEAT_HISTORY     = 30     # beats used to estimate the rolling median IBI
MISSING_BEAT_RATIO       = 1.5    # IBI > ratio * recent median => missed beat(s)
MISSING_BEAT_MAX_FILL_S  = 20.0   # gaps longer than this are not filled
MISSING_BEAT_MIN_HISTORY = 10     # require at least this many IBIs before adapting

# Shimmer CSV: column 0 = timestamp (ms), column 5 = PPG
_SHIMMER_TIMESTAMP_COL = 0


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def _load_shimmer_ppg_signal(rec_path):
    """
    Load timestamp + PPG from ``rawData/shimmer.csv`` or legacy files.

    Returns a tuple (timestamps_s, ppg_array), or (None, None) if no
    usable file is found.
    """
    raw_dir = os.path.join(rec_path, "rawData")
    candidates = [
        ("shimmer.csv",      5),
        ("rawShimmer_0.csv", 5),
        ("ppg.tmp",          1),
        ("ppg.csv",          1),
    ]
    for name, ppg_col in candidates:
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, header=None)
        except Exception:
            continue
        if df.shape[1] <= ppg_col:
            continue
        timestamps = df[_SHIMMER_TIMESTAMP_COL].values.astype(float)
        ppg        = df[ppg_col].values.astype(float)
        timestamps = normalise_timestamps_to_seconds(timestamps)
        return timestamps, ppg

    printWarning(
        f"[heartProcessor] No Shimmer PPG file in {raw_dir} "
        "(expected shimmer.csv, rawShimmer_0.csv, ppg.tmp, or ppg.csv). "
        "Shimmer was likely not used for this recording."
    )
    return None, None


# -------------------------------------------------------------------
# Physiological IBI filtering
# -------------------------------------------------------------------

def _filter_physiological_ibi(peak_indices, fs):
    """
    Drop spurious peaks whose IBI is physiologically too short.

    When two consecutive peaks are separated by less than _IBI_MIN_MS
    (~200 bpm), the second one is a duplicate detection and is removed.
    Long-IBI handling (missed beats) is delegated to ``_fill_missing_beats``,
    which uses adaptive thresholds and gap-mask awareness.
    """
    peaks = list(np.asarray(peak_indices, dtype=float))
    i = 0
    while i < len(peaks) - 1:
        ibi_ms = (peaks[i + 1] - peaks[i]) / fs * 1000.0
        if ibi_ms < _IBI_MIN_MS:
            del peaks[i + 1]
        else:
            i += 1
    return np.round(peaks).astype(int)


# -------------------------------------------------------------------
# Adaptive missing-beat filler
# -------------------------------------------------------------------

def _fill_missing_beats(
    peak_idx,
    fs,
    gap_mask=None,
    history=MISSING_BEAT_HISTORY,
    min_history=MISSING_BEAT_MIN_HISTORY,
    ratio=MISSING_BEAT_RATIO,
    max_fill_s=MISSING_BEAT_MAX_FILL_S,
):
    """Fill short stretches of missed beats with synthetic peaks.

    Walks the peak series once. For each pair (p[i], p[i+1]), the IBI is
    compared to the rolling median of the previous *history* accepted IBIs.
    If the IBI is more than *ratio* times the recent median, the gap is
    treated as one or more missed beats and filled with evenly-spaced
    synthetic peaks. Gaps longer than *max_fill_s* or that overlap the
    *gap_mask* (hardware gaps + glitch-rejected regions) are left untouched.

    History guard against double-detection
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    After rolling z-score normalisation the PPG diastolic notch can become
    prominent enough for NeuroKit2 to detect it as a second peak within the
    same cardiac cycle, producing spurious IBIs of ~350–600 ms.  These pass
    the physiological lower bound (``_IBI_MIN_MS`` = 300 ms) but would bias
    the rolling median downward if added to ``ibi_history``.

    As ``recent_median`` drifts toward half the true value the long-gap
    threshold drops below genuine IBIs, causing synthetic peaks to be
    inserted between every real beat and the apparent heart rate to double.

    To prevent this, only IBIs that are at least ``recent_median / ratio``
    (the symmetric lower bound of the long-gap detector) are admitted into
    the history.  Shorter IBIs are still walked past without modification
    — the real peak is kept and the spurious one is NOT removed here
    (removal is the responsibility of ``_filter_physiological_ibi``) — but
    they do not corrupt the rolling median used by the filler.

    Parameters
    ----------
    peak_idx : array-like
        Peak sample indices in the *fs* domain.
    fs : float
        Sample rate of the index domain (Hz).
    gap_mask : np.ndarray[bool] or None
        True at every sample that belongs to a hardware gap or
        artifact-rejected region. A long IBI that straddles any such
        sample is left unfilled because the absence of beats there
        reflects legitimate missing signal.
    history : int
        Number of recent IBIs used to compute the rolling median.
    min_history : int
        Minimum accepted IBIs required before adaptive detection kicks
        in. Before this, the global median of all observed IBIs is used
        as a warm-up estimate so the early signal is still cleaned.
    ratio : float
        IBI / recent_median threshold above which one or more beats are
        considered missing.  The symmetric lower guard for the history
        is ``recent_median / ratio``.
    max_fill_s : float
        Gaps longer than this duration are not filled (treated as long
        dropouts rather than missed beats).

    Returns
    -------
    peak_idx_filled : np.ndarray[int]
        Peak indices including the inserted synthetic peaks, sorted.
    synthetic_mask : np.ndarray[bool]
        Same length as *peak_idx_filled*; True at every inserted
        synthetic peak, False at every real peak.
    n_inserted : int
        Total number of synthetic peaks inserted.
    """
    peaks = np.asarray(peak_idx, dtype=float).tolist()
    if len(peaks) < 2:
        return (
            np.asarray(peaks, dtype=int),
            np.zeros(len(peaks), dtype=bool),
            0,
        )

    is_synth = [False] * len(peaks)
    ibi_history = []  # in samples; updated as we walk through accepted IBIs

    global_median_ibi = float(np.median(np.diff(peaks))) if len(peaks) > 1 else 0.0
    max_fill_samples = max_fill_s * fs

    gap_mask_arr = None
    n_gap = 0
    if gap_mask is not None and np.any(gap_mask):
        gap_mask_arr = np.asarray(gap_mask, dtype=bool)
        n_gap = len(gap_mask_arr)

    n_inserted = 0
    i = 0
    while i < len(peaks) - 1:
        ibi = peaks[i + 1] - peaks[i]

        if len(ibi_history) >= min_history:
            recent_median = float(np.median(ibi_history[-history:]))
        else:
            recent_median = global_median_ibi

        if recent_median <= 0:
            ibi_history.append(ibi)
            i += 1
            continue

        is_long_gap = ibi > ratio * recent_median
        is_within_cap = ibi <= max_fill_samples

        overlaps_gap = False
        if is_long_gap and is_within_cap and gap_mask_arr is not None:
            lo = int(max(0, np.floor(peaks[i])))
            hi = int(min(n_gap, np.ceil(peaks[i + 1]) + 1))
            if lo < hi and bool(gap_mask_arr[lo:hi].any()):
                overlaps_gap = True

        if is_long_gap and is_within_cap and not overlaps_gap:
            n_missing = max(1, int(round(ibi / recent_median)) - 1)
            step = ibi / (n_missing + 1)
            for k in range(1, n_missing + 1):
                synth = peaks[i] + k * step
                peaks.insert(i + k, synth)
                is_synth.insert(i + k, True)
            n_inserted += n_missing
            for k in range(n_missing + 1):
                ibi_history.append(step)
            i += n_missing + 1
        else:
            # Admit only IBIs above the symmetric lower bound of the long-gap
            # detector.  IBIs shorter than recent_median / ratio are likely
            # diastolic-notch or noise double-detections; admitting them would
            # drift recent_median downward and trigger spurious synthetic fills
            # for every subsequent genuine heartbeat interval.
            if ibi >= recent_median / ratio:
                ibi_history.append(ibi)
            i += 1

    peaks_arr = np.round(np.asarray(peaks, dtype=float)).astype(int)
    synth_arr = np.asarray(is_synth, dtype=bool)
    return peaks_arr, synth_arr, n_inserted


# -------------------------------------------------------------------
# Amplitude normalisation for peak detection
# -------------------------------------------------------------------

def _normalize_ppg_for_detection(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Apply rolling z-score normalisation to equalise amplitude across regimes.

    NeuroKit2's Elgendi algorithm computes a *global* ``mean(ma_beat)`` to
    set its peak-detection threshold.  When a recording contains segments at
    different PPG amplitudes (e.g. sensor being positioned at the start of
    recording), the high-amplitude segment dominates the threshold and the
    algorithm misses all peaks in the lower-amplitude segment.

    Rolling z-score normalisation resolves this by centering and scaling
    each sample relative to its local neighbourhood.  After normalisation
    every amplitude regime has unit variance, and Elgendi's global threshold
    works correctly for all of them.

    The normalised signal is only used for peak detection; the original
    signal is preserved for all other processing and output.

    Parameters
    ----------
    ppg : np.ndarray
        PPG signal to normalise.  Must be free of NaN (apply ffill/bfill
        before calling this function).
    fs : float
        Sampling rate of *ppg* in Hz.

    Returns
    -------
    ppg_norm : np.ndarray
        Rolling z-score normalised signal, same length as *ppg*.
        Edge samples (first / last half-window) are normalised against
        a partial window via ``min_periods=3``.
        A segment where std is zero (flat line) is left at 0.0.
    """
    win = max(3, int(round(NORMALIZE_WINDOW_S * fs)))
    s         = pd.Series(ppg)
    roll_mean = s.rolling(win, center=True, min_periods=3).mean()
    roll_std  = s.rolling(win, center=True, min_periods=3).std(ddof=0)
    # Flat segments (std == 0) produce NaN after division; replace with 0.0
    # so they do not create spurious peaks downstream.
    roll_std = roll_std.where(roll_std > 0).fillna(1.0)
    return ((s - roll_mean) / roll_std).values


# -------------------------------------------------------------------
# Peak detection
# -------------------------------------------------------------------

def _detect_ppg_peaks(ppg_raw, fs=SAMPLING_RATE, gap_mask=None):
    """
    Upsample raw PPG to _UPSAMPLE_RATE, run NeuroKit2 processing, and
    return the cleaned signal together with peak indices in both the
    upsampled and original-rate domains.

    NaN values in *ppg_raw* (large hardware gaps) are forward-filled
    before NeuroKit2 so the algorithm receives a continuous signal. Any
    peaks detected inside a gap region are discarded afterwards using
    *gap_mask* upsampled to the 256 Hz domain.

    Parameters
    ----------
    ppg_raw  : np.ndarray — PPG at *fs* Hz; may contain NaN in large gaps.
    fs       : float      — sample rate of *ppg_raw*.
    gap_mask : np.ndarray[bool] or None
               True at every sample of *ppg_raw* that belongs to a large
               hardware gap. Peaks falling in those regions are discarded.

    Returns
    -------
    ppg_clean_256 : np.ndarray — cleaned PPG at _UPSAMPLE_RATE; NaN in gap regions.
    peak_idx_256  : np.ndarray — peak sample indices in the 256 Hz domain (gaps removed).
    peak_idx_orig : np.ndarray — peak sample indices mapped back to *fs*.
    t_upsampled   : np.ndarray — time axis (s) for the 256 Hz signal.
    gap_mask_256  : np.ndarray[bool] — gap_mask upsampled to 256 Hz.
    """
    n_original  = len(ppg_raw)
    t_original  = np.linspace(0, n_original / fs, n_original)
    n_up        = int(n_original * _UPSAMPLE_RATE / fs)
    t_upsampled = np.linspace(0, n_original / fs, n_up)

    # Forward-fill NaN so the cubic upsampling produces a continuous signal.
    ppg_filled = pd.Series(ppg_raw).ffill().bfill().values

    # Normalise amplitude before passing to NeuroKit2 so that Elgendi's
    # global threshold works correctly when the recording contains segments
    # at different amplitude levels.  The normalised copy is used only for
    # peak detection; ppg_filled is kept for all other processing.
    ppg_for_nk = _normalize_ppg_for_detection(ppg_filled, fs)

    interpolator  = interp1d(t_original, ppg_for_nk, kind="cubic",
                             fill_value="extrapolate")
    ppg_upsampled = interpolator(t_upsampled)

    signals, info  = nk.ppg_process(ppg_upsampled, sampling_rate=_UPSAMPLE_RATE)
    peak_idx_256   = info["PPG_Peaks"]
    ppg_clean_256  = signals["PPG_Clean"].values

    # Build gap mask at 256 Hz
    if gap_mask is not None and np.any(gap_mask):
        gap_mask_256 = sp_signal.resample(gap_mask.astype(float), n_up) > 0.5
    else:
        gap_mask_256 = np.zeros(n_up, dtype=bool)

    # Discard peaks that fall inside large-gap regions
    if np.any(gap_mask_256):
        in_gap       = gap_mask_256[np.clip(peak_idx_256, 0, n_up - 1)]
        peak_idx_256 = peak_idx_256[~in_gap]

    # NaN-out gap regions in the cleaned PPG for the diagnostic figure
    if np.any(gap_mask_256):
        ppg_clean_256 = ppg_clean_256.copy()
        ppg_clean_256[gap_mask_256] = np.nan

    # Convert upsampled-domain indices back to original-domain indices so that
    # downstream timestamp reconstruction is consistent with the original length.
    peak_idx_orig = np.round(peak_idx_256 * fs / _UPSAMPLE_RATE).astype(int)

    return ppg_clean_256, peak_idx_256, peak_idx_orig, t_upsampled, gap_mask_256


# -------------------------------------------------------------------
# Sliding-window HRV
# -------------------------------------------------------------------

def _sliding_hrv_dataframe(
    peak_indices,
    fs=SAMPLING_RATE,
    fs_out=2.0,
    window=30.0,
    min_beats=10,
    gap_mask=None,
):
    """Compute a sliding-window HRV time-series from peak indices.

    Parameters
    ----------
    peak_indices : array-like
        Sample indices of detected PPG peaks in the *fs* domain.
    fs : float
        Sample rate of the original signal (Hz).
    fs_out : float
        Output sample rate of the HRV time-series (Hz).
    window : float
        Width of the sliding analysis window (seconds).
    min_beats : int
        Minimum number of beats required to compute HRV metrics.
    gap_mask : np.ndarray[bool] or None
        Boolean array at *fs* with True wherever the signal is unavailable
        (hardware gaps, artifact rejection).  Any HRV window whose time
        span overlaps at least one True sample has ALL metrics forced to
        NaN, regardless of how many peaks fall within the window.
        Without this guard, a window that straddles a gap boundary can
        accumulate enough peaks from both edges to pass the ``min_beats``
        check and produce physically meaningless metrics (e.g. an
        artificially inflated SDNN caused by the long inter-peak interval
        spanning the gap).

    Returns
    -------
    pd.DataFrame with columns [Timestamp, mean_hr, mean_nn, sdnn, rmssd,
    pnn50, cvsd, cvnn, n_beats].  All metric columns are NaN for windows
    that contain fewer than *min_beats* or that overlap any gap region.
    """
    peak_times = np.asarray(peak_indices) / fs
    step  = 1.0 / fs_out
    # Start from t=0 so the output aligns with the shared 2 Hz / 0.5 s
    # timebase. Rows before the first full window naturally have fewer than
    # min_beats and will carry NaN for all HRV metrics.
    times = np.arange(0.0, peak_times[-1], step)

    has_gap_mask = gap_mask is not None and np.any(gap_mask)
    gap_arr      = np.asarray(gap_mask, dtype=bool) if has_gap_mask else None

    rows = []
    for t in times:
        mask      = (peak_times >= t - window) & (peak_times <= t)
        win_peaks = _filter_physiological_ibi(
            np.asarray(peak_indices)[mask], fs
        )

        row = {
            "Timestamp": t,
            "mean_hr":   np.nan,
            "mean_nn":   np.nan,
            "sdnn":      np.nan,
            "rmssd":     np.nan,
            "pnn50":     np.nan,
            "cvsd":      np.nan,
            "cvnn":      np.nan,
            "n_beats":   int(mask.sum()),
        }

        # Nullify windows that overlap any gap / rejected region.
        # A window spanning a gap boundary would mix beats from before and
        # after the dropout, producing an artificially large IBI and
        # inflated HRV metrics.  Forcing NaN is the correct representation.
        if has_gap_mask:
            lo_s = max(0, int((t - window) * fs))
            hi_s = min(len(gap_arr), int(t * fs) + 1)
            if lo_s < hi_s and gap_arr[lo_s:hi_s].any():
                rows.append(row)   # all metrics already NaN
                continue

        if len(win_peaks) < min_beats:
            rows.append(row)
            continue

        try:
            ibis    = np.diff(win_peaks) / fs * 1000.0   # inter-beat intervals in ms
            mean_nn = float(ibis.mean())
            sdnn    = float(ibis.std(ddof=1))
            diff_ibis = np.diff(ibis)
            rmssd   = float(np.sqrt(np.mean(diff_ibis ** 2)))
            pnn50   = float(np.mean(np.abs(diff_ibis) > 50.0) * 100.0)
            cvsd    = rmssd / mean_nn if mean_nn > 0 else np.nan
            cvnn    = sdnn  / mean_nn if mean_nn > 0 else np.nan
            row["mean_nn"] = mean_nn
            row["sdnn"]    = sdnn
            row["rmssd"]   = rmssd
            row["pnn50"]   = pnn50
            row["cvsd"]    = cvsd
            row["cvnn"]    = cvnn
            row["mean_hr"] = 60000.0 / mean_nn if mean_nn > 0 else np.nan
        except Exception:
            pass

        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Diagnostic figure
# -------------------------------------------------------------------

def _save_ppg_figure(ppg_clean_256, peak_idx_256, t_upsampled, out_dir, synth_mask=None):
    """
    Save ``ppg_overview.png`` to *out_dir*.

    Plots the NeuroKit2-cleaned PPG waveform (256 Hz) in black with red
    vertical lines at real heartbeat peaks and dashed orange lines at
    synthetic fills inserted by ``_fill_missing_beats``. NaN values
    (large hardware gaps) appear as natural breaks in the trace.

    Parameters
    ----------
    ppg_clean_256 : np.ndarray — cleaned PPG at 256 Hz; NaN inside large gaps
    peak_idx_256  : np.ndarray — heartbeat peak indices in the 256 Hz domain
    t_upsampled   : np.ndarray — time axis in seconds for the 256 Hz signal
    out_dir       : str        — directory where the figure is saved
    synth_mask    : np.ndarray[bool] or None
                    Same length as *peak_idx_256*; True at synthetic fills.
                    When None, all peaks are treated as real.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t_upsampled, ppg_clean_256, color="black", linewidth=0.8,
            label="PPG (cleaned)")

    peak_idx_256 = np.asarray(peak_idx_256)
    if synth_mask is None:
        synth_mask = np.zeros(len(peak_idx_256), dtype=bool)
    else:
        synth_mask = np.asarray(synth_mask, dtype=bool)

    real_idx  = peak_idx_256[~synth_mask]
    synth_idx = peak_idx_256[synth_mask]

    if len(peak_idx_256):
        ppg_finite = ppg_clean_256[np.isfinite(ppg_clean_256)]
        ymin = float(ppg_finite.min()) if len(ppg_finite) else 0.0
        ymax = float(ppg_finite.max()) if len(ppg_finite) else 1.0

        if len(real_idx):
            real_times = t_upsampled[real_idx]
            ax.vlines(real_times, ymin=ymin, ymax=ymax,
                      color="red", linewidth=0.6, alpha=0.7,
                      label=f"Heartbeats (n={len(real_times)})")
        if len(synth_idx):
            synth_times = t_upsampled[synth_idx]
            ax.vlines(synth_times, ymin=ymin, ymax=ymax,
                      color="orange", linewidth=0.8, alpha=0.8,
                      linestyles="dashed",
                      label=f"Synthetic fills (n={len(synth_times)})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("PPG — cleaned signal and detected heartbeat peaks")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ppg_overview.png"), dpi=150)
    plt.close(fig)
    printInfo(f"[heartProcessor] Saved PPG figure → {out_dir}/ppg_overview.png")


def _shade_gap_regions(
    ax,
    timestamps: np.ndarray,
    mask: np.ndarray,
    *,
    color: str = "red",
    alpha: float = 0.25,
) -> None:
    """Overlay contiguous True spans of *mask* as translucent bands on *ax*.

    Uses vectorised edge-detection so it is fast even for long recordings.
    """
    if mask is None or not np.any(mask):
        return
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    edges  = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]   # rising edges → span start indices
    ends   = np.where(edges == -1)[0]  # falling edges → span end indices
    for s, e in zip(starts, ends):
        t0 = timestamps[s]
        t1 = timestamps[min(e, len(timestamps) - 1)]
        ax.axvspan(t0, t1, alpha=alpha, color=color, linewidth=0)


def _save_ppg_spectrogram(
    ppg_raw: np.ndarray,
    timestamps: np.ndarray,
    combined_gap_mask: np.ndarray,
    out_dir: str,
    fs: float = SAMPLING_RATE,
) -> None:
    """Save ``ppg_spectrogram.png`` to *out_dir*.

    Generates a short-time Fourier transform (STFT) spectrogram of the PPG
    signal to expose time-frequency structure that is invisible in the raw
    waveform view.

    Reading the figure:
    - Sustained bright energy inside the cardiac band (dashed white lines,
      ``SQI_CARDIAC_LO``–``SQI_CARDIAC_HI`` Hz) indicates a good-quality PPG
      segment.  A clear fundamental plus one or two harmonics is the ideal
      pattern.
    - Broadband brightness (energy spread across all frequencies) or a very
      dim cardiac band indicate noise, motion artefact, or sensor dropout.
    - Translucent red bands mark samples flagged by the two-stage artifact
      rejection so the rejection decisions can be evaluated alongside the
      spectrogram without switching files.

    Parameters
    ----------
    ppg_raw           : np.ndarray  — resampled PPG at *fs* Hz; NaN in rejected spans
    timestamps        : np.ndarray  — absolute time axis in seconds
    combined_gap_mask : np.ndarray  — True where samples are rejected / unavailable
    out_dir           : str         — destination directory
    fs                : float       — sampling rate in Hz (default: SAMPLING_RATE)
    """
    os.makedirs(out_dir, exist_ok=True)

    # STFT needs a continuous signal — forward-fill NaN so the transform does
    # not propagate artefacts across gap boundaries.  The gaps are still shown
    # as shaded bands in the figure.
    ppg_filled = pd.Series(ppg_raw).ffill().bfill().values

    # Window parameters:
    #   8-second analysis window  → frequency resolution ~0.125 Hz,
    #   sufficient to resolve the HRV LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz)
    #   bands and the full cardiac fundamental + harmonics.
    #   87.5 % overlap → ~1-second temporal resolution.
    nperseg  = int(round(8.0 * fs))
    noverlap = int(round(nperseg * 0.875))
    nfft     = int(2 ** np.ceil(np.log2(max(512, nperseg))))

    freqs, t_spec, Sxx = sp_signal.spectrogram(
        ppg_filled, fs=fs,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window="hann", scaling="density",
    )

    # Align spectrogram time axis with the recording's absolute timestamps.
    t_spec = t_spec + timestamps[0]

    # Limit the display to 0–6 Hz (well above the top of the cardiac band).
    f_limit  = 6.0
    f_sel    = freqs <= f_limit
    freqs_v  = freqs[f_sel]
    Sxx_db   = 10.0 * np.log10(np.maximum(Sxx[f_sel, :], 1e-12))

    # Cap the colour range using robust percentiles to avoid a single bright
    # spike washing out the whole figure.
    vmin = float(np.percentile(Sxx_db, 5))
    vmax = float(np.percentile(Sxx_db, 99))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    # ── Top panel: spectrogram ──────────────────────────────────────────
    ax_spec = axes[0]
    mesh = ax_spec.pcolormesh(
        t_spec, freqs_v, Sxx_db,
        shading="gouraud", cmap="viridis",
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(mesh, ax=ax_spec, label="Power (dB re 1 / Hz)")

    ax_spec.axhline(
        SQI_CARDIAC_LO, color="white", linestyle="--", linewidth=0.9,
        label=f"Cardiac band ({SQI_CARDIAC_LO}–{SQI_CARDIAC_HI} Hz)",
    )
    ax_spec.axhline(SQI_CARDIAC_HI, color="white", linestyle="--", linewidth=0.9)
    ax_spec.axhspan(SQI_CARDIAC_LO, SQI_CARDIAC_HI, alpha=0.07, color="white")

    _shade_gap_regions(ax_spec, t_spec,
                       # Downsample mask to spectrogram time resolution
                       np.array([
                           combined_gap_mask[
                               int(np.clip((t - timestamps[0]) * fs, 0, len(combined_gap_mask) - 1))
                           ]
                           for t in t_spec
                       ]))

    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_ylim(0, f_limit)
    ax_spec.set_title(
        "PPG spectrogram — cardiac band: dashed white lines   |"
        "   red: rejected / gap regions"
    )
    ax_spec.legend(loc="upper right", fontsize=8)

    # ── Bottom panel: raw PPG waveform for context ──────────────────────
    ax_raw = axes[1]
    ax_raw.plot(timestamps, ppg_raw, color="black", linewidth=0.5)
    _shade_gap_regions(ax_raw, timestamps, combined_gap_mask)
    ax_raw.set_ylabel("PPG (a.u.)")
    ax_raw.set_xlabel("Time (s)")

    fig.tight_layout()
    path = os.path.join(out_dir, "ppg_spectrogram.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    printInfo(f"[heartProcessor] Saved PPG spectrogram → {path}")


# -------------------------------------------------------------------
# PPG artifact rejection — two-stage (SQI + local MAD)
# -------------------------------------------------------------------

def _ppg_spectral_sqi(
    ppg_raw: np.ndarray,
    fs: float = SAMPLING_RATE,
) -> np.ndarray:
    """Classify each sample as PPG-like or non-PPG via a spectral quality index.

    The signal is divided into overlapping windows of ``SQI_WINDOW_S`` seconds
    (50 % overlap).  Each window is scored by the fraction of its power that
    falls within the cardiac band (``SQI_CARDIAC_LO``–``SQI_CARDIAC_HI`` Hz)
    relative to the total physiological band (0.1–10 Hz).  This ratio is the
    Spectral Quality Index (SQI) for that window.

    Each sample inherits the *best* (highest) SQI across all windows that
    cover it, so a sample is only rejected when *every* overlapping window
    scored it as non-PPG.

    This gate is amplitude-invariant: it rejects broadband noise, motion
    baseline drift, 50/60 Hz interference, flat-line dropout, and any other
    signal that lacks cardiac-frequency periodicity, regardless of amplitude.

    Parameters
    ----------
    ppg_raw : np.ndarray
        PPG signal at *fs* Hz; may contain NaN values (hardware gaps).
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    non_ppg_mask : np.ndarray[bool]
        True at every sample whose best-window SQI fell below
        ``SQI_THRESHOLD``.  Samples that could not be scored (NaN-dominated
        windows, or samples beyond the last full window) are conservatively
        marked True (rejected).
    """
    n      = len(ppg_raw)
    win_n  = int(round(SQI_WINDOW_S * fs))
    step_n = max(1, win_n // 2)   # 50 % overlap

    # Accumulate the best (max) SQI seen across all overlapping windows for
    # each sample.  Start as NaN so unscored samples can be identified.
    best_sqi = np.full(n, np.nan)

    for start in range(0, n - win_n + 1, step_n):
        seg = ppg_raw[start : start + win_n]

        # Skip windows that are mostly NaN (hardware gaps).
        if np.mean(np.isnan(seg)) > 0.5:
            continue

        # Fill the minority of NaN samples so Welch does not fail.
        seg_clean = pd.Series(seg).ffill().bfill().to_numpy()
        if np.any(np.isnan(seg_clean)):
            continue

        freqs, psd = sp_signal.welch(seg_clean, fs=fs, nperseg=min(win_n, len(seg_clean)))

        cardiac_band = (freqs >= SQI_CARDIAC_LO) & (freqs <= SQI_CARDIAC_HI)
        physio_band  = (freqs >= 0.1) & (freqs <= 10.0)

        p_cardiac = float(np.trapezoid(psd[cardiac_band], freqs[cardiac_band]))
        p_physio  = float(np.trapezoid(psd[physio_band],  freqs[physio_band]))

        sqi_val = p_cardiac / p_physio if p_physio > 0.0 else 0.0

        best_sqi[start : start + win_n] = np.fmax(
            best_sqi[start : start + win_n], sqi_val
        )

    # Samples never covered by a scorable window get SQI = 0 (conservative reject).
    best_sqi = np.where(np.isnan(best_sqi), 0.0, best_sqi)

    non_ppg_mask = best_sqi < SQI_THRESHOLD
    n_non_ppg    = int(non_ppg_mask.sum())
    printInfo(
        f"[heartProcessor] PPG Stage-1 SQI: {n_non_ppg} / {n} samples "
        f"({100.0 * n_non_ppg / n:.1f} %) classified as non-PPG "
        f"(SQI < {SQI_THRESHOLD})"
    )
    return non_ppg_mask


def _save_ppg_rejected_figure(
    ppg_raw: np.ndarray,
    sqi_mask: np.ndarray,
    mad_mask: np.ndarray,
    timestamps: np.ndarray,
    out_dir: str,
) -> None:
    """Save a three-colour rejection map to ``ppg_rejected.png``.

    Colour coding:
        black  — accepted sample
        orange — rejected by Stage-1 SQI (non-PPG window)
        red    — rejected by Stage-2 local MAD (glitch within valid PPG)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build a single categorical array: 0=valid, 1=SQI-rejected, 2=MAD-rejected.
    # MAD takes precedence in the rare case both flags are set.
    category = np.zeros(len(ppg_raw), dtype=np.int8)
    category[sqi_mask] = 1
    category[mad_mask] = 2

    color_map   = {0: "black",  1: "darkorange", 2: "red"}
    label_map   = {
        0: "Accepted",
        1: f"Non-PPG — SQI < {SQI_THRESHOLD}",
        2: f"Glitch — MAD-z > {Z_THRESHOLD} (±{REJECT_MARGIN_S:.0f} s)",
    }
    first_seen  = {0: True, 1: True, 2: True}

    fig, ax = plt.subplots(figsize=(14, 4))
    n = len(ppg_raw)
    i = 0
    while i < n:
        cat = int(category[i])
        j   = i + 1
        while j < n and int(category[j]) == cat:
            j += 1

        label = label_map[cat] if first_seen[cat] else None
        first_seen[cat] = False

        ax.plot(
            timestamps[i:j],
            ppg_raw[i:j],
            color=color_map[cat],
            linewidth=0.8,
            label=label,
        )
        i = j

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PPG (raw a.u.)")
    ax.set_title(
        f"PPG — rejection map  "
        f"(Stage 1: SQI < {SQI_THRESHOLD} | "
        f"Stage 2: MAD-z > {Z_THRESHOLD}, ±{REJECT_MARGIN_S:.0f} s, "
        f"cap {REJECT_MAX_PCT:.0f} %)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(out_dir, "ppg_rejected.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    printInfo(f"[heartProcessor] Saved rejection map → {out_path}")


def apply_ppg_artifact_rejection(
    ppg_raw: np.ndarray,
    timestamps: np.ndarray,
    out_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply two-stage artifact rejection to *ppg_raw*.

    Stage 1 — Spectral Quality Index
        Each sliding window is scored by the fraction of its power in the
        cardiac band (``SQI_CARDIAC_LO``–``SQI_CARDIAC_HI`` Hz).  Windows
        below ``SQI_THRESHOLD`` are classified as non-PPG and blanked to NaN.
        This gate is amplitude-invariant: stable noise is rejected because it
        carries no cardiac-frequency power, not because of its level.

    Stage 2 — Local MAD glitch detector
        Within windows that passed Stage 1, a rolling MAD-based z-score flags
        sudden spikes or clipping transients.  A local (rolling) window of
        ``REJECT_WINDOW_S`` seconds is used so that valid PPG at a different
        amplitude regime (e.g. after sensor repositioning) is compared to its
        own neighbourhood rather than a global statistic.

    The two stages are strictly sequential: Stage 2 operates only on samples
    not already blanked by Stage 1, so its safety cap (``REJECT_MAX_PCT``)
    reflects the true proportion of within-PPG glitches.

    Parameters
    ----------
    ppg_raw : np.ndarray
        PPG signal at ``SAMPLING_RATE`` Hz; may contain NaN values from
        gap-aware resampling.
    timestamps : np.ndarray
        Time axis in seconds (same length as *ppg_raw*).
    out_dir : str
        Directory where ``ppg_rejected.png`` is saved.

    Returns
    -------
    ppg_fixed : np.ndarray
        *ppg_raw* with all rejected samples set to NaN.
    reject_mask : np.ndarray[bool]
        True at every sample rejected by either stage.
    """
    ppg_fixed = ppg_raw.copy()

    # ------------------------------------------------------------------
    # Stage 1 — Spectral Quality Index
    # ------------------------------------------------------------------
    sqi_mask            = _ppg_spectral_sqi(ppg_fixed)
    ppg_fixed[sqi_mask] = np.nan

    # ------------------------------------------------------------------
    # Stage 2 — Local MAD glitch detector (on SQI-surviving samples only)
    # ------------------------------------------------------------------
    ppg_sq = ppg_fixed ** 2   # NaN where SQI-rejected; ignored by nanmedian

    win        = max(3, int(round(REJECT_WINDOW_S * SAMPLING_RATE)) | 1)  # odd
    s          = pd.Series(ppg_sq)
    roll_med   = s.rolling(win, center=True, min_periods=3).median()
    roll_mad   = (s - roll_med).abs().rolling(win, center=True, min_periods=3).median()
    sq_mad_std = (roll_mad * 1.4826).replace(0.0, np.nan)

    if sq_mad_std.isna().all():
        printWarning(
            "[heartProcessor] PPG Stage-2: rolling MAD is zero everywhere — "
            "skipping MAD glitch detection."
        )
        reject_mask = sqi_mask.copy()
        _save_ppg_rejected_figure(ppg_raw, sqi_mask, np.zeros(len(ppg_raw), dtype=bool),
                                  timestamps, out_dir)
        return ppg_fixed, reject_mask

    z_scores     = ((s - roll_med) / sq_mad_std).abs()
    outlier_mask = (z_scores > Z_THRESHOLD).fillna(False).to_numpy()

    n_outliers = int(outlier_mask.sum())
    printInfo(f"[heartProcessor] PPG Stage-2 MAD: {n_outliers} outlier samples "
              f"(local MAD-z > {Z_THRESHOLD})")

    n_margin    = int(round(REJECT_MARGIN_S * SAMPLING_RATE))
    struct      = np.ones(2 * n_margin + 1, dtype=bool)
    mad_mask    = binary_dilation(outlier_mask, structure=struct)

    # Do not double-count samples already blanked by Stage 1.
    mad_mask_new = mad_mask & ~sqi_mask

    n_rejected = int(mad_mask_new.sum())
    pct        = 100.0 * n_rejected / len(ppg_raw)
    printInfo(
        f"[heartProcessor] PPG Stage-2 MAD: {n_rejected} / {len(ppg_raw)} "
        f"new samples rejected after margin expansion ({pct:.1f} %)"
    )

    # Safety cap: if Stage-2 rejection would consume more than REJECT_MAX_PCT
    # of the signal, the detector has likely found no real glitch.
    if pct > REJECT_MAX_PCT:
        printWarning(
            f"[heartProcessor] Stage-2 MAD would blank {pct:.1f} % of the "
            f"signal (cap = {REJECT_MAX_PCT:.0f} %) — skipping MAD rejection."
        )
        mad_mask_new = np.zeros(len(ppg_raw), dtype=bool)

    ppg_fixed[mad_mask_new] = np.nan
    reject_mask             = sqi_mask | mad_mask_new

    _save_ppg_rejected_figure(ppg_raw, sqi_mask, mad_mask_new, timestamps, out_dir)

    total_rejected = int(reject_mask.sum())
    printInfo(
        f"[heartProcessor] PPG artifact rejection complete: "
        f"{total_rejected} / {len(ppg_raw)} samples blanked total "
        f"({100.0 * total_rejected / len(ppg_raw):.1f} %)"
    )
    return ppg_fixed, reject_mask


# -------------------------------------------------------------------
# Main PPG processing
# -------------------------------------------------------------------

def process_ppg_to_dataframe(ppg_raw, fs=SAMPLING_RATE):
    """
    Process raw PPG to a sliding-window HRV time-series DataFrame.

    Upsamples the raw signal to _UPSAMPLE_RATE before peak detection because
    the Elgendi algorithm is unreliable at the native 51.2 Hz (only ~44 samples
    per beat), which causes missed peaks and catastrophically inflated RMSSD.

    After peak detection, IBIs outside the physiological range are discarded
    before HRV computation as a secondary guard.
    """
    _, _, peak_idx_orig, _, _ = _detect_ppg_peaks(ppg_raw, fs)
    peak_idx_orig, _, _ = _fill_missing_beats(peak_idx_orig, fs=fs)
    return _sliding_hrv_dataframe(peak_idx_orig, fs=fs)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def computeHeartDynamics(recPath, show=False):
    """
    Compute heart rate dynamics from Shimmer PPG data.

    Pipeline:
      1. Load raw signal with original hardware timestamps.
      2. Resample to a strict 51.2 Hz grid (gap-aware); save resampled
         data and a statistics report to ``features/ppg/``.
      3. Two-stage artifact rejection:
           Stage 1 — Spectral Quality Index (SQI): sliding windows scored
             by cardiac-band power fraction; non-PPG windows (noise, motion,
             flat lines) blanked to NaN regardless of amplitude.
           Stage 2 — Local MAD glitch detector: rolling MAD-based z-score
             within SQI-passing segments; catches sudden spikes / clipping.
         Rejected samples are set to NaN and ``ppg_resampled.csv`` is
         overwritten with the cleaned signal.  A three-colour rejection map
         is saved to ``features/ppg/ppg_rejected.png``.
      4. Peak detection via NeuroKit2 at 256 Hz; peaks in gap/rejected
         regions are discarded.
      5. Save diagnostic figures to ``features/ppg/``:
           ppg_overview.png   — cleaned waveform with annotated peaks
           ppg_spectrogram.png — STFT spectrogram (0–6 Hz) with cardiac-band
             reference lines and rejection regions overlaid; bright sustained
             energy in the cardiac band indicates good-quality PPG.
      6. Sliding-window HRV computation; any 30 s window that overlaps a
         gap or rejected region has ALL metrics forced to NaN so that the
         boundary artefact (spuriously large IBI spanning the gap) does
         not produce physically meaningless metrics.
      7. Save results to ``results/HeartDynamics.csv``.

    Writes:
      - ``features/ppg/ppg_resampled.csv``   (cleaned; NaN in rejected regions)
      - ``features/ppg/ppg_resample_report.txt``
      - ``features/ppg/ppg_rejected.png``    (three-colour rejection map)
      - ``features/ppg/ppg_spectrogram.png`` (STFT, 0–6 Hz, with cardiac-band
        reference lines and rejection region overlay)
      - ``features/ppg/ppg_normalized.csv``  (rolling z-score at SAMPLING_RATE;
        same Timestamp/Value layout as ppg_resampled.csv; NaN in rejected /
        gap regions; this is the signal that NeuroKit2 peak detection uses)
      - ``features/ppg/ppg_overview.png``
      - ``results/HeartDynamics.csv``  columns: [Timestamp (s), mean_hr,
        mean_nn, sdnn, rmssd, pnn50, cvsd, cvnn, n_beats] at 2 Hz / 0.5 s

    Args:
        recPath: Path to the recording directory.
        show:    Unused; kept for API compatibility.
    """
    printInfo("[heartProcessor] Computing Heart Dynamics")

    bpm_hrv_path = os.path.join(recPath, "results", "HeartDynamics.csv")
    if os.path.isfile(bpm_hrv_path):
        printInfo("[heartProcessor] Heart dynamics already computed, using cached results")
        return

    timestamps, ppg_raw = _load_shimmer_ppg_signal(recPath)
    if ppg_raw is None:
        return

    ppg_features_dir = os.path.join(recPath, "features", "ppg")

    # Step 1 — resample onto a regular 51.2 Hz grid using original timestamps
    timestamps, ppg_raw, gap_mask = resample_to_grid(
        timestamps, ppg_raw, "ppg", ppg_features_dir
    )

    if len(ppg_raw) < 250:
        printWarning("[heartProcessor] Shimmer recording too short to compute PPG")
        return

    # Step 2 — two-stage artifact rejection (SQI + local MAD).
    # Rejected samples are set to NaN and the fixed signal overwrites the
    # resampled CSV so every downstream artifact reflects the cleaned data.
    ppg_raw, reject_mask = apply_ppg_artifact_rejection(
        ppg_raw, timestamps, ppg_features_dir
    )
    ppg_df = pd.read_csv(os.path.join(ppg_features_dir, "ppg_resampled.csv"))
    ppg_df["Value"] = ppg_raw
    ppg_df.to_csv(os.path.join(ppg_features_dir, "ppg_resampled.csv"), index=False)

    # The combined gap mask covers both hardware gaps and rejected artifact
    # regions so that peaks are discarded from all silent/bad spans.
    combined_gap_mask = gap_mask | reject_mask

    # Save the rolling-normalised signal that will be fed to NeuroKit2.
    # This is useful for visual inspection and debugging: unlike the raw or
    # resampled signal, it equalises amplitude across regimes and shows the
    # waveform shape that the peak detector actually operates on.
    # NaN is restored at all rejected / gap positions to match the convention
    # used by ppg_resampled.csv.
    ppg_filled_for_norm = pd.Series(ppg_raw).ffill().bfill().values
    ppg_norm             = _normalize_ppg_for_detection(ppg_filled_for_norm, SAMPLING_RATE)
    ppg_norm[combined_gap_mask] = np.nan
    norm_path = os.path.join(ppg_features_dir, "ppg_normalized.csv")
    pd.DataFrame({"Timestamp": timestamps, "Value": ppg_norm}).to_csv(norm_path, index=False)
    printInfo(f"[heartProcessor] Saved normalised PPG (z-score, {NORMALIZE_WINDOW_S:.0f} s window) → {norm_path}")

    try:
        ppg_clean_256, peak_idx_256, peak_idx_orig, t_upsampled, gap_mask_256 = (
            _detect_ppg_peaks(ppg_raw, gap_mask=combined_gap_mask)
        )
    except Exception as e:
        printError(f"[heartProcessor] PPG peak detection failed: {e}")
        return

    peak_idx_orig, _, n_inserted_orig = _fill_missing_beats(
        peak_idx_orig, fs=SAMPLING_RATE, gap_mask=combined_gap_mask,
    )
    peak_idx_256, synth_mask_256, _ = _fill_missing_beats(
        peak_idx_256, fs=_UPSAMPLE_RATE, gap_mask=gap_mask_256,
    )
    printInfo(
        f"[heartProcessor] Inserted {n_inserted_orig} synthetic peaks to fill "
        f"missed beats (adaptive threshold = {MISSING_BEAT_RATIO}x recent "
        f"median IBI, max gap = {MISSING_BEAT_MAX_FILL_S:.0f} s)"
    )

    # Step 3 — diagnostic figures
    _save_ppg_figure(
        ppg_clean_256, peak_idx_256, t_upsampled, ppg_features_dir,
        synth_mask=synth_mask_256,
    )
    _save_ppg_spectrogram(ppg_raw, timestamps, combined_gap_mask, ppg_features_dir)

    try:
        hrv_df = _sliding_hrv_dataframe(peak_idx_orig, gap_mask=combined_gap_mask)
    except Exception as e:
        printError(f"[heartProcessor] HRV computation failed: {e}")
        return

    if hrv_df.empty:
        printWarning("[heartProcessor] No valid HRV windows computed")
        return

    os.makedirs(os.path.join(recPath, "results"), exist_ok=True)
    hrv_df.to_csv(bpm_hrv_path, index=False)
    printInfo(f"[heartProcessor] Heart dynamics computation completed → {bpm_hrv_path}")
