"""
Heart rate dynamics from Shimmer PPG.

Before peak detection, the raw signal is resampled onto a strict 51.2 Hz
grid using the original hardware timestamps (see shimmerResampler). The
grid signal is then upsampled to 256 Hz for NeuroKit2 (Elgendi algorithm)
peak detection. Peaks detected inside large hardware-gap regions are
discarded. A diagnostic figure is saved to ``features/ppg/ppg_overview.png``.

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

# Artifact-rejection constants (squared z-score method)
Z_THRESHOLD     = 2.5    # |z| beyond which a sample is flagged as an outlier
REJECT_MARGIN_S = 5.0    # seconds to reject around each outlier centre

# Elgendi peak detection requires ≥100 Hz for reliable operation.
# Upsample to this rate before processing so the algorithm has adequate resolution.
_UPSAMPLE_RATE = 256.0

# Physiological IBI bounds (ms). IBIs outside this window are missed/spurious
# peaks and must be excluded before computing RMSSD.
_IBI_MIN_MS = 300.0    # ~200 bpm
_IBI_MAX_MS = 2000.0   # ~30 bpm

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
        f"[heartProcessor] No Shimmer PPG file in {raw_dir}. "
        "Shimmer was likely not used for this recording."
    )
    return None, None


# -------------------------------------------------------------------
# Physiological IBI filtering
# -------------------------------------------------------------------

def _filter_physiological_ibi(peak_indices, fs):
    """
    Sanitize peaks so all remaining IBIs are physiologically plausible.

    Two artifact types are handled differently:
    - Too short (IBI < _IBI_MIN_MS): spurious detection — discard the
      second peak of the pair.
    - Too long  (IBI > _IBI_MAX_MS): missed beat — insert a synthetic
      peak at the midpoint so RMSSD is not dominated by a single large gap.
    """
    peaks = list(np.asarray(peak_indices, dtype=float))
    i = 0
    while i < len(peaks) - 1:
        ibi_ms = (peaks[i + 1] - peaks[i]) / fs * 1000.0
        if ibi_ms < _IBI_MIN_MS:
            del peaks[i + 1]
        elif ibi_ms > _IBI_MAX_MS:
            synthetic = (peaks[i] + peaks[i + 1]) / 2.0
            peaks.insert(i + 1, synthetic)
            i += 1  # advance past the now-valid first half-gap
        else:
            i += 1
    return np.round(peaks).astype(int)


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

    interpolator  = interp1d(t_original, ppg_filled, kind="cubic",
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

    Returns
    -------
    pd.DataFrame with columns [Timestamp, mean_hr, mean_nn, sdnn, rmssd,
    pnn50, cvsd, cvnn, n_beats].
    """
    peak_times = np.asarray(peak_indices) / fs
    step  = 1.0 / fs_out
    # Start from t=0 so the output aligns with the shared 2 Hz / 0.5 s
    # timebase. Rows before the first full window naturally have fewer than
    # min_beats and will carry NaN for all HRV metrics.
    times = np.arange(0.0, peak_times[-1], step)

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

def _save_ppg_figure(ppg_clean_256, peak_idx_256, t_upsampled, out_dir):
    """
    Save ``ppg_overview.png`` to *out_dir*.

    Plots the NeuroKit2-cleaned PPG waveform (256 Hz) in black with a
    red vertical line at every detected heartbeat peak. NaN values
    (large hardware gaps) appear as natural breaks in the trace.

    Parameters
    ----------
    ppg_clean_256 : np.ndarray — cleaned PPG at 256 Hz; NaN inside large gaps
    peak_idx_256  : np.ndarray — heartbeat peak indices in the 256 Hz domain
    t_upsampled   : np.ndarray — time axis in seconds for the 256 Hz signal
    out_dir       : str        — directory where the figure is saved
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t_upsampled, ppg_clean_256, color="black", linewidth=0.8,
            label="PPG (cleaned)")

    peak_times = t_upsampled[peak_idx_256]
    if len(peak_times):
        ppg_finite = ppg_clean_256[np.isfinite(ppg_clean_256)]
        ymin = float(ppg_finite.min()) if len(ppg_finite) else 0.0
        ymax = float(ppg_finite.max()) if len(ppg_finite) else 1.0
        ax.vlines(peak_times, ymin=ymin, ymax=ymax,
                  color="red", linewidth=0.6, alpha=0.7,
                  label=f"Heartbeats (n={len(peak_times)})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.set_title("PPG — cleaned signal and detected heartbeat peaks")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ppg_overview.png"), dpi=150)
    plt.close(fig)
    printInfo(f"[heartProcessor] Saved PPG figure → {out_dir}/ppg_overview.png")


# -------------------------------------------------------------------
# PPG artifact rejection (squared z-score method)
# -------------------------------------------------------------------

def _save_ppg_rejected_figure(
    ppg_raw: np.ndarray,
    reject_mask: np.ndarray,
    timestamps: np.ndarray,
    out_dir: str,
) -> None:
    """Save ``ppg_rejected.png`` to *out_dir*.

    Segments where *reject_mask* is True are drawn in red; the rest in black.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4))

    n           = len(ppg_raw)
    first_red   = True
    first_black = True
    i           = 0

    while i < n:
        is_rejected = bool(reject_mask[i])
        j = i + 1
        while j < n and bool(reject_mask[j]) == is_rejected:
            j += 1

        color      = "red" if is_rejected else "black"
        label: str | None = None
        if is_rejected and first_red:
            label     = f"Rejected (outlier ± {REJECT_MARGIN_S:.0f} s)"
            first_red = False
        elif not is_rejected and first_black:
            label       = "Accepted"
            first_black = False

        ax.plot(
            timestamps[i:j],
            ppg_raw[i:j],
            color=color,
            linewidth=0.8,
            label=label,
        )
        i = j

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PPG (raw a.u.)")
    ax.set_title(
        f"PPG — rejection map  (squared z-score > {Z_THRESHOLD} σ, "
        f"±{REJECT_MARGIN_S:.0f} s margin)"
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
    """Apply squared z-score artifact rejection to *ppg_raw*.

    Parameters
    ----------
    ppg_raw : np.ndarray
        PPG signal at ``SAMPLING_RATE`` Hz; may already contain NaN values
        from gap-aware resampling.
    timestamps : np.ndarray
        Time axis in seconds (same length as *ppg_raw*).
    out_dir : str
        Directory where ``ppg_rejected.png`` is saved.

    Returns
    -------
    ppg_fixed : np.ndarray
        *ppg_raw* with rejected samples set to NaN.
    reject_mask : np.ndarray[bool]
        True at every sample that was rejected.
    """
    ppg_sq  = ppg_raw ** 2
    sq_mean = np.nanmean(ppg_sq)
    sq_std  = np.nanstd(ppg_sq)

    if sq_std == 0:
        printWarning(
            "[heartProcessor] PPG signal has zero variance after squaring — "
            "skipping artifact rejection."
        )
        return ppg_raw.copy(), np.zeros(len(ppg_raw), dtype=bool)

    z_scores     = (ppg_sq - sq_mean) / sq_std
    outlier_mask = np.abs(z_scores) > Z_THRESHOLD

    n_outliers = int(outlier_mask.sum())
    printInfo(f"[heartProcessor] PPG outlier samples (|z| > {Z_THRESHOLD} σ): {n_outliers}")

    n_margin    = int(round(REJECT_MARGIN_S * SAMPLING_RATE))
    struct      = np.ones(2 * n_margin + 1, dtype=bool)
    reject_mask = binary_dilation(outlier_mask, structure=struct)

    n_rejected = int(reject_mask.sum())
    pct        = 100.0 * n_rejected / len(ppg_raw)
    printInfo(
        f"[heartProcessor] PPG samples rejected after margin expansion: "
        f"{n_rejected} / {len(ppg_raw)}  ({pct:.1f} %)"
    )

    _save_ppg_rejected_figure(ppg_raw, reject_mask, timestamps, out_dir)

    ppg_fixed              = ppg_raw.copy()
    ppg_fixed[reject_mask] = np.nan
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
      3. Artifact rejection via squared z-score outlier detection; rejected
         samples are nullified and ``ppg_resampled.csv`` is overwritten with
         the cleaned signal; saves ``features/ppg/ppg_rejected.png``.
      4. Peak detection via NeuroKit2 at 256 Hz; peaks in gap/rejected
         regions are discarded.
      5. Save diagnostic figure to ``features/ppg/ppg_overview.png``.
      6. Sliding-window HRV computation.
      7. Save results to ``results/HeartDynamics.csv``.

    Writes:
      - ``features/ppg/ppg_resampled.csv``   (cleaned; NaN in rejected regions)
      - ``features/ppg/ppg_resample_report.txt``
      - ``features/ppg/ppg_rejected.png``
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

    # Step 2 — artifact rejection via squared z-score outlier detection.
    # Rejected samples are set to NaN and the fixed signal overwrites the
    # resampled CSV so every downstream artefact reflects the cleaned data.
    ppg_raw, reject_mask = apply_ppg_artifact_rejection(
        ppg_raw, timestamps, ppg_features_dir
    )
    ppg_df = pd.read_csv(os.path.join(ppg_features_dir, "ppg_resampled.csv"))
    ppg_df["Value"] = ppg_raw
    ppg_df.to_csv(os.path.join(ppg_features_dir, "ppg_resampled.csv"), index=False)

    # The combined gap mask covers both hardware gaps and rejected artifact
    # regions so that peaks are discarded from all silent/bad spans.
    combined_gap_mask = gap_mask | reject_mask

    try:
        ppg_clean_256, peak_idx_256, peak_idx_orig, t_upsampled, _ = (
            _detect_ppg_peaks(ppg_raw, gap_mask=combined_gap_mask)
        )
    except Exception as e:
        printError(f"[heartProcessor] PPG peak detection failed: {e}")
        return

    # Step 3 — diagnostic figure
    _save_ppg_figure(ppg_clean_256, peak_idx_256, t_upsampled, ppg_features_dir)

    try:
        hrv_df = _sliding_hrv_dataframe(peak_idx_orig)
    except Exception as e:
        printError(f"[heartProcessor] HRV computation failed: {e}")
        return

    if hrv_df.empty:
        printWarning("[heartProcessor] No valid HRV windows computed")
        return

    os.makedirs(os.path.join(recPath, "results"), exist_ok=True)
    hrv_df.to_csv(bpm_hrv_path, index=False)
    printInfo(f"[heartProcessor] Heart dynamics computation completed → {bpm_hrv_path}")
