"""
EDA / GSR arousal processing for Shimmer data.

Decomposes raw EDA into tonic (SCL) and phasic (SCR) components using
cvxEDA (Greco et al., IEEE TBME 2015). Before decomposition, the raw
signal is resampled onto a strict 51.2 Hz grid using the original
hardware timestamps (see shimmerResampler). A MAD-based artifact
rejection step follows. Discrete SCR events are detected from the 4 Hz
phasic component and saved to a diagnostic figure alongside the filtered
EDA and tonic traces.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks

from nucleuskit_pipeline.logging_utils import printInfo, printWarning, printError
from nucleuskit_pipeline.shimmer.vendors.cvxEDA import cvxEDA
from nucleuskit_pipeline.shimmer.processor.resampler import resample_to_grid, normalise_timestamps_to_seconds

# -------------------------------------------------------------------
# Signal constants
# -------------------------------------------------------------------
SAMPLING_RATE    = 51.2   # native Shimmer EDA sample rate (Hz)
CVXEDA_RATE      = 4      # EDA is a slow signal; cvxEDA runs on downsampled data
PHASIC_CUTOFF    = 0.5    # Hz lowpass applied to phasic after decomposition
ARTIFACT_WINDOW  = 5.0    # seconds for rolling MAD window in artifact rejection
ARTIFACT_MAD_THR = 5.0    # MADs from local median to flag as artifact

# SCR peak detection (on the 4 Hz phasic signal)
_SCR_MIN_HEIGHT     = 0.05   # minimum phasic amplitude (z-score units)
_SCR_MIN_PROMINENCE = 0.05   # minimum peak prominence (z-score units)
_SCR_MIN_DISTANCE_S = 1.0    # minimum seconds between consecutive SCR events

# Shimmer CSV: column 0 = timestamp (ms), column 4 = EDA / GSR
_SHIMMER_TIMESTAMP_COL = 0

# Candidate files and the column index that holds the EDA/GSR value.
# Checked in order; the first file that exists and has enough columns wins.
#   shimmer.csv / rawShimmer_0.csv  — full multi-channel Shimmer export (col 4)
#   gsr.tmp / gsr.csv               — two-column legacy dump (col 1): timestamps, EDA
_EDA_CANDIDATES = [
    ("shimmer.csv",      4),
    ("rawShimmer_0.csv", 4),
    ("gsr.tmp",          1),
    ("gsr.csv",          1),
]


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def _load_shimmer_eda_signal(rec_path):
    """
    Load timestamp + EDA from ``rawData/shimmer.csv``, legacy
    ``rawShimmer_0.csv``, or minimal ``gsr.tmp``.

    ``shimmer.csv`` / ``rawShimmer_0.csv`` are full multi-channel Shimmer
    exports (no header); column 4 contains raw GSR resistance in kΩ.
    ``gsr.tmp`` / ``gsr.csv`` are headerless two-column files (timestamps,
    EDA) where column 1 holds the EDA/GSR value directly.

    Conversion to conductance (µS) is performed later, after resampling
    onto the regular grid.

    Returns a two-column DataFrame (0 = time in seconds from start, 1 = EDA
    in kΩ), or None if no usable file is present.
    """
    raw_dir = os.path.join(rec_path, "rawData")
    for name, eda_col in _EDA_CANDIDATES:
        path = os.path.join(raw_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, header=None)
        except Exception:
            continue
        if df.shape[1] <= eda_col:
            printWarning(
                f"[edaProcessor] {path} has fewer than {eda_col + 1} columns; "
                f"cannot read EDA (column {eda_col})."
            )
            continue
        sig = df[[_SHIMMER_TIMESTAMP_COL, eda_col]].copy()
        sig.columns = [0, 1]
        sig[0] = normalise_timestamps_to_seconds(sig[0].values)
        return sig

    printWarning(
        f"[edaProcessor] No Shimmer EDA file in {raw_dir} "
        "(expected shimmer.csv, rawShimmer_0.csv, gsr.tmp, or gsr.csv). "
        "Shimmer was likely not used for this recording."
    )
    return None


# -------------------------------------------------------------------
# Artifact rejection
# -------------------------------------------------------------------

def reject_eda_artifacts(eda_data, gap_mask=None):
    """
    Detect and linearly interpolate over EDA artifacts using a MAD-based
    local outlier criterion on the raw signal.

    Samples deviating more than ARTIFACT_MAD_THR * local_MAD from the
    rolling median are flagged, expanded by a 1-second buffer, then
    replaced via linear interpolation.

    Parameters
    ----------
    eda_data : array-like
        EDA signal at SAMPLING_RATE (51.2 Hz). May contain NaN in large-gap
        regions if the signal has been resampled with gap preservation.
    gap_mask : np.ndarray[bool] or None
        If provided, indices flagged as True correspond to true hardware
        disconnects. After interpolation those positions are reset to NaN
        so the large gaps are never filled in.
    """
    eda = pd.Series(np.array(eda_data, dtype=float))
    win = int(ARTIFACT_WINDOW * SAMPLING_RATE)

    rolling_med = eda.rolling(window=win, center=True, min_periods=1).median()
    rolling_mad = (
        (eda - rolling_med)
        .abs()
        .rolling(window=win, center=True, min_periods=1)
        .median()
    )
    # Floor at 0.05 µS — prevents flat segments (MAD=0) from flagging all
    # subsequent samples as outliers.
    rolling_mad = rolling_mad.clip(lower=0.05)

    outliers = (eda - rolling_med).abs() > ARTIFACT_MAD_THR * rolling_mad
    outliers = (
        outliers
        .rolling(window=int(SAMPLING_RATE), center=True, min_periods=1)
        .max()
        .astype(bool)
    )

    n_flagged = int(outliers.sum())
    if n_flagged > 0:
        printInfo(
            f"[edaProcessor] reject_eda_artifacts: flagged {n_flagged} samples "
            f"({n_flagged / SAMPLING_RATE:.1f} s) for interpolation"
        )

    eda_clean = eda.copy()
    eda_clean[outliers] = np.nan
    eda_clean = eda_clean.interpolate(method="linear", limit_direction="both")

    # Re-apply large-gap NaN so the interpolation above does not fill
    # true hardware disconnects.
    if gap_mask is not None and np.any(gap_mask):
        eda_clean[gap_mask] = np.nan

    return eda_clean.values


# -------------------------------------------------------------------
# EDA decomposition
# -------------------------------------------------------------------

def _decompose_eda(eda_clean, gap_mask=None):
    """
    Downsample, z-score normalise, run cvxEDA, and lowpass-filter the
    phasic component.

    Parameters
    ----------
    eda_clean : np.ndarray
        Artifact-rejected EDA signal at SAMPLING_RATE (51.2 Hz).
        May contain NaN in large-gap regions.
    gap_mask : np.ndarray[bool] or None
        True at every sample that belongs to a large hardware gap.
        After cvxEDA and resampling back to original length, those
        positions are set to NaN in both outputs.

    Returns
    -------
    tonic_down  : np.ndarray — tonic at CVXEDA_RATE (4 Hz)
    phasic_down : np.ndarray — lowpass-filtered phasic at CVXEDA_RATE (4 Hz)
    gap_mask_down : np.ndarray[bool] — gap_mask downsampled to CVXEDA_RATE
    """
    n_original = len(eda_clean)

    # Forward-fill NaN so cvxEDA receives a continuous array.
    # The gap_mask is used later to restore NaN in the outputs.
    eda_for_solver = pd.Series(eda_clean).ffill().bfill().values

    n_down   = int(round(n_original * CVXEDA_RATE / SAMPLING_RATE))
    eda_down = signal.resample(eda_for_solver, n_down)

    # Z-score on the gap-filled signal (gaps would have distorted the mean)
    eda_norm = (eda_down - np.mean(eda_down)) / (np.std(eda_down) + 1e-12)

    phasic_down, _, tonic_down, _, _, _, _ = cvxEDA(eda_norm, 1.0 / CVXEDA_RATE)

    nyq = CVXEDA_RATE / 2.0
    b, a = signal.butter(3, PHASIC_CUTOFF / nyq, btype="low")
    phasic_down = signal.filtfilt(b, a, phasic_down)

    # Downsample the gap mask to match the 4 Hz output
    if gap_mask is not None and np.any(gap_mask):
        gap_mask_down = signal.resample(gap_mask.astype(float), n_down) > 0.5
    else:
        gap_mask_down = np.zeros(n_down, dtype=bool)

    return tonic_down, phasic_down, gap_mask_down


def compute_eda_components(eda_data):
    """
    Decompose raw EDA into tonic (SCL) and phasic (SCR) components via cvxEDA.

    Pipeline:
      1. Artifact rejection on the raw 51.2 Hz signal (MAD-based interpolation)
      2. Downsample to 4 Hz (standard for EDA decomposition)
      3. Z-score normalisation (safe after artifact rejection in step 1)
      4. cvxEDA decomposition
      5. Lowpass filter on phasic to attenuate solver-induced noise
      6. Resample both components back to the original length

    Parameters
    ----------
    eda_data : array-like
        Raw EDA signal in µS sampled at SAMPLING_RATE (51.2 Hz).

    Returns
    -------
    pd.DataFrame with columns ['TonicEDA', 'PhasicEDA'].
    Values are in z-score units of the downsampled, artifact-cleaned signal.
    """
    n_original = len(eda_data)
    eda_clean  = reject_eda_artifacts(eda_data)
    tonic_down, phasic_down, _ = _decompose_eda(eda_clean)

    tonic  = signal.resample(tonic_down,  n_original)
    phasic = signal.resample(phasic_down, n_original)

    return pd.DataFrame({"TonicEDA": tonic, "PhasicEDA": phasic})


# -------------------------------------------------------------------
# 2 Hz resampling
# -------------------------------------------------------------------

def _downsample_to_2hz(timestamps, *arrays):
    """
    Downsample one or more arrays from the 51.2 Hz grid to a 2 Hz / 0.5 s
    timebase, matching the shared output convention for all pipeline metrics.

    Uses pandas TimedeltaIndex + resample('500ms').mean() so that:
    - Each 0.5 s bin is the mean of the ~26 original samples it contains.
    - NaN samples (large gaps) produce NaN bins automatically.
    - The resulting timestamps are exact multiples of 0.5 s from t=0.

    Parameters
    ----------
    timestamps : np.ndarray — regular 51.2 Hz time axis in seconds.
    *arrays    : np.ndarray — one or more value arrays of the same length.

    Returns
    -------
    (t_2hz, arr1_2hz, arr2_2hz, ...) — tuple of length len(arrays)+1.
    """
    idx = pd.to_timedelta(timestamps, unit="s")
    out_arrays = []
    ref_index  = None
    for arr in arrays:
        resampled = pd.Series(arr, index=idx).resample("500ms").mean()
        if ref_index is None:
            ref_index = resampled.index
        out_arrays.append(resampled.values)

    t_2hz = ref_index.total_seconds().values
    return (t_2hz,) + tuple(out_arrays)


# -------------------------------------------------------------------
# SCR event detection
# -------------------------------------------------------------------

def _detect_scr_events(phasic_down, gap_mask_down=None, fs=CVXEDA_RATE):
    """
    Detect discrete SCR events as peaks in the 4 Hz phasic component.

    Peak detection is performed at the native cvxEDA output rate (4 Hz)
    before resampling, where interpolation artefacts cannot shift onsets.
    Peaks that fall inside a large-gap window are discarded.

    Returns
    -------
    scr_times      : np.ndarray — SCR onset times in seconds from recording start.
    scr_amplitudes : np.ndarray — Peak amplitude in z-score units of the phasic signal.
    scr_prominences: np.ndarray — Peak prominence in z-score units.
    """
    min_distance = int(fs * _SCR_MIN_DISTANCE_S)
    peaks, properties = find_peaks(
        np.nan_to_num(phasic_down, nan=0.0),
        height=_SCR_MIN_HEIGHT,
        prominence=_SCR_MIN_PROMINENCE,
        distance=max(1, min_distance),
    )

    if gap_mask_down is not None and np.any(gap_mask_down):
        keep           = ~gap_mask_down[peaks]
        peaks          = peaks[keep]
        for key in properties:
            properties[key] = properties[key][keep]

    return peaks / fs, properties["peak_heights"], properties["prominences"]


# -------------------------------------------------------------------
# Diagnostic figure
# -------------------------------------------------------------------

def _save_eda_figure(timestamps, eda_clean, tonic, scr_times, out_dir):
    """
    Save ``eda_overview.png`` to *out_dir*.

    The artifact-rejected EDA signal is normalised to share the same
    y-scale as the cvxEDA tonic component. NaN values (large gaps) are
    rendered as natural breaks in the trace. SCR events are red vertical
    lines.

    Parameters
    ----------
    timestamps : np.ndarray  — time axis in seconds (51.2 Hz grid)
    eda_clean  : np.ndarray  — artifact-rejected EDA; NaN inside large gaps
    tonic      : np.ndarray  — tonic component resampled to original length
    scr_times  : np.ndarray  — SCR event times in seconds
    out_dir    : str         — directory where the figure is saved
    """
    os.makedirs(out_dir, exist_ok=True)

    # Z-score using only valid (non-NaN) samples
    valid = ~np.isnan(eda_clean)
    eda_z = np.full_like(eda_clean, np.nan)
    if valid.sum() > 1:
        mu, sigma = np.mean(eda_clean[valid]), np.std(eda_clean[valid])
        eda_z[valid] = (eda_clean[valid] - mu) / (sigma + 1e-12)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(timestamps, eda_z, color="black", linewidth=0.8, label="EDA (filtered, z-scored)")
    ax.plot(timestamps, tonic,  color="blue",  linewidth=1.2, label="Tonic EDA")

    for t in scr_times:
        ax.axvline(x=t, color="red", linewidth=0.7, alpha=0.8)
    if len(scr_times):
        ax.axvline(x=scr_times[0], color="red", linewidth=0.7, alpha=0.8,
                   label=f"SCR events (n={len(scr_times)})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (z-score)")
    ax.set_title("EDA — filtered signal, tonic component and SCR events")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_overview.png"), dpi=150)
    plt.close(fig)
    printInfo(f"[edaProcessor] Saved EDA figure → {out_dir}/eda_overview.png")


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def computeArousal(recPath, show=False):
    """
    Compute arousal from EDA/GSR data using cvxEDA decomposition.

    Pipeline:
      1. Load raw GSR signal (kΩ) with original hardware timestamps.
      2. Resample to a strict 51.2 Hz grid (gap-aware, in kΩ); save resampled
         data and a statistics report to ``features/eda/``.
      3. Convert kΩ → µS  (conductance = 1000 / resistance). All subsequent
         steps operate in µS.
      4. Artifact rejection (MAD-based, in µS), preserving large-gap NaN.
      5. cvxEDA tonic/phasic decomposition (gap regions restored to NaN).
      6. SCR event detection on the 4 Hz phasic component.
      7. Save diagnostic figure to ``features/eda/eda_overview.png``.
      8. Save results to ``results/Arousal.csv``.

    Writes:
      - ``features/eda/eda_resampled.csv``    (resistance, kΩ, before conversion)
      - ``features/eda/eda_resample_report.txt``
      - ``features/eda/eda_overview.png``      (conductance, µS)
      - ``results/Arousal.csv``  columns: [Timestamp (s), TonicEDA, PhasicEDA]
        at 2 Hz / 0.5 s, matching the shared pipeline timebase.

    Args:
        recPath: Path to the recording directory.
        show:    Unused; kept for API compatibility.
    """
    printInfo("[edaProcessor] Computing Arousal")

    arousal_path = os.path.join(recPath, "results", "Arousal.csv")
    if os.path.isfile(arousal_path):
        printInfo("[edaProcessor] Arousal already computed, using cached results")
        return

    sig = _load_shimmer_eda_signal(recPath)
    if sig is None:
        return

    eda_features_dir = os.path.join(recPath, "features", "eda")

    # Step 1 — resample onto a regular 51.2 Hz grid using original timestamps.
    # Linear interpolation is performed in resistance space (kΩ) where it is
    # valid; conversion to conductance follows immediately after.
    timestamps, eda_raw, gap_mask = resample_to_grid(
        sig[0].values, sig[1].values, "eda", eda_features_dir
    )

    # Step 2 — convert resistance (kΩ) → conductance (µS).
    # cvxEDA and the MAD artifact detector both expect conductance: arousal
    # deflections are positive in µS (rising conductance) and the 0.05 µS
    # MAD floor is physiologically meaningful.
    # NaN samples (large gaps) propagate correctly through the division.
    eda_raw = 1000.0 / eda_raw

    try:
        # Step 4 — artifact rejection (preserves large-gap NaN)
        eda_clean = reject_eda_artifacts(eda_raw, gap_mask=gap_mask)

        # Step 5 — cvxEDA decomposition (forward-fills NaN for solver,
        #           then restores NaN in large-gap regions)
        tonic_down, phasic_down, gap_mask_down = _decompose_eda(
            eda_clean, gap_mask=gap_mask
        )

        n_original = len(eda_raw)
        tonic  = signal.resample(tonic_down,  n_original)
        phasic = signal.resample(phasic_down, n_original)

        # Restore NaN in large-gap regions of the final outputs
        if np.any(gap_mask):
            tonic[gap_mask]  = np.nan
            phasic[gap_mask] = np.nan

        # Step 6 — SCR event detection (peaks in gap regions discarded)
        scr_times, scr_amplitudes, scr_prominences = _detect_scr_events(
            phasic_down, gap_mask_down=gap_mask_down
        )

    except Exception as e:
        printError(f"[edaProcessor] cvxEDA decomposition failed: {e}")
        return

    # Step 7 — diagnostic figure (uses full 51.2 Hz arrays for visual fidelity)
    _save_eda_figure(timestamps, eda_clean, tonic, scr_times, eda_features_dir)

    # Step 7b — save per-event SCR properties to features/eda/SCR_events.csv
    scr_df = pd.DataFrame({
        "Timestamp":   scr_times,
        "Amplitude":   scr_amplitudes,
        "Prominence":  scr_prominences,
    })
    scr_df.to_csv(os.path.join(eda_features_dir, "SCR_events.csv"), index=False)
    printInfo(f"[edaProcessor] SCR events saved → {eda_features_dir}/SCR_events.csv")

    # Step 8 — downsample to 2 Hz / 0.5 s and save results CSV
    t_2hz, tonic_2hz, phasic_2hz = _downsample_to_2hz(timestamps, tonic, phasic)

    os.makedirs(os.path.join(recPath, "results"), exist_ok=True)
    components = pd.DataFrame({"TonicEDA": tonic_2hz, "PhasicEDA": phasic_2hz})
    components.insert(0, "Timestamp", t_2hz)
    components.to_csv(arousal_path, index=False)
    printInfo(f"[edaProcessor] Arousal computation completed → {arousal_path}")


def loadArousal(basepath):
    """
    Load arousal data from CSV.

    Returns
    -------
    numpy.ndarray or None
    """
    arousal_path = os.path.join(basepath, "results", "Arousal.csv")

    if not os.path.isfile(arousal_path):
        printWarning(f"[edaProcessor] Arousal file not found: {arousal_path}")
        return None

    try:
        df = pd.read_csv(arousal_path)
        return df.values
    except Exception as e:
        printError(f"[edaProcessor] Error loading arousal data: {e}")
        return None
