"""
Shimmer timestamp-aware resampling.

Projects irregular Shimmer samples (effective ~48 Hz) onto a strict
nominal-rate grid using the original hardware timestamps and linear
interpolation. True disconnects (gaps >= GAP_THRESHOLD_S) are preserved
as NaN rather than being filled in.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Spring 2026
"""

import os
import textwrap

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from nucleuskit_pipeline.logging_utils import printInfo, printWarning
from nucleuskit_pipeline.shared.file_interface import normalise_timestamps_to_seconds  # noqa: F401 — re-exported for callers

NOMINAL_FS      = 51.2   # Hz — nominal Shimmer sampling rate
GAP_THRESHOLD_S = 5.0    # seconds — gaps at or above this are true disconnects


def resample_to_grid(
    timestamps,
    values,
    signal_name,
    out_dir,
    nominal_fs=NOMINAL_FS,
    gap_threshold_s=GAP_THRESHOLD_S,
):
    """
    Project an irregularly sampled Shimmer signal onto a strict nominal-rate grid.

    Parameters
    ----------
    timestamps : array-like
        Original hardware timestamps in seconds from recording start.
        Must be monotonically increasing.
    values : array-like
        Signal values corresponding to each timestamp.
    signal_name : str
        Short label used for the output filenames (e.g. ``"eda"``, ``"ppg"``).
    out_dir : str
        Directory where ``{signal_name}_resampled.csv`` and
        ``{signal_name}_resample_report.txt`` are written.
    nominal_fs : float
        Target sample rate of the output grid (default 51.2 Hz).
    gap_threshold_s : float
        Gaps >= this duration are treated as true disconnects and
        preserved as NaN in the output (default 5.0 s).

    Returns
    -------
    t_grid   : np.ndarray, shape (N,)
        Regular time axis at ``nominal_fs``, in seconds.
    resampled : np.ndarray, shape (N,)
        Interpolated signal values. Points inside large gaps are NaN.
    gap_mask : np.ndarray[bool], shape (N,)
        True at every grid point that falls inside a large gap.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    values     = np.asarray(values,     dtype=float)

    n_raw    = len(timestamps)
    t_first  = timestamps[0]
    t_last   = timestamps[-1]
    duration = t_last - t_first

    # ------------------------------------------------------------------
    # 1. Effective sample rate
    # ------------------------------------------------------------------
    effective_fs = (n_raw - 1) / duration if duration > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. Detect large gaps in the original timestamps
    # ------------------------------------------------------------------
    dt         = np.diff(timestamps)
    gap_idx    = np.where(dt >= gap_threshold_s)[0]
    gaps       = [(timestamps[i], timestamps[i + 1], dt[i]) for i in gap_idx]

    if gaps:
        printWarning(
            f"[shimmerResampler] {signal_name.upper()}: {len(gaps)} large gap(s) "
            f">= {gap_threshold_s:.0f} s detected — those regions will be NaN."
        )

    # ------------------------------------------------------------------
    # 3. Build regular grid
    # ------------------------------------------------------------------
    t_grid = np.arange(t_first, t_last, 1.0 / nominal_fs)
    n_grid = len(t_grid)

    # ------------------------------------------------------------------
    # 4. Interpolate (NaN outside bounds, linear between raw samples)
    # ------------------------------------------------------------------
    interpolator = interp1d(
        timestamps, values,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    resampled = interpolator(t_grid)

    # ------------------------------------------------------------------
    # 5. Build gap mask and NaN-out large-gap regions
    # ------------------------------------------------------------------
    gap_mask = np.zeros(n_grid, dtype=bool)
    for t_start, t_end, _ in gaps:
        gap_mask |= (t_grid >= t_start) & (t_grid <= t_end)

    resampled[gap_mask] = np.nan
    n_nan = int(gap_mask.sum())

    # ------------------------------------------------------------------
    # 6. Statistics
    # ------------------------------------------------------------------
    # Net new samples introduced by the denser regular grid (excluding
    # samples that fall inside true-gap NaN regions).
    n_added = max(0, n_grid - n_raw - n_nan)

    # ------------------------------------------------------------------
    # 7. Save resampled CSV
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    resampled_csv = os.path.join(out_dir, f"{signal_name}_resampled.csv")
    pd.DataFrame({"Timestamp": t_grid, "Value": resampled}).to_csv(
        resampled_csv, index=False
    )

    # ------------------------------------------------------------------
    # 8. Save report
    # ------------------------------------------------------------------
    label = signal_name.upper()
    gap_lines = "".join(
        f"  Gap {k:>2d}:  t = {t_s:8.2f} s  →  {t_e:8.2f} s   duration = {dur:.2f} s\n"
        for k, (t_s, t_e, dur) in enumerate(gaps, start=1)
    ) or "  (none)\n"

    report = textwrap.dedent(f"""\
        Shimmer {label} Resample Report
        {'=' * (len(label) + 24)}
        Raw samples             : {n_raw}
        Duration                : {duration:.2f} s
        Effective sample rate   : {effective_fs:.2f} Hz  (nominal: {nominal_fs} Hz)
        Grid samples at {nominal_fs} Hz : {n_grid}
        NaN samples (large gaps): {n_nan}  ({n_nan / nominal_fs:.2f} s)
        Added by interpolation  : {n_added}  (grid - raw - NaN)

        Large gaps detected (>= {gap_threshold_s:.1f} s)
    """) + gap_lines

    report_path = os.path.join(out_dir, f"{signal_name}_resample_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    printInfo(
        f"[shimmerResampler] {label}: {n_raw} raw → {n_grid} grid samples "
        f"({n_added} added, {n_nan} NaN), eff. {effective_fs:.2f} Hz. "
        f"Report → {report_path}"
    )

    return t_grid, resampled, gap_mask
