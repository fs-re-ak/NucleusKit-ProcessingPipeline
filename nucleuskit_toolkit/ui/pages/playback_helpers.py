# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 RE-AK Technologies Inc.
"""Pure-Python helpers for the playback page.

This module intentionally has **no Qt or pyqtgraph imports** so that its
functions can be imported and unit-tested in headless / CI environments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reak_core.models import EMOTION_COLUMNS

# ---------------------------------------------------------------------------
# Column ordering / allowlists
# ---------------------------------------------------------------------------

# Columns to display from Cognition.csv.  Only Engagement is shown for now;
# Focus, CognitiveLoad, Frontal, and Lateralization are kept for future use.
COGNITION_COLUMNS: set[str] = {"Engagement"}


def ordered_playback_columns(plot_cols: list[str]) -> list[str]:
    """Neutral first (drawn behind), then EMOTION_COLUMNS order, then any extras.

    Every column that reaches ``plot_cols`` is returned — including columns
    whose values are all-zero or all-NaN.  Do NOT add a filter for constant
    or empty series here.
    """
    seen: set[str] = set()
    out: list[str] = []
    if "Neutral" in plot_cols:
        out.append("Neutral")
        seen.add("Neutral")
    for c in EMOTION_COLUMNS:
        if c in plot_cols and c not in seen:
            out.append(c)
            seen.add(c)
    for c in plot_cols:
        if c not in seen:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Session time span
# ---------------------------------------------------------------------------

# Shared configuration for result metric files.  Mirrors _METRIC_CONFIGS in
# playback_page.py (kept in sync manually; the file names are the source of
# truth).
_RESULT_FILES = [
    "Emotions.csv",
    "Arousal.csv",
    "Cognition.csv",
    "HeartDynamics.csv",
]


def session_time_range(session_dir: str) -> tuple[float, float] | None:
    """Return ``(t_min, t_max)`` from the union of all available result CSVs.

    Only the Timestamp column is read so this stays fast.  The range covers
    every row in every result file, including rows whose metric values are
    ``NULL``/NaN — those timestamps are still part of the session and must
    appear on the X axis even though the corresponding Y values are empty.

    Returns ``None`` when no result file is readable.
    """
    root = Path(session_dir).expanduser().resolve()
    t_min: float | None = None
    t_max: float | None = None
    for filename in _RESULT_FILES:
        path = root / "results" / filename
        if not path.is_file():
            continue
        try:
            ts = pd.read_csv(path, usecols=["Timestamp"])["Timestamp"].to_numpy(dtype=np.float64)
        except Exception:
            continue
        if ts.size == 0:
            continue
        lo, hi = float(np.nanmin(ts)), float(np.nanmax(ts))
        if t_min is None or lo < t_min:
            t_min = lo
        if t_max is None or hi > t_max:
            t_max = hi
    if t_min is None or t_max is None:
        return None
    return t_min, t_max


# ---------------------------------------------------------------------------
# NaN-preserving smoothing
# ---------------------------------------------------------------------------


def smooth_y(y_raw: np.ndarray, k: int) -> np.ndarray:
    """Gaussian-kernel smooth, NaN-aware.  k=1 returns the array unchanged.

    The kernel standard deviation is chosen so that the Gaussian has the same
    second moment as a rectangular window of width ``k``::

        sigma = sqrt((k^2 - 1) / 12)          # matches box-filter variance

    Kernel support is ±3σ (rounded up), so the effective half-width grows with
    ``k`` but the taper is smooth rather than flat.

    NaN positions in the input are always NaN in the output so that
    leading/trailing/interior null regions remain as graph gaps.  Edges and
    samples neighbouring a gap are renormalised over the finite samples inside
    the kernel window (equivalent to ``min_periods=1`` in a rolling mean).
    """
    if k <= 1:
        return y_raw

    # Build the 1-D Gaussian kernel.
    sigma = np.sqrt((k * k - 1) / 12.0)
    if sigma < 1e-9:
        return y_raw
    half = int(np.ceil(3.0 * sigma))
    xs = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()

    # Replace NaN with 0 for convolution; track which positions are finite.
    finite_mask = np.isfinite(y_raw).astype(np.float64)
    y_filled = np.where(finite_mask, y_raw, 0.0)

    # Pad both arrays by `half` samples so that np.convolve in "valid" mode
    # returns exactly len(y_raw) samples.  This avoids the size mismatch that
    # np.convolve mode="same" introduces when the kernel is longer than the
    # signal (it returns max(N_signal, N_kernel) not N_signal).
    y_padded = np.pad(y_filled, half, constant_values=0.0)
    m_padded = np.pad(finite_mask, half, constant_values=0.0)

    # Convolve values and the finite-sample weights separately so that edge
    # and gap-adjacent samples are renormalised over available data.
    conv_values = np.convolve(y_padded, kernel, mode="valid")
    conv_weight = np.convolve(m_padded, kernel, mode="valid")

    # Where weight > 0 we have at least one finite neighbour; divide to get
    # the weighted mean.  Where weight == 0 all neighbours were NaN too.
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = np.where(conv_weight > 0.0, conv_values / conv_weight, np.nan)

    # Always restore NaN at positions that were NaN in the raw input so that
    # leading/trailing null regions are never painted on the graph.
    smoothed[np.isnan(y_raw)] = np.nan
    return smoothed
