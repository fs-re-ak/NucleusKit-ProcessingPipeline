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
    """Centered rolling mean, NaN-aware.  k=1 returns the array unchanged.

    Input NaN positions are always NaN in the output so that
    leading/trailing/interior null regions remain as graph gaps after
    smoothing.  The rolling window skips NaN neighbours when computing the
    mean for finite positions (``min_periods=1``), which smooths across
    short interior gaps without inventing values at truly null samples.
    """
    if k <= 1:
        return y_raw
    smoothed = (
        pd.Series(y_raw)
        .rolling(k, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    # The centered window can pick up finite neighbours and write a value
    # into a position that was NaN in y_raw (e.g. the leading null region
    # of an Emotions.csv where the first few windows were invalid).
    # Restoring NaN at those positions keeps null regions empty on the graph.
    smoothed[np.isnan(y_raw)] = np.nan
    return smoothed
