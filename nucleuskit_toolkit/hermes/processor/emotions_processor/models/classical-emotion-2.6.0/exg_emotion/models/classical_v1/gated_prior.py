"""Gated p95 + cold-start prior for 2.6.0 amplitude calibration.

Runtime module (bundled in the release). Rest median is the session median
of the windows passed to ``set_subject_context``. p95 is computed only on
windows above ``k`` times that rest median. Scale shrinks from
``rest * prior_ratio`` toward the session gated p95 as active windows
accumulate (``n0`` pseudo-counts).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from exg_emotion.models.classical_v1.features import (
    _DEFAULT_EMG_INDICES,
    compute_rms_batch,
    compute_wl_batch,
    overall_emg_power,
)

GATE_K = 2.0
PRIOR_N0 = 10.0
MIN_ACTIVE = 1
RMS_CHANNEL_NAMES = [
    "Supraorbital_L",
    "Supraorbital_R",
    "Zygomatic_L",
    "Zygomatic_R",
    "Temporal_EEG",
    "Glabella",
    "Temporal_R",
    "Nasolabial",
]
WL_CHANNEL_NAMES = [n for n in RMS_CHANNEL_NAMES if n != "Temporal_EEG"]


def expressive_window_mask(
    windows: np.ndarray,
    k: float = GATE_K,
) -> tuple[np.ndarray, float]:
    """Windows whose mean EMG RMS exceeds k times the session median."""
    if windows.shape[0] == 0:
        return np.zeros((0,), dtype=bool), 0.0
    rms = compute_rms_batch(windows)
    power = overall_emg_power(rms, _DEFAULT_EMG_INDICES)
    rest = float(np.median(power))
    return power > (k * rest), rest


def _gated_p95_block(
    M: np.ndarray, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rest = np.median(M, axis=0)
    thresh = k * rest
    mask = M > thresh
    n_active = mask.sum(axis=0).astype(int)
    p95 = np.array(rest, copy=True)
    for ch in range(M.shape[1]):
        vals = M[mask[:, ch], ch]
        if vals.size >= MIN_ACTIVE:
            p95[ch] = float(np.percentile(vals, 95))
    loc = rest
    scale = np.maximum(p95 - loc, 1e-12)
    return loc, scale, n_active, p95


def gated_scale_result(windows: np.ndarray, k: float = GATE_K) -> dict[str, Any]:
    rms = compute_rms_batch(windows)
    wl = compute_wl_batch(windows[:, _DEFAULT_EMG_INDICES, :])
    rms_loc, rms_scale, n_rms, p95_rms = _gated_p95_block(rms, k)
    wl_loc, wl_scale, n_wl, p95_wl = _gated_p95_block(wl, k)
    mask, rest_power = expressive_window_mask(windows, k=k)
    n_expr = int(mask.sum())
    n_win = int(windows.shape[0])
    return {
        "rms_loc": rms_loc,
        "rms_scale": rms_scale,
        "wl_loc": wl_loc,
        "wl_scale": wl_scale,
        "n_active_rms": n_rms,
        "n_active_wl": n_wl,
        "p95_rms": p95_rms,
        "p95_wl": p95_wl,
        "rest_rms": rms_loc,
        "rest_wl": wl_loc,
        "n_expressive": n_expr,
        "n_windows": n_win,
        "expressive_fraction": float(n_expr / n_win) if n_win else 0.0,
        "rest_power": rest_power,
    }


def gated_p95_stats(
    windows: np.ndarray,
    k: float = GATE_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = gated_scale_result(windows, k=k)
    return g["rms_loc"], g["rms_scale"], g["wl_loc"], g["wl_scale"]


def apply_prior(
    gated: dict[str, Any],
    prior_rms: np.ndarray,
    prior_wl: np.ndarray,
    n0: float = PRIOR_N0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rms = np.asarray(gated["n_active_rms"], dtype=float)
    n_wl = np.asarray(gated["n_active_wl"], dtype=float)
    w_rms = n_rms / (n_rms + float(n0))
    w_wl = n_wl / (n_wl + float(n0))
    rest_rms = np.asarray(gated["rest_rms"], dtype=float)
    rest_wl = np.asarray(gated["rest_wl"], dtype=float)
    cold_rms = rest_rms * np.asarray(prior_rms, dtype=float)
    cold_wl = rest_wl * np.asarray(prior_wl, dtype=float)
    p95_rms = w_rms * np.asarray(gated["p95_rms"], dtype=float) + (1.0 - w_rms) * cold_rms
    p95_wl = w_wl * np.asarray(gated["p95_wl"], dtype=float) + (1.0 - w_wl) * cold_wl
    rms_scale = np.maximum(p95_rms - rest_rms, 1e-12)
    wl_scale = np.maximum(p95_wl - rest_wl, 1e-12)
    return rest_rms, rms_scale, rest_wl, wl_scale


def gated_prior_stats(
    windows: np.ndarray,
    prior_rms: np.ndarray,
    prior_wl: np.ndarray,
    k: float = GATE_K,
    n0: float = PRIOR_N0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return apply_prior(gated_scale_result(windows, k=k), prior_rms, prior_wl, n0=n0)


def prior_weight(n_active: np.ndarray, n0: float = PRIOR_N0) -> float:
    """Mean across channels of n / (n + n0). 0 = pure prior, 1 = pure gated p95."""
    n = np.asarray(n_active, dtype=float)
    return float(np.mean(n / (n + float(n0)))) if n.size else 0.0


def load_prior_table(path) -> dict[str, Any]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "prior_rms": np.asarray(data["prior_rms"], dtype=float),
        "prior_wl": np.asarray(data["prior_wl"], dtype=float),
        "gate_k": float(data.get("gate_k", GATE_K)),
        "n0": float(data.get("n0", PRIOR_N0)),
        "raw": data,
    }
