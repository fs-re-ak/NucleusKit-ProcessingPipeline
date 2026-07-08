"""Causal real-time adaptation of the batch NLMS adaptive decorrelation algorithm.

Reference: referenceCode/adaptiveFilter/process_eeg.py

Algorithm summary
-----------------
For each channel i, selected other channels serve as predictors.
A weight vector w (one per channel) estimates and subtracts the cross-channel
contribution from channel i, leaving a decorrelated residual.

Weights are updated on every sample (no quiet-period gate), so the filter
adapts during high-amplitude EMG bursts where cross-channel crosstalk is
most visible.

Predictor connectivity is configurable via an optional bool mask; the real-time
viewer default excludes mutual regression among AF8, AF7, and BROW_L.
"""

from __future__ import annotations

import numpy as np

from nucleuskit_toolkit.hermes.constants import HermesConstants

# ── Hyperparameters (mirrored from reference) ─────────────────────────────────
MU              = 0.05      # NLMS step size
EPSILON         = 1e-6      # normalisation floor
W_MAX           = 0.5       # per-weight magnitude clip
LEAKAGE         = 1e-4      # weight decay factor
CORRECTION_CAP  = 1.0       # cap correction at this fraction of |x_i| (1 = no limit)

# Frontal cluster: no mutual cross-regression in the default real-time mask.
_FRONTAL_CLUSTER = {
    HermesConstants.CHANNELS["AF8"],
    HermesConstants.CHANNELS["AF7"],
    HermesConstants.CHANNELS["BROW_L"],
}


def build_default_predictor_mask(n_channels: int = 8) -> np.ndarray:
    """Return the default real-time predictor mask (n, n) bool.

    mask[i, j] is True when channel j may predict channel i.
    Diagonal is always False.  AF8, AF7, and BROW_L do not predict each other.
    """
    mask = np.ones((n_channels, n_channels), dtype=bool)
    np.fill_diagonal(mask, False)
    for i in _FRONTAL_CLUSTER:
        for j in _FRONTAL_CLUSTER:
            if i != j:
                mask[i, j] = False
    return mask


def predictor_indices_from_mask(mask: np.ndarray) -> list[list[int]]:
    """Convert a predictor mask to per-channel predictor index lists."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.shape[0]
    return [[j for j in range(n) if mask[i, j]] for i in range(n)]


def nlms_nan_safe(fn, data: np.ndarray) -> np.ndarray:
    """Apply an NLMS push_batch callable with NaN preservation.

    NaN cells (e.g. from electrode saturation masking) are zero-filled before
    the filter so that they do not corrupt weight updates. The NaN mask is
    restored on the output so downstream consumers see the same missing-sample
    pattern as before.

    Parameters
    ----------
    fn   : callable — typically CausalNLMSFilter.push_batch
    data : (n_samples, n_channels) array, may contain NaN

    Returns
    -------
    out : same shape as data; processed where finite, NaN where input was NaN
    """
    nan_mask = np.isnan(data)
    if nan_mask.any():
        filled = np.where(nan_mask, 0.0, data)
        result = fn(filled)
        result[nan_mask] = np.nan
        return result
    return fn(data)


class CausalNLMSFilter:
    """Sample-by-sample NLMS adaptive decorrelation for n_channels EEG/EMG.

    Usage
    -----
    filt = CausalNLMSFilter(n_channels=8, fs=250.0)
    out  = filt.push_batch(raw_batch)   # (n, 8) → (n, 8)
    filt.reset()                        # zero all state (on toggle-off)
    """

    def __init__(
        self,
        n_channels: int = 8,
        fs: float = 250.0,
        predictor_mask: np.ndarray | None = None,
    ) -> None:
        self._n_ch = n_channels
        self._fs   = fs

        if predictor_mask is None:
            self._predictor_idx = [
                [j for j in range(n_channels) if j != i]
                for i in range(n_channels)
            ]
        else:
            mask = np.asarray(predictor_mask, dtype=bool)
            if mask.shape != (n_channels, n_channels):
                raise ValueError(
                    f"predictor_mask must be ({n_channels}, {n_channels}), got {mask.shape}"
                )
            self._predictor_idx = predictor_indices_from_mask(mask)

        self._reset_state()

    # ── Public API ────────────────────────────────────────────────────────────

    def push_batch(self, batch: np.ndarray) -> np.ndarray:
        """Process a batch of samples through the causal NLMS filter.

        Parameters
        ----------
        batch : (n_samples, n_channels) float array

        Returns
        -------
        out : (n_samples, n_channels) decorrelated float array
        """
        batch = np.asarray(batch, dtype=np.float64)
        n_samples = batch.shape[0]
        out = np.empty_like(batch)

        for t in range(n_samples):
            x = batch[t]  # (n_channels,)

            for i in range(self._n_ch):
                pidx = self._predictor_idx[i]
                xi = float(x[i])

                if not pidx:
                    out[t, i] = xi
                    continue

                p = x[pidx]
                c_hat = float(self._w[i] @ p)

                # Cap correction energy
                if abs(c_hat) > CORRECTION_CAP * abs(xi):
                    scale = CORRECTION_CAP * abs(xi) / (abs(c_hat) + 1e-12)
                    c_hat *= scale

                y = xi - c_hat
                out[t, i] = y

                norm = EPSILON + float(p @ p)
                self._w[i] = (1.0 - LEAKAGE) * self._w[i] + (MU / norm) * y * p
                self._w[i] = np.clip(self._w[i], -W_MAX, W_MAX)

        return out

    def reset(self) -> None:
        """Zero all adaptive weights."""
        self._reset_state()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._w = [np.zeros(len(pidx)) for pidx in self._predictor_idx]
