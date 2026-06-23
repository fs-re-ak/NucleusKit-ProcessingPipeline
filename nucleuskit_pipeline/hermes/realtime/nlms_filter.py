"""Causal real-time adaptation of the batch NLMS adaptive decorrelation algorithm.

Reference: referenceCode/adaptiveFilter/process_eeg.py

Algorithm summary
-----------------
For each channel i, the other n_channels-1 channels serve as predictors.
A weight vector w (one per channel) estimates and subtracts the cross-channel
contribution from channel i, leaving a decorrelated residual.

Weights are updated only during "quiet" periods — when the local RMS envelope
is below a threshold relative to a running background estimate — to avoid
adapting on high-amplitude EMG bursts.

Batch → causal adaptations
---------------------------
- quiet_mask: replaced by a causal sliding-window RMS (ring buffer over the
  last RMS_WIN_SAMPLES samples) compared against a slow exponential moving
  average of that RMS (approximates the batch median).
- All other hyperparameters are carried over unchanged from the reference.
"""

from __future__ import annotations

import numpy as np

# ── Hyperparameters (mirrored from reference) ─────────────────────────────────
MU              = 0.05      # NLMS step size
EPSILON         = 1e-6      # normalisation floor
W_MAX           = 0.5       # per-weight magnitude clip
LEAKAGE         = 1e-4      # weight decay factor
RMS_WIN_SEC     = 0.10      # sliding RMS window length (s)
QUIET_THRESH_F  = 0.50      # quiet if rms_env < factor × ema_rms
CORRECTION_CAP  = 0.50      # cap correction at this fraction of |x_i|
EMA_ALPHA       = 0.005     # EMA decay for background RMS (≈ batch median)


class CausalNLMSFilter:
    """Sample-by-sample NLMS adaptive decorrelation for n_channels EEG/EMG.

    Usage
    -----
    filt = CausalNLMSFilter(n_channels=8, fs=250.0)
    out  = filt.push_batch(raw_batch)   # (n, 8) → (n, 8)
    filt.reset()                        # zero all state (on toggle-off)
    """

    def __init__(self, n_channels: int = 8, fs: float = 250.0) -> None:
        self._n_ch = n_channels
        self._fs   = fs
        self._rms_win = max(1, int(RMS_WIN_SEC * fs))

        # Predictor indices: for channel i → all other channel indices
        self._predictor_idx = [
            [j for j in range(n_channels) if j != i]
            for i in range(n_channels)
        ]

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
                pidx  = self._predictor_idx[i]
                p     = x[pidx]                     # (n_channels-1,)
                c_hat = float(self._w[i] @ p)
                xi    = float(x[i])

                # Cap correction energy
                if abs(c_hat) > CORRECTION_CAP * abs(xi):
                    scale = CORRECTION_CAP * abs(xi) / (abs(c_hat) + 1e-12)
                    c_hat *= scale

                y = xi - c_hat
                out[t, i] = y

                # ── Quiet-period detection ────────────────────────────────────
                # Push xi² into ring buffer, compute current RMS envelope
                buf    = self._rms_sq_buf[i]
                idx    = self._rms_buf_idx[i]
                buf[idx] = xi * xi
                self._rms_buf_idx[i] = (idx + 1) % self._rms_win

                rms_env = float(np.sqrt(np.mean(buf)))

                # Update slow EMA of RMS (background level ≈ batch median)
                ema = self._ema_rms[i]
                ema = (1.0 - EMA_ALPHA) * ema + EMA_ALPHA * rms_env
                self._ema_rms[i] = ema

                quiet = rms_env < QUIET_THRESH_F * (ema + EPSILON)

                # ── NLMS weight update ────────────────────────────────────────
                if quiet:
                    norm = EPSILON + float(p @ p)
                    self._w[i] = (1.0 - LEAKAGE) * self._w[i] + (MU / norm) * y * p
                    self._w[i] = np.clip(self._w[i], -W_MAX, W_MAX)

        return out

    def reset(self) -> None:
        """Zero all adaptive weights and buffer state."""
        self._reset_state()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        n_pred = self._n_ch - 1
        self._w           = [np.zeros(n_pred) for _ in range(self._n_ch)]
        self._rms_sq_buf  = [np.zeros(self._rms_win) for _ in range(self._n_ch)]
        self._rms_buf_idx = [0] * self._n_ch
        self._ema_rms     = [0.0] * self._n_ch
