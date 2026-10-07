"""
Concrete WindowSource implementation.

:class:`StreamingWindowSource` accepts raw EMG ``(N, 8)`` and yields windows
by applying a zero-phase Butterworth bandpass to the **full recording** then
slicing into overlapping windows.  Each yielded window carries the filtered
``(n_channels, n_samples)`` array ready for model inference.

The former RmsCsvWindowSource is no longer supported: ``rmsSignals.csv`` is
now a write-only output, not a replay input.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import (
    NULL_WINDOW_NAN_THRESHOLD,
)
from .windows import EmotionWindow


class StreamingWindowSource:
    """
    Batch filtfilt + windowing source operating on a pre-loaded EMG array.

    Applies a zero-phase (``sosfiltfilt``) Butterworth bandpass to the
    **full recording** in one pass, then slides a window over the result
    with configurable length and hop.  Each :class:`EmotionWindow` carries
    the filtered ``(n_channels, window_samples)`` array.

    NaN-invalidated samples (hardware disconnects) are filled with zero
    before filtering so that the filter does not diverge, but the original
    NaN positions are preserved for null-window detection: a window where
    more than :data:`~...constants.NULL_WINDOW_NAN_THRESHOLD` of samples
    were originally invalid is emitted as an invalid window.

    Parameters
    ----------
    eeg_data:
        ``(N, 8)`` array of EMG samples already NaN-invalidated for hardware
        disconnects (i.e. :func:`~...session._invalidate_disconnected_samples`
        has been applied).
    timestamps:
        Optional ``(N,)`` hardware timestamp array (seconds).  Window
        timestamps are the median of the in-window hardware timestamps;
        falls back to the sample-count clock when ``None``.
    window_sec:
        Window duration in seconds.  Defaults to ``2.0`` (500 samples at
        250 Hz) to match the classical-emotion 2.5.0 training geometry.
    step_sec:
        Classification step size in seconds.  Defaults to ``0.5`` so the
        output grid matches the existing 2 Hz ``Emotions.csv`` schema.
    sampling_rate:
        Acquisition rate in Hz.
    bandpass_low / bandpass_high:
        Butterworth bandpass cutoffs in Hz.  Use ``15``–``40`` Hz (no NLMS).
    filter_order:
        Butterworth SOS filter order.
    """

    def __init__(
        self,
        eeg_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        window_sec: float = 2.0,
        step_sec: float = 0.5,
        sampling_rate: int = 250,
        bandpass_low: float = 15.0,
        bandpass_high: float = 40.0,
        filter_order: int = 4,
    ) -> None:
        self._timestamps = timestamps
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_sec * sampling_rate)
        self.step_samples = int(step_sec * sampling_rate)

        # Record which samples were originally NaN (before filling for filter).
        nan_mask = np.isnan(eeg_data)  # (N, 8)

        # Fill NaN with 0 so the filter propagates clean zeros rather than NaN.
        clean = np.where(nan_mask, 0.0, eeg_data).astype(np.float64)

        # Zero-phase bandpass applied once across the full recording.
        nyq = sampling_rate / 2.0
        sos = butter(
            filter_order,
            [bandpass_low / nyq, bandpass_high / nyq],
            btype="bandpass",
            output="sos",
        )
        self._filtered = sosfiltfilt(sos, clean, axis=0)  # (N, 8)

        # Per-sample flag: True when ANY channel was originally invalid.
        self._nan_any = nan_mask.any(axis=1)  # (N,)

    def __iter__(self) -> Iterator[EmotionWindow]:
        n_samples = self._filtered.shape[0]
        use_hw_ts = self._timestamps is not None

        start = 0
        while start + self.window_samples <= n_samples:
            end = start + self.window_samples

            # Null-window check: fraction of originally-NaN samples in window.
            nan_fraction = float(self._nan_any[start:end].mean())
            invalid = nan_fraction > NULL_WINDOW_NAN_THRESHOLD

            # Timestamp: median hardware clock or sample-count midpoint.
            if use_hw_ts:
                ts = float(np.median(self._timestamps[start:end]))
            else:
                mid = start + self.window_samples / 2
                ts = mid / self.sampling_rate

            if invalid:
                yield EmotionWindow(timestamp=ts, samples=None, is_invalid=True)
            else:
                # Shape: (n_channels, window_samples) = (8, 500)
                window = self._filtered[start:end, :].T.copy()
                yield EmotionWindow(
                    timestamp=ts,
                    samples=window.astype(np.float64),
                    is_invalid=False,
                )

            start += self.step_samples
