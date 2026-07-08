"""
Concrete WindowSource implementations.

- :class:`StreamingWindowSource` — accepts raw EMG ``(N, 8)`` and yields
  windows via bandpass filtering + ring-buffer windowing.  Mirrors the
  behaviour of the former ``StreamingEMGClassifier`` but stops before
  inference, yielding only per-window channel RMS.

- :class:`RmsCsvWindowSource` — reads an existing ``rmsSignals.csv`` and
  yields one window per row.  Used when raw EXG processing has already been
  done in a prior run.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn.preprocessing import normalize  # noqa: F401 (not used here, kept for clarity)

from nucleuskit_toolkit.hermes.rms_columns import (
    CANONICAL_CHANNEL_NAMES,
    normalize_rms_dataframe,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import (
    NULL_WINDOW_NAN_THRESHOLD,
)
from .windows import EmotionWindow


class StreamingWindowSource:
    """
    Bandpass-filter + windowing source that operates on a pre-loaded EMG array.

    Parameters
    ----------
    eeg_data:
        ``(N, 8)`` array of EMG samples in µV, already NaN-invalidated for
        hardware disconnects.  NaN values are replaced with 0 before filtering
        so the filter state does not collapse, but the NaN flag is preserved
        for null-window detection.
    timestamps:
        Optional ``(N,)`` hardware timestamp array (seconds).  When provided,
        window timestamps are the median of the hardware timestamps in the
        window buffer.  Falls back to the sample-count clock when ``None``.
    window_sec:
        Analysis window length in seconds.
    step_sec:
        Classification cadence in seconds.
    sampling_rate:
        EEG acquisition rate in Hz.
    bandpass_low / bandpass_high:
        Butterworth bandpass cutoffs in Hz.
    filter_order:
        Butterworth filter order.
    """

    def __init__(
        self,
        eeg_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        window_sec: float = 1.0,
        step_sec: float = 0.5,
        sampling_rate: int = 250,
        bandpass_low: float = 15.0,
        bandpass_high: float = 45.0,
        filter_order: int = 4,
    ) -> None:
        self._eeg = eeg_data
        self._ts = timestamps
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.sampling_rate = sampling_rate
        self.window_samples = int(window_sec * sampling_rate)
        self.step_samples = int(step_sec * sampling_rate)
        self.n_channels = eeg_data.shape[1]

        nyq = sampling_rate / 2.0
        self._b, self._a = butter(
            filter_order,
            [bandpass_low / nyq, bandpass_high / nyq],
            btype="bandpass",
        )

    def __iter__(self) -> Iterator[EmotionWindow]:
        buffer: deque = deque(maxlen=self.window_samples)
        nan_flags: deque = deque(maxlen=self.window_samples)
        ts_buffer: deque = deque(maxlen=self.window_samples)
        use_hw_ts = self._ts is not None

        zi: Optional[np.ndarray] = None
        samples_since_classify = 0
        total_samples = 0

        for sample_idx in range(self._eeg.shape[0]):
            raw_row = self._eeg[sample_idx, :]
            nan_flags.append(bool(np.isnan(raw_row).any()))
            if use_hw_ts:
                ts_buffer.append(self._ts[sample_idx])

            clean_row = np.nan_to_num(raw_row, nan=0.0).reshape(1, self.n_channels)

            if zi is None:
                zi_1ch = lfilter_zi(self._b, self._a)
                zi = zi_1ch[:, np.newaxis] * clean_row

            filtered, zi = lfilter(self._b, self._a, clean_row, axis=0, zi=zi)
            buffer.append(filtered[0])

            samples_since_classify += 1
            total_samples += 1

            if (
                samples_since_classify >= self.step_samples
                and len(buffer) == self.window_samples
            ):
                samples_since_classify = 0

                if use_hw_ts and len(ts_buffer) == self.window_samples:
                    ts = float(np.median(ts_buffer))
                else:
                    ts = total_samples / self.sampling_rate - self.window_sec / 2

                invalid = (sum(nan_flags) / len(nan_flags)) > NULL_WINDOW_NAN_THRESHOLD

                if invalid:
                    yield EmotionWindow(timestamp=ts, channel_rms=None, is_invalid=True)
                else:
                    window = np.array(buffer)
                    rms = np.sqrt(np.mean(window ** 2, axis=0))
                    yield EmotionWindow(
                        timestamp=ts,
                        channel_rms=np.asarray(rms, dtype=float),
                        is_invalid=False,
                    )


class RmsCsvWindowSource:
    """
    Window source backed by an existing ``rmsSignals.csv``.

    Each row in the CSV becomes one :class:`EmotionWindow`.  Rows with any
    NaN channel value are emitted as invalid windows (no reclassification).
    """

    def __init__(self, csv_path: str) -> None:
        self._df = normalize_rms_dataframe(pd.read_csv(csv_path))

    def __iter__(self) -> Iterator[EmotionWindow]:
        channel_cols = list(CANONICAL_CHANNEL_NAMES)
        for _, row in self._df.iterrows():
            ts = float(row["Timestamp"])
            rms = row[channel_cols].to_numpy(dtype=float)
            if np.isnan(rms).any():
                yield EmotionWindow(timestamp=ts, channel_rms=None, is_invalid=True)
            else:
                yield EmotionWindow(timestamp=ts, channel_rms=rms, is_invalid=False)
