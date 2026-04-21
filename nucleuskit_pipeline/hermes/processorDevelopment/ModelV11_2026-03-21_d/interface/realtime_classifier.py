"""
Real-time streaming EMG emotion classifier — ModelV10.

Provides ``StreamingEMGClassifier``, a stateful object that accepts one raw
EEG/EMG sample at a time and periodically fires an emotion probability
estimate using the trained two-stage classifier.

Design decisions
----------------
* **Causal single-pass Butterworth filter** with stored IIR state (``zi``).
  The batch pipeline uses zero-phase forward-backward filtering, which is not
  realisable in real time.  The single forward pass introduces a small phase
  delay (roughly half the filter order / sample-rate ≈ 16 ms at 250 Hz) but
  is otherwise equivalent in frequency response.  Because RMS is computed over
  a full window and we L2-normalise across channels, the systematic phase
  offset has negligible effect on the classifier output.

* **Circular buffer** (``collections.deque(maxlen=window_samples)``).  New
  filtered samples are appended; the oldest is automatically evicted.

* **Classification cadence** — fires every ``step_samples`` new samples,
  provided the buffer contains a full window.

* **Feature extraction** matches the batch RMSExtractor exactly:
  1. RMS over the window (per channel).
  2. ``AVG_RMS`` = mean of per-channel RMS, computed before L2 normalisation.
  3. L2-normalise the 8 channel RMS values.
  4. Feature vector = [normalised_rms..., avg_rms] — matches ``feature_columns.json``.

Usage
-----
  from realtime_classifier import StreamingEMGClassifier

  clf = StreamingEMGClassifier(clf_dir="…/classifier", window_sec=1.0)
  # With 2-window post-artefact cooldown:
  clf = StreamingEMGClassifier(clf_dir="…/classifier", window_sec=1.0, cooldown_windows=2)

  for row in eeg_stream:                     # row: (n_channels,) raw μV
      result = clf.push_sample(row)
      if result is not None:
          # result: {"Neutral": 0.12, "Anger": 0.05, …}
          print(f"t={clf.time_sec:.2f}s  {result}")
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_HERMES_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import DEFAULT_HW, HardwareConfig  # noqa: E402
from inference import TwoStageClassifier       # noqa: E402


class StreamingEMGClassifier:
    """Stateful, sample-by-sample EMG emotion classifier.

    Parameters
    ----------
    clf_dir:
        Path to the ``classifier/`` directory produced by ``train_and_save()``.
    window_sec:
        Analysis window length in seconds (1.0 or 2.0 recommended).
    step_sec:
        Classification cadence — how often a new estimate is produced, in
        seconds.  Default 0.5 s (every 125 samples at 250 Hz).
    sampling_rate:
        EEG acquisition rate in Hz.  Must match the rate used during training.
    hw:
        Hardware configuration (bandpass filter parameters and channel count).
    cooldown_windows:
        Number of classification windows to reject after a detected artefact.
        0 (default) = stateless, no cooldown.  > 0 = stateful; ``reset()``
        resets both the signal buffer and the cooldown counter.
    threshold_override:
        Override the AVG_RMS artefact gate threshold at runtime without
        retraining.  ``None`` (default) uses the value from
        ``artefact_config.json``.  Example: ``threshold_override=35.0``.
    """

    def __init__(
        self,
        clf_dir: str,
        window_sec: float = 1.0,
        step_sec: float = 0.5,
        sampling_rate: int = 250,
        hw: HardwareConfig = DEFAULT_HW,
        cooldown_windows: int = 0,
        threshold_override: Optional[float] = None,
    ) -> None:
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.sampling_rate = sampling_rate
        self.hw = hw

        self.window_samples: int = int(window_sec * sampling_rate)
        self.step_samples: int = int(step_sec * sampling_rate)
        self.n_channels: int = hw.n_channels

        # Two-stage classifier (loaded once)
        self._clf = TwoStageClassifier.load(
            clf_dir,
            cooldown_windows=cooldown_windows,
            threshold_override=threshold_override,
        )

        # Circular buffer — holds the last `window_samples` filtered samples
        self._buffer: deque = deque(maxlen=self.window_samples)

        # Counters
        self._samples_since_classify: int = 0
        self._total_samples: int = 0

        # Causal Butterworth bandpass filter coefficients
        nyq = sampling_rate / 2.0
        self._b, self._a = butter(
            hw.filter_order,
            [hw.bandpass_low / nyq, hw.bandpass_high / nyq],
            btype="bandpass",
        )

        # IIR filter state — shape (filter_ord, n_channels); initialised on
        # the first call to push_sample() using the actual signal level.
        self._zi: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_sample(self, raw_sample) -> Optional[Dict[str, float]]:
        """Feed one raw EEG sample and optionally receive a classification.

        Parameters
        ----------
        raw_sample : array-like, shape (n_channels,)
            One raw EMG frame in native acquisition units (μV).

        Returns
        -------
        dict[str, float] or None
            Probability dict ``{emotion_label: probability}`` when a
            classification fires (every ``step_samples`` new samples after the
            buffer is full).  Returns ``None`` otherwise.
        """
        raw = np.asarray(raw_sample, dtype=float).reshape(1, self.n_channels)

        # Lazy-init filter state from the first sample value
        if self._zi is None:
            zi_1ch = lfilter_zi(self._b, self._a)                  # (filter_ord,)
            # Broadcast to all channels, scaled by initial signal level
            self._zi = zi_1ch[:, np.newaxis] * raw                  # (filter_ord, n_ch)

        # Apply causal single-pass filter, updating state in-place
        filtered, self._zi = lfilter(
            self._b, self._a, raw, axis=0, zi=self._zi
        )                                                            # filtered: (1, n_ch)

        self._buffer.append(filtered[0])
        self._samples_since_classify += 1
        self._total_samples += 1

        # Fire only when the buffer is full AND the step cadence is reached
        if (
            self._samples_since_classify >= self.step_samples
            and len(self._buffer) == self.window_samples
        ):
            self._samples_since_classify = 0
            return self._classify()

        return None

    def push_batch(self, raw_batch: np.ndarray):
        """Feed a 2-D batch of samples and yield (time, proba) pairs.

        Parameters
        ----------
        raw_batch : np.ndarray, shape (n_samples, n_channels)

        Yields
        ------
        (time_sec: float, proba: dict[str, float])
        """
        for row in raw_batch:
            result = self.push_sample(row)
            if result is not None:
                yield self.time_sec, result

    def reset(self) -> None:
        """Reset internal state for a new recording."""
        self._buffer.clear()
        self._samples_since_classify = 0
        self._total_samples = 0
        self._zi = None
        self._clf.reset_artefact_state()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time_sec(self) -> float:
        """Wall-clock time position of the most recently processed sample."""
        return self._total_samples / self.sampling_rate

    @property
    def labels(self) -> List[str]:
        """All output emotion labels (Neutral first, then alphabetical)."""
        return self._clf.all_classes

    @property
    def is_buffer_full(self) -> bool:
        """True once enough samples have been accumulated to classify."""
        return len(self._buffer) == self.window_samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self) -> Dict[str, float]:
        """Extract RMS features from the current buffer and classify."""
        window = np.array(self._buffer)                             # (window_samples, n_ch)

        # Per-channel RMS
        rms = np.sqrt(np.mean(window ** 2, axis=0))                 # (n_ch,)
        avg_rms = float(np.mean(rms))

        # L2-normalise channel RMS only (matches RMSExtractor behaviour)
        norm_rms = normalize(rms.reshape(1, -1)).squeeze()          # (n_ch,)

        # Full feature vector: [normalised_rms..., avg_rms]
        x = np.append(norm_rms, avg_rms)

        return self._clf.predict_proba(x)

    def __repr__(self) -> str:
        return (
            f"StreamingEMGClassifier("
            f"window={self.window_sec}s, step={self.step_sec}s, "
            f"fs={self.sampling_rate}Hz, "
            f"labels={self.labels}"
            f")"
        )
