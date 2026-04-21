"""
Real-time streaming EMG emotion classifier.

Provides ``StreamingEMGClassifier``, a stateful object that accepts one raw
EEG/EMG sample at a time and periodically fires an emotion probability
estimate using the trained two-stage classifier.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from sklearn.preprocessing import normalize

from nucleuskit_pipeline.hermes.emotion.config import DEFAULT_HW, HardwareConfig
from nucleuskit_pipeline.hermes.emotion.inference import TwoStageClassifier


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

        self._clf = TwoStageClassifier.load(
            clf_dir,
            cooldown_windows=cooldown_windows,
            threshold_override=threshold_override,
        )

        self._buffer: deque = deque(maxlen=self.window_samples)

        self._samples_since_classify: int = 0
        self._total_samples: int = 0

        nyq = sampling_rate / 2.0
        self._b, self._a = butter(
            hw.filter_order,
            [hw.bandpass_low / nyq, hw.bandpass_high / nyq],
            btype="bandpass",
        )

        self._zi: Optional[np.ndarray] = None
        self._last_channel_rms: Optional[np.ndarray] = None
        self._last_model_features: Optional[np.ndarray] = None
        self._last_predicted_label: Optional[str] = None
        self._last_predicted_confidence: Optional[float] = None

    def push_sample(self, raw_sample) -> Optional[Dict[str, float]]:
        """Feed one raw EEG sample and optionally receive a classification."""
        raw = np.asarray(raw_sample, dtype=float).reshape(1, self.n_channels)

        if self._zi is None:
            zi_1ch = lfilter_zi(self._b, self._a)
            self._zi = zi_1ch[:, np.newaxis] * raw

        filtered, self._zi = lfilter(
            self._b, self._a, raw, axis=0, zi=self._zi
        )

        self._buffer.append(filtered[0])
        self._samples_since_classify += 1
        self._total_samples += 1

        if (
            self._samples_since_classify >= self.step_samples
            and len(self._buffer) == self.window_samples
        ):
            self._samples_since_classify = 0
            return self._classify()

        return None

    def push_batch(self, raw_batch: np.ndarray):
        """Feed a 2-D batch of samples and yield (time, proba) pairs."""
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
        self._last_channel_rms = None
        self._last_model_features = None
        self._last_predicted_label = None
        self._last_predicted_confidence = None
        self._clf.reset_artefact_state()

    @property
    def time_sec(self) -> float:
        return self._total_samples / self.sampling_rate

    @property
    def labels(self) -> List[str]:
        return self._clf.all_classes

    @property
    def is_buffer_full(self) -> bool:
        return len(self._buffer) == self.window_samples

    @property
    def last_channel_rms(self) -> Optional[np.ndarray]:
        """Per-channel RMS (model channel order) from the last classification, or None."""
        return self._last_channel_rms

    @property
    def last_model_features(self) -> Optional[np.ndarray]:
        """1-D feature vector passed to :class:`TwoStageClassifier` (L2 RMS + AVG_RMS), or None."""
        return self._last_model_features

    @property
    def last_predicted_label(self) -> Optional[str]:
        """Discrete label from :meth:`TwoStageClassifier.predict` for the last window, or None."""
        return self._last_predicted_label

    @property
    def last_predicted_confidence(self) -> Optional[float]:
        """Confidence paired with :attr:`last_predicted_label`, or None."""
        return self._last_predicted_confidence

    @property
    def model_feature_columns(self) -> List[str]:
        """Ordered feature names matching ``feature_columns.json`` (same order as ``last_model_features``)."""
        return self._clf.feature_columns

    def _classify(self) -> Dict[str, float]:
        window = np.array(self._buffer)

        rms = np.sqrt(np.mean(window**2, axis=0))
        avg_rms = float(np.mean(rms))

        self._last_channel_rms = np.asarray(rms, dtype=float).copy()

        norm_rms = normalize(rms.reshape(1, -1)).squeeze()
        x = np.append(norm_rms, avg_rms)
        self._last_model_features = np.asarray(x, dtype=float).copy()

        proba, label, confidence = self._clf.infer(x)
        self._last_predicted_label = label
        self._last_predicted_confidence = float(confidence)
        return proba

    def __repr__(self) -> str:
        return (
            f"StreamingEMGClassifier("
            f"window={self.window_sec}s, step={self.step_sec}s, "
            f"fs={self.sampling_rate}Hz, "
            f"labels={self.labels}"
            f")"
        )
