"""
V12EmotionModel — concrete :class:`EmotionModel` backed by ``TwoStageClassifier``.

Feature construction (L2-normalised channel RMS + AVG_RMS) is centralised here,
replacing the duplicated logic that previously existed in both
``StreamingEMGClassifier._classify()`` and ``_model_features_from_channel_rms()``.
"""

from __future__ import annotations

from os import path

import numpy as np
from sklearn.preprocessing import normalize

from nucleuskit_pipeline.hermes.processor.emotions_processor.interface.model import (
    EmotionModel,
    EmotionWindowResult,
)
from .inference import TwoStageClassifier
from .config import DEFAULT_HW, HardwareConfig


class V12EmotionModel(EmotionModel):
    """
    ModelV12 — two-stage discriminant classifier with AVG_RMS artefact gate.

    Parameters
    ----------
    clf:
        Loaded :class:`TwoStageClassifier` instance.
    hw:
        Hardware configuration (used for channel count / naming metadata).
    """

    def __init__(self, clf: TwoStageClassifier, hw: HardwareConfig = DEFAULT_HW) -> None:
        self._clf = clf
        self._hw = hw

    @classmethod
    def load(cls, weights_dir: str | None = None, **kwargs) -> "V12EmotionModel":
        """
        Load weights from *weights_dir* (defaults to the bundled ``weights/``
        directory shipped with this package).

        Accepted keyword arguments are forwarded to
        :meth:`TwoStageClassifier.load` (e.g. ``cooldown_windows``,
        ``threshold_override``).
        """
        if weights_dir is None:
            weights_dir = path.join(path.dirname(path.abspath(__file__)), "weights")
        clf = TwoStageClassifier.load(weights_dir, **kwargs)
        return cls(clf)

    def reset(self) -> None:
        """Reset artefact-gate cooldown state between recordings."""
        self._clf.reset_artefact_state()

    def infer_from_rms(self, channel_rms: np.ndarray) -> EmotionWindowResult:
        """
        Classify one window from its raw per-channel RMS.

        Constructs the 9-element feature vector (L2-normalised channel RMS +
        scalar AVG_RMS) and delegates to :class:`TwoStageClassifier`.
        """
        rms = np.asarray(channel_rms, dtype=float).reshape(self._hw.n_channels)
        avg_rms = float(np.mean(rms))
        norm_rms = normalize(rms.reshape(1, -1)).squeeze()
        x = np.append(norm_rms, avg_rms)

        proba, label, confidence = self._clf.infer(x)

        return EmotionWindowResult(
            probabilities=proba,
            label=label,
            confidence=float(confidence),
            channel_rms=np.asarray(rms, dtype=float).copy(),
            model_features=np.asarray(x, dtype=float).copy(),
        )

    @property
    def feature_columns(self) -> list[str]:
        return self._clf.feature_columns

    @property
    def emotion_labels(self) -> list[str]:
        return self._clf.all_classes

    def __repr__(self) -> str:
        return f"V12EmotionModel(labels={self.emotion_labels})"
