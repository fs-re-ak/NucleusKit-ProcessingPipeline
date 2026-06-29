"""
Abstract base class for pluggable emotion models.

Every model implementation must subclass :class:`EmotionModel` and implement
its abstract methods.  The pipeline runner depends only on this interface,
so swapping V12 for a future model requires nothing more than registering the
new subclass in ``models/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmotionWindowResult:
    """Output of one classification window."""

    probabilities: dict[str, float]
    """Per-emotion probability estimates keyed by emotion label."""

    label: str
    """Winning emotion label (highest probability, after artefact gating)."""

    confidence: float
    """Probability of the winning label."""

    channel_rms: np.ndarray
    """Raw per-channel RMS values (8 elements, model channel order), before normalisation."""

    model_features: np.ndarray
    """1-D feature vector actually passed to the classifier (L2 RMS + AVG_RMS)."""


class EmotionModel(ABC):
    """Stateful, pluggable emotion classifier interface."""

    @classmethod
    @abstractmethod
    def load(cls, weights_dir: str, **kwargs) -> "EmotionModel":
        """Load model weights from ``weights_dir`` and return a ready instance."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal artefact / cooldown state between recordings."""
        ...

    @abstractmethod
    def infer_from_rms(self, channel_rms: np.ndarray) -> EmotionWindowResult:
        """
        Classify one window given its per-channel RMS.

        Parameters
        ----------
        channel_rms:
            Raw per-channel RMS, shape ``(n_channels,)``.  The model is
            responsible for any normalisation (e.g. L2 + AVG_RMS).

        Returns
        -------
        EmotionWindowResult
        """
        ...

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Ordered feature names matching the model input vector."""
        ...

    @property
    @abstractmethod
    def emotion_labels(self) -> list[str]:
        """All emotion class labels the model can output."""
        ...
