"""
Abstract base class for pluggable emotion models.

Every model implementation must subclass :class:`EmotionModel` and implement
its abstract methods.  The pipeline runner depends only on this interface,
so swapping models requires nothing more than registering the new subclass
in ``models/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.preprocess import (
    DEFAULT_PREPROCESS_SPEC,
    PreprocessSpec,
)


@dataclass(frozen=True)
class EmotionWindowResult:
    """Output of one classification window."""

    probabilities: dict[str, float]
    """Per-emotion probability estimates keyed by title-case emotion label
    matching :data:`~...constants.EMOTION_COLUMNS`."""

    label: str
    """Winning emotion label (title-case, e.g. ``"Happiness"``)."""

    confidence: float
    """Probability of the winning label."""

    channel_rms: np.ndarray
    """Raw per-channel RMS (8 elements, Hermes channel order), computed from
    the filtered window samples before any model-internal normalisation.
    Written to ``rmsSignals.csv``."""

    model_features: np.ndarray
    """1-D feature vector actually passed to the classifier.
    Written to ``emotionClassifierInputs.csv``."""


class EmotionModel(ABC):
    """Stateful, pluggable emotion classifier interface."""

    @classmethod
    @abstractmethod
    def load(cls, release_dir: str | None = None, **kwargs) -> "EmotionModel":
        """Load model from *release_dir* and return a ready instance."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal calibration / artefact state between recordings."""
        ...

    def set_subject_context(self, windows: np.ndarray) -> None:
        """Calibrate per-subject normalisation stats before inference.

        Parameters
        ----------
        windows:
            Array of shape ``(n_windows, n_channels, n_samples)`` containing
            all valid windows from the current recording (already bandpass-
            filtered, no NaN).  The base implementation is a no-op; subclasses
            that require calibration must override this.
        """

    @abstractmethod
    def predict(self, samples: np.ndarray) -> EmotionWindowResult:
        """Classify one EXG window.

        Parameters
        ----------
        samples:
            Filtered EXG window of shape ``(n_channels, n_samples)``.

        Returns
        -------
        EmotionWindowResult
        """
        ...

    @property
    def model_name(self) -> str:
        """Short model family name, e.g. ``"classical-emotion"``.

        Override in subclasses that have a meaningful identity.  The base
        implementation returns an empty string so legacy adapters do not
        need to be changed.
        """
        return ""

    @property
    def model_version(self) -> str:
        """Semantic version string, e.g. ``"2.5.0"``.

        Override in subclasses that have a meaningful version.  The base
        implementation returns an empty string.
        """
        return ""

    @property
    def release_dir(self) -> Path | None:
        """Filesystem path of the loaded release, if the model is file-backed."""
        return None

    @property
    def preprocess_spec(self) -> PreprocessSpec:
        """Filter settings the window source must apply before ``predict``.

        Default is the 2.5.0 contract (15–40 Hz, no notch).
        """
        return DEFAULT_PREPROCESS_SPEC

    @property
    def emotions_csv_uses_argmax(self) -> bool:
        """When true, ``Emotions.csv`` stores a one-hot of the winning class.

        The default writes the per-class probability vector.  Releases that
        treat emotion intensity as the most likely class override this.
        """
        return False

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Ordered feature names matching the model input vector."""
        ...

    @property
    @abstractmethod
    def emotion_labels(self) -> list[str]:
        """All emotion class labels the model can output (title-case)."""
        ...
