"""Abstract base class for all EmotionModel implementations.

This is the only surface that the production system touches.  Every model
family — classical, CNN, transformer — must satisfy this contract.  The
runtime layer never imports training, sklearn, PyTorch, or any other
framework-specific code through this path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from exg_emotion.core.types import EmotionPrediction


class EmotionModel(ABC):
    """Stateless callable that maps an EXG window to an EmotionPrediction.

    Implementations are responsible for their own input validation.  The
    recommended pattern is to call :func:`exg_emotion.core.validation.validate_input`
    at the top of ``predict()`` with the model's own manifest spec.
    """

    @abstractmethod
    def predict(self, samples: np.ndarray) -> EmotionPrediction:
        """Classify a single EXG window.

        Parameters
        ----------
        samples:
            Float32 array of shape ``[n_channels, n_samples]``.  The window
            must already be filtered and segmented by the caller.

        Returns
        -------
        EmotionPrediction
            The winning label, per-class probabilities, model identity, and
            optional signal-quality fields.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model family name, e.g. ``"classical-emotion"``."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string, e.g. ``"0.1.0"``."""
