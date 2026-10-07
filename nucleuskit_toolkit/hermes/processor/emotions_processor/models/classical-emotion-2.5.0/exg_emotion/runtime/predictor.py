"""Thin production predictor: load once, call repeatedly.

This is the recommended entry point for any application that wants to run
EXG emotion classification without caring which model family is active.

Usage
-----
>>> from exg_emotion.runtime.predictor import Predictor
>>> predictor = Predictor.from_registry("releases/", model="production")
>>> prediction = predictor.predict(window)  # window: np.ndarray [8, 500]
>>> print(prediction.label)
>>> print(prediction.probabilities)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from exg_emotion.core.interfaces import EmotionModel
from exg_emotion.core.types import EmotionPrediction
from exg_emotion.runtime.registry import ModelRegistry


class Predictor:
    """Single-model predictor facade.

    Parameters
    ----------
    model:
        A loaded EmotionModel instance.
    """

    def __init__(self, model: EmotionModel) -> None:
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model.name

    @property
    def model_version(self) -> str:
        return self._model.version

    def set_subject_context(self, windows: np.ndarray) -> None:
        """Calibrate per-subject normalisation stats (required for extended_nm_a).

        Parameters
        ----------
        windows:
            Shape ``[n_windows, n_channels, n_samples]`` for this subject.
        """
        fn = getattr(self._model, "set_subject_context", None)
        if fn is None:
            return
        fn(windows)

    def predict(self, samples: np.ndarray) -> EmotionPrediction:
        """Classify one EXG window.

        Parameters
        ----------
        samples:
            Float32 array of shape ``[n_channels, n_samples]``.

        Returns
        -------
        EmotionPrediction
        """
        return self._model.predict(samples)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        releases_dir: str | Path,
        model: str = "production",
        version: Optional[str] = None,
    ) -> "Predictor":
        """Load a model through the registry and return a ready Predictor.

        Parameters
        ----------
        releases_dir:
            Path to the ``releases/`` directory.
        model:
            Model name or alias (e.g. ``"production"``).
        version:
            Optional explicit version; skips alias lookup when given.
        """
        registry = ModelRegistry(releases_dir)
        loaded = registry.load(model, version)
        return cls(loaded)

    @classmethod
    def from_release_dir(cls, release_dir: str | Path) -> "Predictor":
        """Load directly from a release artifact directory."""
        from exg_emotion.runtime.loader import load_from_dir
        return cls(load_from_dir(release_dir))
