"""Domain exceptions for the exg_emotion system."""

from __future__ import annotations


class ExgEmotionError(Exception):
    """Base class for all exg_emotion errors."""


class InputValidationError(ExgEmotionError):
    """Raised when a signal window does not match the model's input spec.

    Attributes
    ----------
    expected
        What the model declared in its manifest.
    got
        What the caller actually passed.
    """

    def __init__(self, message: str, expected=None, got=None) -> None:
        super().__init__(message)
        self.expected = expected
        self.got = got


class ModelNotFoundError(ExgEmotionError):
    """Raised when the registry cannot locate a requested model or alias."""


class ReleaseExistsError(ExgEmotionError):
    """Raised when attempting to overwrite an existing immutable release."""


class ExperimentNotFoundError(ExgEmotionError):
    """Raised when the requested experiment directory does not exist."""
