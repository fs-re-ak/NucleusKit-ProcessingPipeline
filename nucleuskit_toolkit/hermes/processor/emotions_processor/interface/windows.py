"""
Window data structures and the WindowSource protocol.

A :class:`WindowSource` is any iterable that yields :class:`EmotionWindow`
objects.  Concrete implementations live in ``streaming.py``; external code
can supply custom sources for testing or offline replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass
class EmotionWindow:
    """One analysis window ready for classification."""

    timestamp: float
    """Representative timestamp for this window (seconds)."""

    samples: np.ndarray | None
    """Filtered EXG window of shape ``(n_channels, n_samples)``.
    ``None`` when ``is_invalid`` is ``True``."""

    is_invalid: bool
    """
    ``True`` when the window contains too many hardware-invalid (NaN) samples
    and should be emitted as a null row rather than classified.
    """


@runtime_checkable
class WindowSource(Protocol):
    """Structural protocol for window producers."""

    def __iter__(self) -> Iterator[EmotionWindow]:
        ...
