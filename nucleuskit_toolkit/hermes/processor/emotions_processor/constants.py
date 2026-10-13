"""Shared constants for the emotions processor."""

from nucleuskit_toolkit.hermes.constants import DISCONNECT_VALUE, OUTPUT_RMS_COLUMNS

NULL_WINDOW_NAN_THRESHOLD = 0.10
"""
Fraction of samples in a window that must be hardware-invalid (NaN) before
the whole window is flagged as a null window.
"""

EMOTION_COLUMNS: list[str] = [
    "Neutral",
    "Happiness",
    "Anger",
    "Surprise",
    "Contempt",
    "Disgust",
    "Fear",
    "Sadness",
]
"""Canonical emotion label order used in ``Emotions.csv`` (downstream consumers depend on this)."""

RMS_COLUMNS: list[str] = ["Timestamp", *OUTPUT_RMS_COLUMNS]
"""Column names for ``rmsSignals.csv`` (Timestamp + 8 output RMS names)."""

__all__ = [
    "DISCONNECT_VALUE",
    "NULL_WINDOW_NAN_THRESHOLD",
    "EMOTION_COLUMNS",
    "OUTPUT_RMS_COLUMNS",
    "RMS_COLUMNS",
]
