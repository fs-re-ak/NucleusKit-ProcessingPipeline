"""Shared constants for the emotions processor."""

DISCONNECT_VALUE = 187500.0
"""Raw ADC value that indicates a hardware electrode disconnect."""

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

RMS_COLUMNS: list[str] = [
    "Timestamp",
    "AF8",
    "AF7",
    "CHEEK_R",
    "CHEEK_L",
    "EAR_R",
    "AFz",
    "BROW_L",
    "NOSE",
]
"""Column names for ``rmsSignals.csv`` (Timestamp + 8 canonical channel names)."""
