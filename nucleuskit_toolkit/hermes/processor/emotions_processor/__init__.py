"""
Emotions Processor

Computes emotional states from 8-channel surface EMG recorded by the Hermes
device.  The default model is classical-emotion 2.5.0 (logistic regression,
``extended_nm_a`` feature set).  Other vendored releases (e.g. 2.6.0) are
selected by registry key.

Public API::

    from nucleuskit_toolkit.hermes.processor.emotions_processor import (
        computeEmotions,
        EMOTION_COLUMNS,
    )

Model selection::

    computeEmotions(recpath)                              # default: classical-emotion (2.5.0)
    computeEmotions(recpath, model_name="classical-emotion")  # alias for 2.5.0
    computeEmotions(recpath, model_name="classical-emotion-2.6.0")

Architecture overview:

- ``constants.py``    — shared constants (EMOTION_COLUMNS, thresholds, …)
- ``session.py``      — ``computeEmotions()`` orchestration (I/O, incremental logic)
- ``pipeline.py``     — ``run_window_pipeline()`` — two-pass classification loop
- ``report.py``       — visual statistics report generator
- ``interface/``      — ``EmotionModel`` ABC, ``EmotionWindow``, window sources
- ``models/``         — model registry + ``classical_emotion/`` adapter

Signal path::

    raw 8-ch EXG
        → hardware invalidation (NaN)
        → release-specific notch + bandpass (from config.json; 2.5.0 is 15–40 Hz)
        → 2.0 s windows, 0.5 s hop (500 samples at 250 Hz)
        → set_subject_context (per-subject z-score calibration on valid windows)
        → predict each window (25-D feature vector → logistic regression)
        → Emotions.csv at 2 Hz

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter–Spring 2026
"""

from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import EMOTION_COLUMNS
from nucleuskit_toolkit.hermes.processor.emotions_processor.session import computeEmotions

__all__ = ["computeEmotions", "EMOTION_COLUMNS"]
