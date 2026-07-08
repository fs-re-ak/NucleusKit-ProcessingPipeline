"""
Emotions Processor

Computes emotional states from 8-channel surface EMG recorded by the Hermes
device using a pluggable two-stage classifier.

Public API::

    from nucleuskit_toolkit.hermes.processor.emotions_processor import (
        computeEmotions,
        EMOTION_COLUMNS,
    )

Model selection::

    computeEmotions(recpath, model_name="v12")   # default; explicit

Architecture overview:

- ``constants.py``    — shared constants (EMOTION_COLUMNS, thresholds, …)
- ``session.py``      — ``computeEmotions()`` orchestration (I/O, incremental logic)
- ``pipeline.py``     — ``run_window_pipeline()`` — single classification loop
- ``report.py``       — visual statistics report generator
- ``interface/``      — ``EmotionModel`` ABC, ``EmotionWindow``, window sources
- ``models/``         — model registry + per-model implementations (``v12/``, …)

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter–Spring 2026
"""

from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import EMOTION_COLUMNS
from nucleuskit_toolkit.hermes.processor.emotions_processor.session import computeEmotions

__all__ = ["computeEmotions", "EMOTION_COLUMNS"]
