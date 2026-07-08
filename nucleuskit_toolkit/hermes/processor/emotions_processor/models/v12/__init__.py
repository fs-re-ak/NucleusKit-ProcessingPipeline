"""
V12 emotion model — TwoStageClassifier wrapper.

Implements :class:`~nucleuskit_toolkit.hermes.processor.emotions_processor.interface.model.EmotionModel`
for the bundled ModelV12 weights.

Usage::

    from nucleuskit_toolkit.hermes.processor.emotions_processor.models.v12 import V12EmotionModel

    model = V12EmotionModel.load(weights_dir)
    result = model.infer_from_rms(channel_rms)
"""

from .model import V12EmotionModel

__all__ = ["V12EmotionModel"]
