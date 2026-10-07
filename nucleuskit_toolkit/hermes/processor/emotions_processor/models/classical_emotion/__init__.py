"""
classical-emotion 2.5.0 adapter for the NucleusKit emotions pipeline.

Wraps the vendored ``exg_emotion`` package (classical-emotion 2.5.0) as a
:class:`~...interface.model.EmotionModel` that the unified pipeline can load
from the registry under the key ``"classical-emotion"``.
"""

from .model import ClassicalEmotionAdapter

__all__ = ["ClassicalEmotionAdapter"]
