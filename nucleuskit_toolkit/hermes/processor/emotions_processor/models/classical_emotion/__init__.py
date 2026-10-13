"""
classical-emotion adapter for the NucleusKit emotions pipeline.

Wraps a vendored ``exg_emotion`` release as an
:class:`~...interface.model.EmotionModel`.  The default load path is
``classical-emotion-2.5.0``; ``get_model`` can point the same adapter at
any discovered release directory.
"""

from .model import ClassicalEmotionAdapter

__all__ = ["ClassicalEmotionAdapter"]
