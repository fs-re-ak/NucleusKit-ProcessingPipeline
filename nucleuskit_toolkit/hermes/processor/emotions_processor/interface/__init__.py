"""
Abstract emotion-model interface.

Defines the contract every emotion model must satisfy and the data structures
shared between the window sources and the pipeline runner.
"""

from .model import EmotionModel, EmotionWindowResult
from .windows import EmotionWindow, WindowSource

__all__ = [
    "EmotionModel",
    "EmotionWindowResult",
    "EmotionWindow",
    "WindowSource",
]
