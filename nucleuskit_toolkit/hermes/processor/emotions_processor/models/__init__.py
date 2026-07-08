"""
Emotion model registry.

Adding a new model:
  1. Place implementation under ``models/<name>/``, implementing
     :class:`~...interface.model.EmotionModel`.
  2. Register it in ``_REGISTRY`` below.
  3. The pipeline picks it up via :func:`get_model`.
"""

from __future__ import annotations

from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.model import EmotionModel
from nucleuskit_toolkit.hermes.processor.emotions_processor.models.v12 import V12EmotionModel

_REGISTRY: dict[str, type[EmotionModel]] = {
    "v12": V12EmotionModel,
}

DEFAULT_MODEL = "v12"


def get_model(name: str | None = None, **kwargs) -> EmotionModel:
    """
    Load and return an :class:`EmotionModel` by registry name.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"v12"``).  Defaults to :data:`DEFAULT_MODEL`.
    **kwargs:
        Forwarded to the model's ``load()`` classmethod (e.g.
        ``weights_dir``, ``cooldown_windows``).
    """
    key = name or DEFAULT_MODEL
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown emotion model {key!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key].load(**kwargs)
