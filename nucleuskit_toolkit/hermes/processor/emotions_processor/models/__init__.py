"""
Emotion model registry.

Vendored releases are discovered automatically: any subdirectory of this
package that contains ``manifest.json`` is selectable by directory name.
``classical-emotion`` remains an alias for ``classical-emotion-2.5.0``.

Adding a classical-emotion release:
  1. Copy the release folder here (``manifest.json``, ``config.json``,
     ``pipeline.joblib``, ``exg_emotion/``, plus any extra artifacts).
  2. Declare ``bandpass_hz`` / ``notch_hz`` in ``config.json``.
  3. Add a ``package-data`` entry in ``pyproject.toml``.
"""

from __future__ import annotations

from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.model import EmotionModel
from nucleuskit_toolkit.hermes.processor.emotions_processor.models.catalog import (
    DEFAULT_MODEL,
    DEFAULT_RELEASE_KEY,
    EmotionModelInfo,
    available_model_keys,
    canonicalize_model_key,
    list_models,
    load_experiment_id,
    load_preprocess_spec,
    resolve_model,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.models.classical_emotion import (
    ClassicalEmotionAdapter,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_RELEASE_KEY",
    "ClassicalEmotionAdapter",
    "EmotionModelInfo",
    "available_model_keys",
    "canonicalize_model_key",
    "get_model",
    "list_models",
    "load_experiment_id",
    "load_preprocess_spec",
    "resolve_model",
]


def get_model(name: str | None = None, **kwargs) -> EmotionModel:
    """
    Load and return an :class:`EmotionModel` by registry name.

    Parameters
    ----------
    name:
        Registry key (directory name) or the ``classical-emotion`` alias.
        Defaults to :data:`DEFAULT_MODEL` (2.5.0).
    **kwargs:
        Forwarded to the model's ``load()`` classmethod.  An explicit
        ``release_dir`` bypasses discovery.
    """
    release_dir = kwargs.pop("release_dir", None)
    if release_dir:
        return ClassicalEmotionAdapter.load(release_dir=release_dir, **kwargs)
    info = resolve_model(name)
    return ClassicalEmotionAdapter.load(release_dir=str(info.release_dir), **kwargs)
