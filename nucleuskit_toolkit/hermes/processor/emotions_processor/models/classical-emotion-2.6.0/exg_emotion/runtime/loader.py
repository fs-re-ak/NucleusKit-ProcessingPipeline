"""Model loader: read a release directory and instantiate an EmotionModel.

The loader is the only place that knows how to map the ``runtime.implementation``
field in ``manifest.json`` to a concrete Python class.  Training and experiment
code are never imported through this path.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from exg_emotion.core.interfaces import EmotionModel
from exg_emotion.core.exceptions import ModelNotFoundError


def load_from_dir(release_dir: str | Path) -> EmotionModel:
    """Instantiate an EmotionModel from a release artifact directory.

    The directory must contain a ``manifest.json`` with a
    ``runtime.implementation`` field of the form
    ``"package.module:ClassName"``.

    Parameters
    ----------
    release_dir:
        Path to ``releases/<name>/<version>/``.

    Returns
    -------
    EmotionModel
    """
    release_dir = Path(release_dir)
    manifest_path = release_dir / "manifest.json"

    if not manifest_path.exists():
        raise ModelNotFoundError(
            f"No manifest.json found in {release_dir}"
        )

    manifest = json.loads(manifest_path.read_text())
    implementation = manifest.get("runtime", {}).get("implementation")

    if not implementation:
        raise ModelNotFoundError(
            f"manifest.json in {release_dir} has no 'runtime.implementation' field"
        )

    cls = _import_class(implementation)

    # Each model class is responsible for providing a from_release_dir factory.
    if not hasattr(cls, "from_release_dir"):
        raise ModelNotFoundError(
            f"{implementation} does not implement 'from_release_dir(release_dir)'"
        )

    return cls.from_release_dir(release_dir)


def _import_class(dotted: str):
    """Import ``"package.module:ClassName"`` and return the class."""
    if ":" not in dotted:
        raise ModelNotFoundError(
            f"implementation string must be 'module:Class', got '{dotted}'"
        )
    module_path, class_name = dotted.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ModelNotFoundError(
            f"Could not import module '{module_path}': {exc}"
        ) from exc

    if not hasattr(module, class_name):
        raise ModelNotFoundError(
            f"Module '{module_path}' has no class '{class_name}'"
        )

    return getattr(module, class_name)
