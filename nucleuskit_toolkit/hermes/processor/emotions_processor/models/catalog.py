"""Discover vendored emotion-model releases and their preprocess contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.preprocess import (
    PreprocessSpec,
)

_MODELS_DIR = Path(__file__).parent

# Backward-compatible alias used by existing callers of get_model().
DEFAULT_MODEL = "classical-emotion"
DEFAULT_RELEASE_KEY = "classical-emotion-2.5.0"


@dataclass(frozen=True)
class EmotionModelInfo:
    """One discovered release under ``emotions_processor/models/``."""

    key: str
    name: str
    version: str
    release_dir: Path
    display_label: str


def discover_releases() -> dict[str, EmotionModelInfo]:
    """Return ``{directory_name: info}`` for every folder that has ``manifest.json``."""
    found: dict[str, EmotionModelInfo] = {}
    if not _MODELS_DIR.is_dir():
        return found
    for child in sorted(_MODELS_DIR.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = str(manifest.get("name") or child.name)
        version = str(manifest.get("version") or "")
        label = f"{name} {version}".strip() if version else name
        found[child.name] = EmotionModelInfo(
            key=child.name,
            name=name,
            version=version,
            release_dir=child,
            display_label=label,
        )
    return found


def list_models() -> list[EmotionModelInfo]:
    """Discovered releases, sorted by registry key (directory name)."""
    return list(discover_releases().values())


def available_model_keys() -> list[str]:
    """Registry keys callers may pass, including the ``classical-emotion`` alias."""
    return sorted({DEFAULT_MODEL, *discover_releases()})


def canonicalize_model_key(name: str | None, version: str = "") -> str:
    """Map the legacy alias (and missing keys) onto a release directory name.

    ``classical-emotion`` and ``None`` resolve to 2.5.0 unless *version* names
    a different classical-emotion release.
    """
    if not name or name == DEFAULT_MODEL:
        if version and version != "2.5.0":
            return f"{DEFAULT_MODEL}-{version}"
        return DEFAULT_RELEASE_KEY
    return name


def resolve_model(name: str | None) -> EmotionModelInfo:
    """Resolve a registry key or alias to a discovered release.

    Raises
    ------
    ValueError
        If *name* is not the alias and does not match a release directory.
    """
    requested = name or DEFAULT_MODEL
    key = canonicalize_model_key(requested)
    releases = discover_releases()
    if key not in releases:
        raise ValueError(
            f"Unknown emotion model {requested!r}. Available: {available_model_keys()}"
        )
    return releases[key]


def load_preprocess_spec(release_dir: str | Path) -> PreprocessSpec:
    """Read bandpass/notch from a release ``config.json``.

    Missing or invalid ``bandpass_hz`` falls back to 15–40 Hz and no notch.
    ``notch_hz`` is optional once a bandpass is declared.
    """
    config_path = Path(release_dir) / "config.json"
    if not config_path.is_file():
        return PreprocessSpec(used_fallback=True)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PreprocessSpec(used_fallback=True)

    band = config.get("bandpass_hz")
    if not (isinstance(band, (list, tuple)) and len(band) == 2):
        return PreprocessSpec(used_fallback=True)

    notch_raw = config.get("notch_hz")
    notch_hz: float | None
    try:
        notch_hz = float(notch_raw) if notch_raw not in (None, "", 0, 0.0) else None
    except (TypeError, ValueError):
        notch_hz = None

    return PreprocessSpec(
        bandpass_low=float(band[0]),
        bandpass_high=float(band[1]),
        notch_hz=notch_hz,
        used_fallback=False,
    )


def load_experiment_id(release_dir: str | Path) -> str | None:
    """Return ``training.experiment_id`` from the release manifest, if present."""
    manifest_path = Path(release_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    experiment_id = manifest.get("training", {}).get("experiment_id")
    return str(experiment_id) if experiment_id else None
