"""ModelRegistry — select and load EmotionModels by name, version, or alias.

The registry resolves model aliases (``production``, ``experimental``, etc.)
to concrete name:version pairs and delegates loading to
:func:`~exg_emotion.runtime.loader.load_from_dir`.

Alias file layout (``releases/aliases.json``)::

    {
        "production":   "classical-emotion:0.1.0",
        "experimental": "classical-emotion:0.2.0-dev"
    }

Usage
-----
>>> from exg_emotion.runtime.registry import ModelRegistry
>>> registry = ModelRegistry("releases/")
>>>
>>> # Load by explicit version
>>> model = registry.load("classical-emotion", "0.1.0")
>>>
>>> # Load by alias
>>> model = registry.load("production")
>>>
>>> # List everything
>>> registry.list_releases()    # [(name, version), ...]
>>> registry.list_aliases()     # {"production": "classical-emotion:0.1.0"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from exg_emotion.core.exceptions import ModelNotFoundError
from exg_emotion.core.interfaces import EmotionModel
from exg_emotion.runtime.loader import load_from_dir

_ALIASES_FILE = "aliases.json"


class ModelRegistry:
    """Discover and load versioned model releases.

    Parameters
    ----------
    releases_dir:
        Path to the ``releases/`` directory.
    """

    def __init__(self, releases_dir: str | Path) -> None:
        self.root = Path(releases_dir)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, name: str, version: Optional[str] = None) -> EmotionModel:
        """Load a model by name + version or by alias.

        Parameters
        ----------
        name:
            Model family name (e.g. ``"classical-emotion"``), or an alias
            string (e.g. ``"production"``).  When *version* is ``None`` the
            registry first checks the aliases file; if not found there it
            tries to load the latest version for the given name.
        version:
            Explicit semantic version.  When given, the alias lookup is
            skipped entirely.

        Returns
        -------
        EmotionModel
        """
        if version is None:
            resolved = self._resolve_alias(name)
            if resolved is not None:
                name, version = resolved
            else:
                version = self._latest_version(name)

        release_dir = self.root / name / version
        if not release_dir.exists():
            raise ModelNotFoundError(
                f"Release not found: {name}:{version} (looked in {self.root})"
            )

        return load_from_dir(release_dir)

    # ------------------------------------------------------------------
    # Alias management
    # ------------------------------------------------------------------

    def set_alias(self, alias: str, name: str, version: str) -> None:
        """Create or update an alias mapping.

        Parameters
        ----------
        alias:
            Alias string (e.g. ``"production"``).
        name:
            Model family name.
        version:
            Semantic version string.
        """
        # Verify the target exists before writing the alias
        release_dir = self.root / name / version
        if not release_dir.exists():
            raise ModelNotFoundError(
                f"Cannot alias to non-existent release {name}:{version}"
            )

        aliases = self._load_aliases()
        aliases[alias] = f"{name}:{version}"
        self._save_aliases(aliases)

    def remove_alias(self, alias: str) -> None:
        """Remove an alias."""
        aliases = self._load_aliases()
        if alias not in aliases:
            raise ModelNotFoundError(f"Alias '{alias}' does not exist")
        del aliases[alias]
        self._save_aliases(aliases)

    def list_aliases(self) -> dict[str, str]:
        """Return all currently defined aliases."""
        return self._load_aliases()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_releases(self) -> list[tuple[str, str]]:
        """Return all (name, version) pairs found under the releases dir."""
        results: list[tuple[str, str]] = []
        if not self.root.exists():
            return results
        for name_dir in sorted(self.root.iterdir()):
            if not name_dir.is_dir() or name_dir.name == _ALIASES_FILE:
                continue
            for ver_dir in sorted(name_dir.iterdir()):
                if ver_dir.is_dir() and (ver_dir / "manifest.json").exists():
                    results.append((name_dir.name, ver_dir.name))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_alias(self, alias: str) -> Optional[tuple[str, str]]:
        """Return (name, version) if *alias* is a known alias, else None."""
        aliases = self._load_aliases()
        target = aliases.get(alias)
        if target is None:
            return None
        if ":" not in target:
            raise ModelNotFoundError(
                f"Malformed alias '{alias}' → '{target}' (expected 'name:version')"
            )
        name, version = target.rsplit(":", 1)
        return name, version

    def _latest_version(self, name: str) -> str:
        """Return the lexicographically latest version under *name*."""
        name_dir = self.root / name
        if not name_dir.exists():
            raise ModelNotFoundError(
                f"No releases found for model '{name}' under {self.root}"
            )
        versions = sorted(
            [
                d.name
                for d in name_dir.iterdir()
                if d.is_dir() and (d / "manifest.json").exists()
            ]
        )
        if not versions:
            raise ModelNotFoundError(
                f"Model '{name}' exists but has no valid releases"
            )
        return versions[-1]

    def _aliases_path(self) -> Path:
        return self.root / _ALIASES_FILE

    def _load_aliases(self) -> dict[str, str]:
        path = self._aliases_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def _save_aliases(self, aliases: dict[str, str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._aliases_path().write_text(
            json.dumps(aliases, indent=2), encoding="utf-8"
        )
