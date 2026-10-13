"""Preprocessing contract consumed by :class:`StreamingWindowSource`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessSpec:
    """Filter settings a release expects on already-windowed EXG.

    ``used_fallback`` is True when the release ``config.json`` did not declare
    ``bandpass_hz``, so the 2.5.0 contract (15–40 Hz, no notch) is applied.
    """

    bandpass_low: float = 15.0
    bandpass_high: float = 40.0
    notch_hz: float | None = None
    used_fallback: bool = False


DEFAULT_PREPROCESS_SPEC = PreprocessSpec()
