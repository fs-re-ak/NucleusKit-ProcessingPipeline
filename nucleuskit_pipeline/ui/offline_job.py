"""Headless helpers for the offline pipeline UI (no Qt imports)."""

from __future__ import annotations

import io
import os
import queue

from nucleuskit_pipeline.hermes.processor.fileTools import ensure_session_rawdata_layout


class QueueTextWriter(io.TextIOBase):
    """Write stdout/stderr into a thread-safe queue as UTF-8 text."""

    encoding = "utf-8"

    def __init__(self, q: queue.SimpleQueue[str]):
        super().__init__()
        self._q = q

    def write(self, s: str) -> int:
        if s:
            self._q.put(s)
        return len(s) if isinstance(s, str) else 0

    def flush(self) -> None:
        return None


def session_preflight(folder: str) -> str | None:
    """Return an error message if the folder is unsuitable, else None."""
    if not folder:
        return "Please select a session folder."
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return "Session path is not a directory."
    ensure_session_rawdata_layout(folder)
    raw = os.path.join(folder, "rawData")
    if not os.path.isdir(raw):
        return (
            "No rawData subfolder found. Expected a Nucleus-Kit session directory "
            "with rawData (Hermes EEG/EMG and related files)."
        )
    if not os.listdir(raw):
        return (
            "No recording files found: rawData is empty after preparing the session folder."
        )
    return None


def rms_features_preflight(folder: str) -> str | None:
    """Return an error message if RMS feature CSV is missing, else None."""
    if not folder:
        return "Please select a session folder."
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return "Session path is not a directory."
    rms = os.path.join(folder, "features", "emotions", "rmsSignals.csv")
    if not os.path.isfile(rms):
        return (
            "Missing features/emotions/rmsSignals.csv. Run offline processing for this session first "
            "so emotions features (including RMS) are produced."
        )
    return None


def channel_fixer_preflight(folder: str) -> str | None:
    """Return an error message if the folder cannot be used for channel fixer, else None."""
    return rms_features_preflight(folder)
