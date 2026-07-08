"""Headless helpers for the offline pipeline UI (no Qt imports)."""

from __future__ import annotations

import io
import os
import queue

from nucleuskit_toolkit.session.layout import ensure_session_rawdata_layout


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


def dataset_preflight(folder: str) -> tuple[list[str], str | None]:
    """Return (valid_session_paths, error) for a dataset root folder.

    A valid session subfolder is any direct child directory that contains
    a rawData/ subdirectory (the same criterion used by session_preflight).
    The returned paths are sorted alphabetically.
    """
    if not folder:
        return [], "Please select a dataset root folder."
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return [], "Dataset path is not a directory."

    sessions: list[str] = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError as e:
        return [], f"Cannot read dataset folder: {e}"

    for entry in entries:
        candidate = os.path.join(folder, entry)
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "rawData")):
            sessions.append(candidate)

    if not sessions:
        return [], (
            "No session subfolders found. Expected direct child directories "
            "that each contain a rawData/ subfolder."
        )
    return sessions, None


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


def ppg_fixer_preflight(folder: str) -> str | None:
    """Return an error message if the folder cannot be used for the PPG fixer, else None."""
    if not folder:
        return "Please select a session folder."
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return "Session path is not a directory."
    resampled = os.path.join(folder, "features", "ppg", "ppg_resampled.csv")
    if not os.path.isfile(resampled):
        return (
            "Missing features/ppg/ppg_resampled.csv. Run offline processing for this session "
            "first so PPG features are produced."
        )
    return None


def eeg_regression_preflight(folder: str) -> str | None:
    """Return an error message if the folder cannot be used for EEG regression denoising, else None."""
    if not folder:
        return "Please select a session folder."
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return "Session path is not a directory."
    raw = os.path.join(folder, "rawData")
    if not os.path.isdir(raw):
        return (
            "No rawData subfolder found. Expected a Nucleus-Kit session directory "
            "with a raw EEG file (rawEEG_0.csv, eeg.csv, eeg.tmp, or eegRec_0.csv)."
        )
    eeg_filenames = ["rawEEG_0.csv", "eeg.tmp", "eeg.csv", "eegRec_0.csv"]
    if not any(os.path.isfile(os.path.join(raw, f)) for f in eeg_filenames):
        return (
            "No raw EEG file found in rawData/. "
            "Expected one of: " + ", ".join(eeg_filenames)
        )
    return None
