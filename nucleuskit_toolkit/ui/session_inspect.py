"""Read-only inspection of a session or dataset folder for the Process page (no Qt imports).

The preflight functions of ``offline_job`` prepare the session folder (``ensure_rawdata_layout`` moves loose
recording files into ``rawData/``), so they only run when processing starts. The inspection here only reads: it looks
for the raw files in ``rawData/`` or, before the first run, at the top of the session folder, and reads the first and
last rows of the recordings to estimate the duration without loading them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from reak_core.constants import EDA_FILES, EXG_FILENAMES, PPG_FILES
from reak_core.io.session import RESERVED_DIRS
from reak_core.io.timestamps import normalise_timestamps_to_seconds

VIDEO_FILENAME = "video.mp4"
_HEAD_ROWS = 64
_TAIL_BYTES = 8192
_EDA_NAMES = tuple(name for name, _ in EDA_FILES)
_PPG_NAMES = tuple(name for name, _ in PPG_FILES)


@dataclass(frozen=True)
class SessionInfo:
    path: str
    name: str
    duration_s: float | None
    eeg: bool
    eda: bool
    ppg: bool
    video: bool

    @property
    def missing(self) -> list[str]:
        """Parts of a complete Nucleus-Kit recording that are absent."""
        out = []
        if not self.eeg:
            out.append("EEG/fEMG")
        if not (self.eda or self.ppg):
            out.append("EDA/PPG")
        if not self.video:
            out.append("video")
        return out

    def modalities_text(self) -> str:
        parts = []
        if self.eeg:
            parts.append("EEG/fEMG")
        if self.eda and self.ppg:
            parts.append("EDA/PPG")
        elif self.eda:
            parts.append("EDA")
        elif self.ppg:
            parts.append("PPG")
        text = "/".join(parts) if parts else "no signals"
        return text + (" + video" if self.video else "")

    def summary(self) -> str:
        """For example ``Session P07, 23 min, EEG/fEMG/EDA/PPG + video``."""
        parts = (f"Session {self.name}", format_duration(self.duration_s), self.modalities_text())
        return ", ".join(p for p in parts if p)


@dataclass(frozen=True)
class DatasetInfo:
    path: str
    sessions: tuple[SessionInfo, ...]

    @property
    def incomplete(self) -> list[SessionInfo]:
        return [s for s in self.sessions if s.missing]

    def summary(self) -> str:
        """For example ``14 sessions found, 2 incomplete (missing video)``."""
        n = len(self.sessions)
        text = f"{n} session{'s' if n != 1 else ''} found"
        incomplete = self.incomplete
        if incomplete:
            reasons: list[str] = []
            for session in incomplete:
                for part in session.missing:
                    if part not in reasons:
                        reasons.append(part)
            text += f", {len(incomplete)} incomplete (missing {', '.join(reasons)})"
        return text


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return ""
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def _raw_dir(folder: str) -> str:
    raw = os.path.join(folder, "rawData")
    return raw if os.path.isdir(raw) else folder


def _first_file(directory: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    return None


def _first_value(line: str) -> float | None:
    try:
        return float(line.split(",", 1)[0].strip().strip("'\""))
    except ValueError:
        return None


def file_duration_s(path: str) -> float | None:
    """Duration of a recording from its first rows and its last row, without loading the whole file."""
    try:
        head: list[float] = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                value = _first_value(line)
                if value is not None:
                    head.append(value)
                if len(head) >= _HEAD_ROWS:
                    break
        if len(head) < 2:
            return None
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - _TAIL_BYTES))
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
        if size > _TAIL_BYTES:
            lines = lines[1:]  # the first line read may be cut
        last = None
        for line in reversed(lines):
            last = _first_value(line)
            if last is not None:
                break
        if last is None:
            return None
        seconds = normalise_timestamps_to_seconds(np.asarray(head + [last], dtype=float))
        duration = float(seconds[-1])
        return duration if np.isfinite(duration) and duration > 0 else None
    except OSError:
        return None


def inspect_session(folder: str, *, with_duration: bool = True) -> SessionInfo | None:
    """What a session folder holds, or ``None`` when it has no Hermes or Shimmer recording. Moves nothing."""
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return None
    raw = _raw_dir(folder)
    exg = _first_file(raw, EXG_FILENAMES)
    eda = _first_file(raw, _EDA_NAMES)
    ppg = _first_file(raw, _PPG_NAMES)
    if not (exg or eda or ppg):
        return None
    duration = None
    if with_duration:
        durations = [d for d in (file_duration_s(p) for p in dict.fromkeys(p for p in (exg, eda, ppg) if p)) if d]
        duration = max(durations) if durations else None
    return SessionInfo(
        path=folder,
        name=os.path.basename(os.path.normpath(folder)),
        duration_s=duration,
        eeg=exg is not None,
        eda=eda is not None,
        ppg=ppg is not None,
        video=os.path.isfile(os.path.join(raw, VIDEO_FILENAME)),
    )


def inspect_dataset(folder: str) -> DatasetInfo | None:
    """Sessions found in the direct subfolders of ``folder``, or ``None`` when there is none. Moves nothing."""
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        return None
    sessions = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return None
    for entry in entries:
        candidate = os.path.join(folder, entry)
        if entry in RESERVED_DIRS or not os.path.isdir(candidate):
            continue
        info = inspect_session(candidate, with_duration=False)
        if info is not None:
            sessions.append(info)
    return DatasetInfo(folder, tuple(sessions)) if sessions else None


def inspect_input(text: str, dataset: bool) -> tuple[SessionInfo | DatasetInfo | None, str | None]:
    """``(result, error)`` for the Process page; ``(None, None)`` when no path is entered."""
    text = text.strip()
    if not text:
        return None, None
    path = os.path.abspath(os.path.expanduser(text))
    if not os.path.isdir(path):
        return None, "This folder does not exist."
    if dataset:
        info = inspect_dataset(path)
        if info is not None:
            return info, None
        if inspect_session(path, with_duration=False) is not None:
            return None, (
                "This folder is a single session: choose Single session, or select the folder holding the sessions."
            )
        return None, "No sessions found: expected subfolders containing Hermes or Shimmer recordings."
    session = inspect_session(path)
    if session is not None:
        return session, None
    if inspect_dataset(path) is not None:
        return None, "This folder holds several sessions: choose Dataset, or select one session folder."
    return None, (
        "No recording found: expected Hermes (EEG/fEMG) or Shimmer (EDA/PPG) files in the folder or its rawData/."
    )
