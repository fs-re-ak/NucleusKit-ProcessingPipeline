"""Shared baseline snapshot and operations log under ``features/emotions/original/``."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


def emotions_dir(session_root: str | Path) -> Path:
    return Path(session_root).resolve() / "features" / "emotions"


def original_dir(session_root: str | Path) -> Path:
    return emotions_dir(session_root) / "original"


def working_rms_csv(session_root: str | Path) -> Path:
    return emotions_dir(session_root) / "rmsSignals.csv"


def original_rms_csv(session_root: str | Path) -> Path:
    return original_dir(session_root) / "rmsSignals.csv"


def operations_log_path(session_root: str | Path) -> Path:
    return original_dir(session_root) / "operations.log"


def ensure_baseline_snapshot(session_root: str | Path) -> bool:
    """
    If ``original/rmsSignals.csv`` is missing and the working ``rmsSignals.csv``
    exists, copy working → original. Does not overwrite an existing original.

    Returns True if a new snapshot file was created, False otherwise.
    """
    session_root = Path(session_root).resolve()
    work = working_rms_csv(session_root)
    orig = original_rms_csv(session_root)
    if orig.is_file():
        return False
    if not work.is_file():
        return False
    original_dir(session_root).mkdir(parents=True, exist_ok=True)
    shutil.copy2(work, orig)
    return True


def append_operation(session_root: str | Path, message: str) -> None:
    """Append one UTF-8 log line with an ISO-8601 UTC timestamp prefix."""
    session_root = Path(session_root).resolve()
    log_path = operations_log_path(session_root)
    original_dir(session_root).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} {message.strip()}\n"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line)


def revert_working_to_original(session_root: str | Path) -> None:
    """Replace working ``rmsSignals.csv`` with ``original/rmsSignals.csv`` and log."""
    session_root = Path(session_root).resolve()
    orig = original_rms_csv(session_root)
    work = working_rms_csv(session_root)
    if not orig.is_file():
        raise FileNotFoundError(f"Missing frozen baseline: {orig}")
    emotions_dir(session_root).mkdir(parents=True, exist_ok=True)
    shutil.copy2(orig, work)
    append_operation(session_root, "revert_to_original")
