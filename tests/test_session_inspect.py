"""Read-only inspection of session and dataset folders (Process page)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from nucleuskit_toolkit.ui.session_inspect import (
    format_duration,
    inspect_dataset,
    inspect_input,
    inspect_session,
)

T0 = 1_760_000_000.0  # Unix epoch seconds


def _write_eeg(path: str, seconds: float, fs: float = 250.0, header: bool = False) -> None:
    t = T0 + np.arange(0, seconds, 1 / fs)
    data = np.column_stack([t, np.zeros((t.size, 8))])
    with open(path, "w", encoding="utf-8") as fh:
        if header:
            fh.write("timestamp,AF8,AF7,CHEEK_R,CHEEK_L,EAR_R,AFz,BROW_L,NOSE\n")
        np.savetxt(fh, data, delimiter=",", fmt="%.4f")


def _write_shimmer_ms(path: str, seconds: float, fs: float = 51.2) -> None:
    t = (T0 + np.arange(0, seconds, 1 / fs)) * 1000.0
    np.savetxt(path, np.column_stack([t, np.ones((t.size, 6))]), delimiter=",", fmt="%.3f")


def _session(root: str, name: str, *, eeg: float | None = 120.0, shimmer: float | None = 120.0,
             video: bool = True, raw: bool = True) -> str:
    folder = os.path.join(root, name)
    target = os.path.join(folder, "rawData") if raw else folder
    os.makedirs(target)
    if eeg:
        _write_eeg(os.path.join(target, "eeg.csv"), eeg)
    if shimmer:
        _write_shimmer_ms(os.path.join(target, "shimmer.csv"), shimmer)
    if video:
        with open(os.path.join(target, "video.mp4"), "wb") as fh:
            fh.write(b"\x00")
    return folder


def test_session_summary(tmp_path) -> None:
    folder = _session(str(tmp_path), "P07", eeg=23 * 60, shimmer=60)
    info = inspect_session(folder)
    assert info is not None
    assert info.eeg and info.eda and info.ppg and info.video
    assert info.duration_s == pytest.approx(23 * 60, abs=1)
    assert info.summary() == "Session P07, 23 min, EEG/fEMG/EDA/PPG + video"
    assert info.missing == []


def test_session_files_at_top_level_are_not_moved(tmp_path) -> None:
    folder = _session(str(tmp_path), "S1", raw=False)
    before = sorted(os.listdir(folder))
    info = inspect_session(folder)
    assert info is not None and info.eeg and info.video
    assert sorted(os.listdir(folder)) == before
    assert not os.path.exists(os.path.join(folder, "rawData"))


def test_duration_from_millisecond_timestamps_and_header(tmp_path) -> None:
    folder = _session(str(tmp_path), "S2", eeg=None, shimmer=90, video=False)
    info = inspect_session(folder)
    assert info is not None and not info.eeg
    assert info.duration_s == pytest.approx(90, abs=0.1)
    assert info.missing == ["EEG/fEMG", "video"]

    _write_eeg(os.path.join(folder, "rawData", "eeg.csv"), 45, header=True)
    assert inspect_session(folder).duration_s == pytest.approx(90, abs=0.1)


def test_dataset_counts_incomplete_sessions(tmp_path) -> None:
    root = str(tmp_path)
    _session(root, "A")
    _session(root, "B", video=False)
    _session(root, "C", shimmer=None, video=False)
    os.makedirs(os.path.join(root, "notes"))  # no recording: not a session
    info = inspect_dataset(root)
    assert info is not None
    assert [s.name for s in info.sessions] == ["A", "B", "C"]
    assert info.summary() == "3 sessions found, 2 incomplete (missing video, EDA/PPG)"


def test_inspect_input_errors(tmp_path) -> None:
    root = str(tmp_path)
    session = _session(root, "only")
    assert inspect_input("", dataset=False) == (None, None)
    assert inspect_input(os.path.join(root, "missing"), dataset=False)[1] == "This folder does not exist."
    result, error = inspect_input(session, dataset=True)
    assert result is None and "single session" in error
    result, error = inspect_input(root, dataset=False)
    assert result is None and "several sessions" in error
    empty = os.path.join(root, "empty")
    os.makedirs(empty)
    assert "No recording found" in inspect_input(empty, dataset=False)[1]


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(None, ""), (0, ""), (42, "42 s"), (23 * 60 + 10, "23 min"), (3 * 3600 + 5 * 60, "3 h 05 min")],
)
def test_format_duration(seconds, text) -> None:
    assert format_duration(seconds) == text
