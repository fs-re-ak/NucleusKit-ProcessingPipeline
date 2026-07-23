"""Tests for headless offline GUI helpers (no Qt)."""

from __future__ import annotations

import os

from nucleuskit_toolkit.ui.offline_job import dataset_preflight, session_preflight


def test_session_preflight_empty() -> None:
    assert session_preflight("") is not None


def test_session_preflight_missing_rawdata(tmp_path) -> None:
    d = tmp_path / "sess"
    d.mkdir()
    err = session_preflight(str(d))
    assert err is not None
    assert "empty" in err.lower()


def test_session_preflight_valid(tmp_path) -> None:
    d = tmp_path / "sess"
    (d / "rawData").mkdir(parents=True)
    (d / "rawData" / "placeholder.csv").write_text("x", encoding="utf-8")
    assert session_preflight(str(d)) is None


def test_session_preflight_flat_session_moves_into_rawdata(tmp_path) -> None:
    d = tmp_path / "sess"
    d.mkdir()
    (d / "rawEEG_0.csv").write_text("col\n0\n", encoding="utf-8")
    assert session_preflight(str(d)) is None
    assert (d / "rawData" / "rawEEG_0.csv").is_file()
    assert not (d / "rawEEG_0.csv").exists()


def test_session_preflight_not_dir() -> None:
    assert session_preflight(os.path.join("no", "such", "path", "here")) is not None


def test_dataset_preflight_flat_sessions_auto_create_rawdata(tmp_path) -> None:
    """Flat session subfolders (no rawData/) should be auto-migrated and accepted."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    sess_a = dataset / "sess_a"
    sess_a.mkdir()
    (sess_a / "rawEEG_0.csv").write_text("col\n0\n", encoding="utf-8")
    sess_b = dataset / "sess_b"
    sess_b.mkdir()
    (sess_b / "eeg.csv").write_text("col\n1\n", encoding="utf-8")

    sessions, err = dataset_preflight(str(dataset))

    assert err is None
    assert len(sessions) == 2
    assert (sess_a / "rawData" / "rawEEG_0.csv").is_file()
    assert not (sess_a / "rawEEG_0.csv").exists()
    assert (sess_b / "rawData" / "eeg.csv").is_file()


def test_dataset_preflight_no_valid_sessions_returns_error(tmp_path) -> None:
    """A dataset folder with only empty subfolders should return an error."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "empty_sess").mkdir()

    sessions, err = dataset_preflight(str(dataset))

    assert sessions == []
    assert err is not None
