"""Tests for headless offline GUI helpers (no Qt)."""

from __future__ import annotations

import os

from nucleuskit_pipeline.ui.offline_job import session_preflight


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
