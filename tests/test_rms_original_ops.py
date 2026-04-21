"""Tests for ``rms_original_ops`` (snapshot, log, revert)."""

from __future__ import annotations

from pathlib import Path

import pytest
from nucleuskit_pipeline.hermes.processor.rms_original_ops import (
    append_operation,
    ensure_baseline_snapshot,
    operations_log_path,
    original_rms_csv,
    revert_working_to_original,
    working_rms_csv,
)


def _write_minimal_rms(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Timestamp,AF8,AF7,CHEEK_R,CHEEK_L,EAR_R,AFz,BROW_L,NOSE\n"
        "0.0,1,2,3,4,5,6,7,8\n",
        encoding="utf-8",
    )


def test_ensure_baseline_snapshot_creates_original(tmp_path: Path) -> None:
    session = tmp_path / "sess"
    work = working_rms_csv(session)
    _write_minimal_rms(work)
    assert ensure_baseline_snapshot(session) is True
    orig = original_rms_csv(session)
    assert orig.is_file()
    assert orig.read_text(encoding="utf-8") == work.read_text(encoding="utf-8")
    assert ensure_baseline_snapshot(session) is False


def test_ensure_baseline_snapshot_no_working_file(tmp_path: Path) -> None:
    session = tmp_path / "sess2"
    assert ensure_baseline_snapshot(session) is False


def test_append_operation_and_revert(tmp_path: Path) -> None:
    session = tmp_path / "sess3"
    work = working_rms_csv(session)
    _write_minimal_rms(work)
    assert ensure_baseline_snapshot(session) is True

    append_operation(session, "test_event foo=1")
    log = operations_log_path(session)
    assert log.is_file()
    text = log.read_text(encoding="utf-8")
    assert "test_event foo=1" in text

    work.write_text(
        "Timestamp,AF8,AF7,CHEEK_R,CHEEK_L,EAR_R,AFz,BROW_L,NOSE\n"
        "1.0,9,9,9,9,9,9,9,9\n",
        encoding="utf-8",
    )

    revert_working_to_original(session)
    assert "0.0,1,2,3,4,5,6,7,8" in work.read_text(encoding="utf-8")
    tail = log.read_text(encoding="utf-8")
    assert "revert_to_original" in tail


def test_revert_missing_original_raises(tmp_path: Path) -> None:
    session = tmp_path / "sess4"
    work = working_rms_csv(session)
    _write_minimal_rms(work)
    with pytest.raises(FileNotFoundError):
        revert_working_to_original(session)
