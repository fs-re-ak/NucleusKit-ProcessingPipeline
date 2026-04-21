"""Tests for POV / JSON config resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nucleuskit_pipeline.config import (
    DEFAULT_JSON_NAME,
    LEGACY_JSON_NAME,
    resolve_pov_settings,
)


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_resolve_pov_env_only(isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_POV_DATA_ROOT", "/data")
    monkeypatch.setenv("HERMES_FFMPEG_DIR", "/ffmpeg")
    monkeypatch.setenv("HERMES_POV_SCREEN_ID", "abc")
    pov = resolve_pov_settings()
    assert pov.data_root == "/data"
    assert pov.ffmpeg_bin_dir == "/ffmpeg"
    assert pov.screen_id == "abc"


def test_legacy_then_default_json_merge(isolated_cwd: Path) -> None:
    (isolated_cwd / LEGACY_JSON_NAME).write_text(
        json.dumps({"pov_data_root": "/old", "ffmpeg_dir": "/f1", "screen_id": "s1"}),
        encoding="utf-8",
    )
    (isolated_cwd / DEFAULT_JSON_NAME).write_text(
        json.dumps({"pov_data_root": "/new"}),
        encoding="utf-8",
    )
    pov = resolve_pov_settings()
    assert pov.data_root == "/new"
    assert pov.ffmpeg_bin_dir == "/f1"
    assert pov.screen_id == "s1"


def test_extra_json_overrides_cwd(isolated_cwd: Path) -> None:
    (isolated_cwd / DEFAULT_JSON_NAME).write_text(
        json.dumps({"pov_data_root": "/cwd", "screen_id": "x"}),
        encoding="utf-8",
    )
    extra = isolated_cwd / "extra.json"
    extra.write_text(json.dumps({"pov_data_root": "/extra"}), encoding="utf-8")
    pov = resolve_pov_settings(str(extra))
    assert pov.data_root == "/extra"
    assert pov.screen_id == "x"


def test_env_overrides_json(isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (isolated_cwd / DEFAULT_JSON_NAME).write_text(
        json.dumps({"pov_data_root": "/j", "ffmpeg_dir": "/jffmpeg"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_POV_DATA_ROOT", "/env")
    pov = resolve_pov_settings()
    assert pov.data_root == "/env"
    assert pov.ffmpeg_bin_dir == "/jffmpeg"
