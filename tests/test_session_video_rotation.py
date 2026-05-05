"""Tests for one-time rawData/video.mp4 rotation."""

from __future__ import annotations

import json

from nucleuskit_pipeline.camera.processing import (
    MARKER_FILENAME,
    VIDEO_BASENAME,
    ensure_session_video_rotated_180,
)


def test_skips_when_marker_present(tmp_path) -> None:
    sess = tmp_path / "session"
    raw = sess / "rawData"
    raw.mkdir(parents=True)
    marker = raw / MARKER_FILENAME
    marker.write_text("{}", encoding="utf-8")
    (raw / VIDEO_BASENAME).write_bytes(b"not-a-real-video")

    ensure_session_video_rotated_180(str(sess))

    assert marker.is_file()
    assert (raw / VIDEO_BASENAME).read_bytes() == b"not-a-real-video"


def test_skips_when_no_video(tmp_path) -> None:
    sess = tmp_path / "session"
    raw = sess / "rawData"
    raw.mkdir(parents=True)

    ensure_session_video_rotated_180(str(sess))

    assert not (raw / MARKER_FILENAME).is_file()
