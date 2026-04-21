"""
One-time 180° rotation for the session offline video.

The canonical path is ``rawData/video.mp4`` (see ``ui/pages/playback_page.py``).
After a successful rotation, a marker file is written under ``rawData`` so
re-running offline processing does not rotate again.

Author(s):
    NucleusKit Processing Pipeline
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from nucleuskit_pipeline.config import resolve_pov_settings
from nucleuskit_pipeline.logging_utils import printInfo, printWarning

# Written under <session>/rawData/ after a successful rotation (idempotent flag).
MARKER_FILENAME = "nucleuskit_video_rotated_180.json"

# Session video expected by playback UI.
VIDEO_BASENAME = "video.mp4"

_MARKER_SCHEMA = "nucleuskit.video_rotation.v1"


def _ffmpeg_executable() -> str | None:
    pov = resolve_pov_settings(None)
    if pov and pov.ffmpeg_bin_dir:
        name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
        candidate = os.path.join(pov.ffmpeg_bin_dir, name)
        if os.path.isfile(candidate):
            return candidate
    import shutil

    return shutil.which("ffmpeg")


def ensure_session_video_rotated_180(recpath: str) -> None:
    """
    If ``rawData/nucleuskit_video_rotated_180.json`` exists, do nothing.

    Otherwise, if ``rawData/video.mp4`` exists, rotate it 180° in place (via a
    temp file) using ffmpeg, then write the marker JSON.

    If the video file is missing or ffmpeg is unavailable, log and return.
    """
    recpath = os.path.abspath(recpath)
    raw_data = os.path.join(recpath, "rawData")
    marker_path = os.path.join(raw_data, MARKER_FILENAME)
    video_path = os.path.join(raw_data, VIDEO_BASENAME)

    if os.path.isfile(marker_path):
        printInfo(
            f"[session_video_rotation] Marker present ({MARKER_FILENAME}); skipping 180° rotation"
        )
        return

    if not os.path.isdir(raw_data):
        printWarning(f"[session_video_rotation] No rawData folder at {raw_data}; skipping")
        return

    if not os.path.isfile(video_path):
        printInfo(
            f"[session_video_rotation] No {VIDEO_BASENAME} under rawData; nothing to rotate"
        )
        return

    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        printWarning(
            "[session_video_rotation] ffmpeg not found (configure ffmpeg_dir in "
            "nucleuskit_pipeline_config.json or PATH); skipping rotation"
        )
        return

    tmp_path = os.path.join(raw_data, ".video_rot180_tmp.mp4")
    printInfo(f"[session_video_rotation] Rotating 180° in place: {video_path}")

    base_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        "hflip,vflip",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
    ]

    def _run_with_audio_copy() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            base_cmd + ["-c:a", "copy", tmp_path],
            check=False,
            capture_output=True,
            text=True,
        )

    def _run_video_only() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            base_cmd + ["-an", tmp_path],
            check=False,
            capture_output=True,
            text=True,
        )

    try:
        completed = _run_with_audio_copy()
        if completed.returncode != 0:
            completed = _run_video_only()
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()
            printWarning(
                f"[session_video_rotation] ffmpeg failed (exit {completed.returncode}): {err}"
            )
            return
        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
            printWarning("[session_video_rotation] ffmpeg produced no output; aborting")
            return

        os.replace(tmp_path, video_path)

        payload = {
            "schema": _MARKER_SCHEMA,
            "rotation_degrees": 180,
            "media_file": VIDEO_BASENAME,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

        printInfo(f"[session_video_rotation] Rotation complete; wrote marker {marker_path}")
    except OSError as e:
        printWarning(f"[session_video_rotation] I/O error: {e}")
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
