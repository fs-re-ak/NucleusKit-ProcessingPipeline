"""
Camera and video processing utilities.

Covers two operations that run as part of offline session processing:

1. ``convertPOVMovieClip``  — rename UUID POV file to MKV then re-mux to MP4.
2. ``ensure_session_video_rotated_180`` — rotate rawData/video.mp4 180° in place
   using ffmpeg (idempotent, guarded by a marker file).

Both functions are thin ffmpeg wrappers; the shared ``_ffmpeg_executable``
helper resolves the binary from either an explicit directory or the system PATH
(with an optional fallback through the pipeline config).

Author(s):
    RE-AK Technologies Inc.
    Winter – Spring 2026
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from nucleuskit_pipeline.config import resolve_pov_settings
from nucleuskit_pipeline.logging_utils import printInfo, printWarning


# ---------------------------------------------------------------------------
# Shared ffmpeg helper
# ---------------------------------------------------------------------------

def _ffmpeg_executable(ffmpeg_bin_dir: str | None = None) -> str | None:
    """
    Resolve the ffmpeg binary path.

    Resolution order:
    1. ``ffmpeg_bin_dir`` if provided explicitly.
    2. ``ffmpeg_bin_dir`` from the pipeline config (``nucleuskit_pipeline_config.json``).
    3. System PATH via ``shutil.which``.

    Returns the full path to the executable, or *None* if not found.
    """
    name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"

    if ffmpeg_bin_dir:
        candidate = os.path.join(ffmpeg_bin_dir, name)
        if os.path.isfile(candidate):
            return candidate

    pov = resolve_pov_settings(None)
    if pov and pov.ffmpeg_bin_dir:
        candidate = os.path.join(pov.ffmpeg_bin_dir, name)
        if os.path.isfile(candidate):
            return candidate

    return shutil.which("ffmpeg")


# ---------------------------------------------------------------------------
# POV clip conversion  (MKV → MP4)
# ---------------------------------------------------------------------------

def convertPOVMovieClip(
    screen_id: str | None,
    data_root: str | None,
    ffmpeg_bin_dir: str | None,
) -> None:
    """
    Convert a UUID-named POV camera recording from MKV to MP4.

    The source file lives at ``<data_root>/<screen_id>`` (no extension).
    It is first renamed to ``<screen_id>.mkv``, then re-muxed to
    ``<screen_id>.mp4`` via ``ffmpeg -c copy``.  On failure the original
    name is restored.

    Args:
        screen_id:      File base name (UUID, no extension) inside ``data_root``.
        data_root:      Directory containing the UUID-named POV file.
        ffmpeg_bin_dir: Directory containing the ffmpeg executable.
    """
    if not screen_id or not data_root or not ffmpeg_bin_dir:
        printWarning("[camera] Missing screen_id, data_root, or ffmpeg_bin_dir; skipping POV conversion")
        return

    ffmpeg = _ffmpeg_executable(ffmpeg_bin_dir)
    if not ffmpeg:
        printWarning(f"[camera] ffmpeg not found in {ffmpeg_bin_dir}; skipping POV conversion")
        return

    if not os.path.isdir(data_root):
        printWarning("[camera] data_root not found; skipping POV conversion")
        return

    uuid_full = os.path.join(data_root, screen_id)
    if not os.path.exists(uuid_full):
        printWarning(f"[camera] Source file not found: {uuid_full}; skipping POV conversion")
        return

    mkv_full = os.path.join(data_root, screen_id + ".mkv")
    mp4_full = os.path.join(data_root, screen_id + ".mp4")

    if os.path.exists(mkv_full):
        printInfo("[camera] POV conversion already done; skipping")
        return

    printInfo(f"[camera] Converting POV clip: {screen_id}")
    os.rename(uuid_full, mkv_full)
    try:
        rc = subprocess.run(
            [ffmpeg, "-i", mkv_full, "-c", "copy", mp4_full],
            check=False,
        ).returncode
        if rc != 0 or not os.path.isfile(mp4_full):
            printWarning("[camera] ffmpeg failed or produced no mp4; restoring original name")
            if os.path.isfile(mkv_full):
                os.rename(mkv_full, uuid_full)
            return
        os.rename(mp4_full, uuid_full)
    except OSError as e:
        printWarning(f"[camera] POV conversion failed: {e}")
        if os.path.isfile(mkv_full) and not os.path.exists(uuid_full):
            os.rename(mkv_full, uuid_full)
        return

    if os.path.isfile(mkv_full):
        os.remove(mkv_full)

    printInfo("[camera] POV conversion completed")


# ---------------------------------------------------------------------------
# Session video rotation  (rawData/video.mp4 → 180°)
# ---------------------------------------------------------------------------

# Written under <session>/rawData/ after a successful rotation (idempotent flag).
MARKER_FILENAME = "nucleuskit_video_rotated_180.json"

# Session video expected by the playback UI.
VIDEO_BASENAME = "video.mp4"

_MARKER_SCHEMA = "nucleuskit.video_rotation.v1"


def ensure_session_video_rotated_180(recpath: str) -> None:
    """
    Rotate ``rawData/video.mp4`` 180° in place using ffmpeg (idempotent).

    If ``rawData/nucleuskit_video_rotated_180.json`` already exists the
    function returns immediately, preserving any previous rotation and
    avoiding redundant re-encoding on repeated pipeline runs.

    If the video file or ffmpeg are absent, a warning is logged and the
    function returns without error.

    Args:
        recpath: Root directory of the recording session.
    """
    recpath = os.path.abspath(recpath)
    raw_data = os.path.join(recpath, "rawData")
    marker_path = os.path.join(raw_data, MARKER_FILENAME)
    video_path = os.path.join(raw_data, VIDEO_BASENAME)

    if os.path.isfile(marker_path):
        printInfo(f"[camera] Marker present ({MARKER_FILENAME}); skipping 180° rotation")
        return

    if not os.path.isdir(raw_data):
        printWarning(f"[camera] No rawData folder at {raw_data}; skipping")
        return

    if not os.path.isfile(video_path):
        printInfo(f"[camera] No {VIDEO_BASENAME} under rawData; nothing to rotate")
        return

    ffmpeg = _ffmpeg_executable()
    if not ffmpeg:
        printWarning(
            "[camera] ffmpeg not found (configure ffmpeg_dir in "
            "nucleuskit_pipeline_config.json or add it to PATH); skipping rotation"
        )
        return

    tmp_path = os.path.join(raw_data, ".video_rot180_tmp.mp4")
    printInfo(f"[camera] Rotating 180° in place: {video_path}")

    base_cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-y", "-i", video_path,
        "-vf", "hflip,vflip",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-movflags", "+faststart",
    ]

    def _run_with_audio() -> subprocess.CompletedProcess[str]:
        return subprocess.run(base_cmd + ["-c:a", "copy", tmp_path],
                              check=False, capture_output=True, text=True)

    def _run_video_only() -> subprocess.CompletedProcess[str]:
        return subprocess.run(base_cmd + ["-an", tmp_path],
                              check=False, capture_output=True, text=True)

    try:
        completed = _run_with_audio()
        if completed.returncode != 0:
            completed = _run_video_only()
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()
            printWarning(f"[camera] ffmpeg failed (exit {completed.returncode}): {err}")
            return
        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
            printWarning("[camera] ffmpeg produced no output; aborting")
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

        printInfo(f"[camera] Rotation complete; wrote marker {marker_path}")
    except OSError as e:
        printWarning(f"[camera] I/O error during rotation: {e}")
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
