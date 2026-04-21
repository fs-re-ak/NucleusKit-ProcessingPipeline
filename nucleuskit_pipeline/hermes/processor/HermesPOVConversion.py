"""
Hermes POV Movie Conversion

Converts POV camera recordings from MKV to MP4 format using ffmpeg.
Standalone: requires explicit data_root, ffmpeg directory, and screen id — no hardcoded paths.

Author(s):
    Fred Simard (fs@re-ak.com), ©RE-AK Technologies Inc.
    Winter 2026
"""

from __future__ import annotations

import os
import subprocess
import sys

from nucleuskit_pipeline.logging_utils import printInfo, printWarning


def _ffmpeg_executable(ffmpeg_bin_dir: str) -> str:
    name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    return os.path.join(ffmpeg_bin_dir, name)


def convertPOVMovieClip(screen_id: str | None, data_root: str | None, ffmpeg_bin_dir: str | None) -> None:
    """
    Convert POV camera recording from UUID format to MKV/MP4.

    Args:
        screen_id: File base name (UUID) without extension, under data_root.
        data_root: Directory containing the UUID-named POV file.
        ffmpeg_bin_dir: Directory containing the ffmpeg executable.
    """
    if not screen_id or not data_root or not ffmpeg_bin_dir:
        printWarning("[HermesPOVConversion] Missing screen_id, data_root, or ffmpeg_bin_dir; skipping")
        return

    ffmpeg_exe = _ffmpeg_executable(ffmpeg_bin_dir)
    if not os.path.isfile(ffmpeg_exe):
        printWarning(f"[HermesPOVConversion] ffmpeg not found at {ffmpeg_exe}; skipping")
        return

    printInfo(f"[HermesPOVConversion] Converting POV movie: {screen_id}")

    if not os.path.isdir(data_root):
        printWarning("[HermesPOVConversion] Skipping POV conversion, data_root not found")
        return

    uuid_full = os.path.join(data_root, screen_id)
    if not os.path.exists(uuid_full):
        printWarning(f"[HermesPOVConversion] Skipping POV conversion, file not found: {uuid_full}")
        return

    mkv_full = os.path.join(data_root, screen_id + ".mkv")
    mp4_full = os.path.join(data_root, screen_id + ".mp4")

    if os.path.exists(mkv_full):
        printInfo("[HermesPOVConversion] Conversion already done, skipping")
        return

    os.rename(uuid_full, mkv_full)
    try:
        rc = subprocess.run(
            [ffmpeg_exe, "-i", mkv_full, "-c", "copy", mp4_full],
            check=False,
        ).returncode
        if rc != 0 or not os.path.isfile(mp4_full):
            printWarning("[HermesPOVConversion] ffmpeg failed or produced no mp4; restoring original name")
            if os.path.isfile(mkv_full):
                os.rename(mkv_full, uuid_full)
            return
        os.rename(mp4_full, uuid_full)
    except OSError as e:
        printWarning(f"[HermesPOVConversion] Conversion failed: {e}")
        if os.path.isfile(mkv_full) and not os.path.exists(uuid_full):
            os.rename(mkv_full, uuid_full)
        return

    if os.path.isfile(mkv_full):
        os.remove(mkv_full)

    printInfo("[HermesPOVConversion] POV conversion completed")
