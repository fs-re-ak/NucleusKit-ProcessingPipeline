"""Backend for the Recording Merger tool.

Merges up to 5 recordings into a single session folder containing
``results/`` (CSVs, time-shifted and null-padded) and ``rawData/``
(video concatenated with black-screen gaps).

FFmpeg and ffprobe must be available on PATH.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_RESULT_STEP = 0.5          # seconds per CSV sample (2 Hz)
_VIDEO_CODEC = "libx264"
_AUDIO_CODEC = "aac"
_AUDIO_SR    = 44100
_PIX_FMT     = "yuv420p"


# ── logging helper ─────────────────────────────────────────────────────────────

def _log(q: queue.SimpleQueue[str], msg: str) -> None:
    q.put(msg + "\n")


# ── dependency check ───────────────────────────────────────────────────────────

def _require_ffmpeg() -> None:
    for cmd in ("ffmpeg", "ffprobe"):
        if shutil.which(cmd) is None:
            raise FileNotFoundError(
                f"'{cmd}' not found on PATH. "
                "Install FFmpeg and make sure it is accessible from the command line "
                "before using the Recording Merger."
            )


# ── ffprobe helpers ────────────────────────────────────────────────────────────

def _ffprobe(*args: str) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _probe_duration(path: Path) -> float | None:
    """Return video duration in seconds, or None on failure."""
    try:
        out = _ffprobe(
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        )
        return float(out.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def _probe_video_info(path: Path) -> dict:
    """Return {width, height, fps, has_audio} for the given video file."""
    info: dict = {
        "width": 1920, "height": 1080,
        "fps": 30.0,   "has_audio": False,
    }
    try:
        out = _ffprobe(
            "-show_streams",
            "-show_entries",
            "stream=width,height,r_frame_rate,codec_type",
            "-of", "default=noprint_wrappers=1",
            str(path),
        )
        for line in out.splitlines():
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k == "width" and v not in ("", "N/A"):
                info["width"] = int(v)
            elif k == "height" and v not in ("", "N/A"):
                info["height"] = int(v)
            elif k == "r_frame_rate" and v not in ("", "N/A"):
                n, _, d = v.partition("/")
                info["fps"] = float(n) / float(d) if d and d != "0" else float(n)
            elif k == "codec_type" and v == "audio":
                info["has_audio"] = True
    except (subprocess.CalledProcessError, ValueError):
        pass
    return info


# ── duration resolution ────────────────────────────────────────────────────────

def _resolve_duration(folder: Path, log_q: queue.SimpleQueue[str]) -> float:
    """Return recording duration (s): ffprobe video first, fall back to CSV max timestamp."""
    video = folder / "rawData" / "video.mp4"
    if video.exists():
        dur = _probe_duration(video)
        if dur is not None:
            return dur
        _log(log_q, f"  [warn] ffprobe failed for {video.name}; falling back to CSVs.")

    results_dir = folder / "results"
    max_ts = 0.0
    found = False
    if results_dir.is_dir():
        for csv_path in results_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path, usecols=["Timestamp"])
                if not df.empty:
                    max_ts = max(max_ts, float(df["Timestamp"].max()))
                    found = True
            except Exception:
                pass
    if found:
        return max_ts + _RESULT_STEP

    _log(log_q, f"  [warn] Cannot determine duration for {folder.name!r}; defaulting to 0 s.")
    return 0.0


# ── CSV merging ────────────────────────────────────────────────────────────────

def _collect_csv_names(folders: list[Path]) -> list[str]:
    """Union of *.csv names found in any recording's results/ folder."""
    names: set[str] = set()
    for folder in folders:
        rd = folder / "results"
        if rd.is_dir():
            for p in rd.glob("*.csv"):
                names.add(p.name)
    return sorted(names)


def _merge_csvs(
    folders: list[Path],
    durations: list[float],
    shifts: list[float],
    offsets: list[float],
    output_results: Path,
    log_q: queue.SimpleQueue[str],
) -> None:
    csv_names = _collect_csv_names(folders)
    if not csv_names:
        _log(log_q, "[csv] No result CSVs found in any recording — skipping.")
        return

    for name in csv_names:
        _log(log_q, f"[csv] {name}")

        # Determine the union of data columns from all recordings that have this file
        data_cols: list[str] = []
        seen: set[str] = set()
        for folder in folders:
            p = folder / "results" / name
            if p.exists():
                try:
                    header = pd.read_csv(p, nrows=0)
                    for col in header.columns:
                        if col != "Timestamp" and col not in seen:
                            data_cols.append(col)
                            seen.add(col)
                except Exception:
                    pass

        if not data_cols:
            _log(log_q, f"  [warn] Could not determine columns — skipping {name}.")
            continue

        segments: list[pd.DataFrame] = []

        for i, folder in enumerate(folders):
            shift = shifts[i]
            dur   = durations[i]
            p     = folder / "results" / name

            loaded = False
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    for col in data_cols:
                        if col not in df.columns:
                            df[col] = np.nan
                    df = df[["Timestamp"] + data_cols].copy()
                    df["Timestamp"] += shift
                    segments.append(df)
                    loaded = True
                except Exception as exc:
                    _log(log_q, f"  [warn] Read error in recording {i + 1}: {exc}")

            if not loaded and dur > 0:
                ts_arr = np.arange(0.0, dur, _RESULT_STEP) + shift
                null_df = pd.DataFrame({"Timestamp": ts_arr})
                for col in data_cols:
                    null_df[col] = np.nan
                segments.append(null_df)

            # Null-padded gap before the next recording
            if i < len(folders) - 1 and offsets[i] > 0:
                gap_start = shift + dur
                gap_end   = shifts[i + 1]
                if gap_end > gap_start + _RESULT_STEP * 0.5:
                    ts_arr = np.arange(gap_start, gap_end, _RESULT_STEP)
                    null_df = pd.DataFrame({"Timestamp": ts_arr})
                    for col in data_cols:
                        null_df[col] = np.nan
                    segments.append(null_df)

        if not segments:
            continue

        merged = pd.concat(segments, ignore_index=True)
        merged = merged.sort_values("Timestamp").reset_index(drop=True)
        out_path = output_results / name
        merged.to_csv(out_path, index=False)
        _log(log_q, f"  → {len(merged)} rows  ({out_path.name})")


# ── video merging ──────────────────────────────────────────────────────────────

def _ffmpeg(*args: str) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, "ffmpeg", result.stdout, result.stderr
        )


def _encode_to_ref(
    src: Path,
    dst: Path,
    src_has_audio: bool,
    width: int,
    height: int,
    fps: float,
    ref_has_audio: bool,
    log_q: queue.SimpleQueue[str],
) -> None:
    """Re-encode a source video clip to the reference format."""
    _log(log_q, f"  encode  {src.name} → {dst.name}")
    base_args = [
        "-vf", f"scale={width}:{height}",
        "-r", str(fps), "-vsync", "cfr",
        "-c:v", _VIDEO_CODEC, "-pix_fmt", _PIX_FMT, "-preset", "fast",
    ]
    if ref_has_audio and src_has_audio:
        _ffmpeg("-i", str(src), *base_args,
                "-c:a", _AUDIO_CODEC, "-ar", str(_AUDIO_SR), str(dst))
    elif ref_has_audio and not src_has_audio:
        # Add a silent audio track alongside the video
        _ffmpeg(
            "-i", str(src),
            "-f", "lavfi", "-i", f"anullsrc=r={_AUDIO_SR}:cl=stereo",
            "-map", "0:v", "-map", "1:a",
            *base_args,
            "-c:a", _AUDIO_CODEC, "-ar", str(_AUDIO_SR),
            str(dst),
        )
    else:
        _ffmpeg("-i", str(src), *base_args, "-an", str(dst))


def _make_black_clip(
    dst: Path,
    duration: float,
    width: int,
    height: int,
    fps: float,
    has_audio: bool,
    log_q: queue.SimpleQueue[str],
) -> None:
    """Generate a black-screen clip of the given duration."""
    _log(log_q, f"  black   {duration:.1f} s → {dst.name}")
    lavfi_video = f"color=c=black:s={width}x{height}:r={fps}"
    base_args = [
        "-t", str(duration),
        "-c:v", _VIDEO_CODEC, "-pix_fmt", _PIX_FMT, "-preset", "fast",
    ]
    if has_audio:
        _ffmpeg(
            "-f", "lavfi", "-i", lavfi_video,
            "-f", "lavfi", "-i", f"anullsrc=r={_AUDIO_SR}:cl=stereo",
            "-map", "0:v", "-map", "1:a",
            *base_args,
            "-c:a", _AUDIO_CODEC, "-ar", str(_AUDIO_SR),
            str(dst),
        )
    else:
        _ffmpeg(
            "-f", "lavfi", "-i", lavfi_video,
            *base_args, "-an",
            str(dst),
        )


def _merge_videos(
    folders: list[Path],
    durations: list[float],
    offsets: list[float],
    output_rawdata: Path,
    log_q: queue.SimpleQueue[str],
) -> None:
    video_paths = [folder / "rawData" / "video.mp4" for folder in folders]
    has_videos  = [p.exists() for p in video_paths]

    if not any(has_videos):
        _log(log_q, "[video] No video files found in any recording — skipping.")
        return

    # Use recording 1's video as reference; fall back to first available if rec 1 has none
    ref_idx  = 0 if has_videos[0] else next(i for i, h in enumerate(has_videos) if h)
    ref_info = _probe_video_info(video_paths[ref_idx])
    W, H     = ref_info["width"], ref_info["height"]
    fps      = ref_info["fps"]
    has_audio = ref_info["has_audio"]
    _log(log_q, f"[video] Reference: {W}×{H} @ {fps:.3f} fps  audio={has_audio}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp        = Path(tmp_str)
        clip_paths: list[Path] = []

        for i in range(len(folders)):
            clip_dst = tmp / f"clip_{i:02d}.mp4"
            if has_videos[i]:
                src_info = _probe_video_info(video_paths[i])
                _encode_to_ref(
                    video_paths[i], clip_dst,
                    src_info["has_audio"],
                    W, H, fps, has_audio, log_q,
                )
            else:
                _make_black_clip(clip_dst, durations[i], W, H, fps, has_audio, log_q)
            clip_paths.append(clip_dst)

            if i < len(folders) - 1 and offsets[i] > 0:
                gap_dst = tmp / f"gap_{i:02d}.mp4"
                _make_black_clip(gap_dst, offsets[i], W, H, fps, has_audio, log_q)
                clip_paths.append(gap_dst)

        concat_txt = tmp / "concat.txt"
        concat_txt.write_text(
            "".join(f"file '{p}'\n" for p in clip_paths),
            encoding="utf-8",
        )

        out_video = output_rawdata / "video.mp4"
        _log(log_q, f"[video] Concatenating {len(clip_paths)} clips → video.mp4")
        _ffmpeg(
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(out_video),
        )
        _log(log_q, f"[video] Done → {out_video}")


# ── public API ─────────────────────────────────────────────────────────────────

def merge_recordings(
    folders: list[str],
    offsets: list[float],
    output_dir: str,
    log_q: queue.SimpleQueue[str],
    *,
    skip_video: bool = False,
) -> None:
    """Merge N recordings into a single output folder.

    Parameters
    ----------
    folders:
        Absolute paths to recording root folders (1–5 items).
    offsets:
        Gap in seconds between consecutive recordings.
        Must satisfy ``len(offsets) == len(folders) - 1``.
    output_dir:
        Destination root; created automatically if absent.
    log_q:
        Thread-safe queue for log messages displayed in the UI.
    skip_video:
        When True, skip the video merge step entirely.
    """
    if not skip_video:
        _require_ffmpeg()

    fpaths      = [Path(f) for f in folders]
    output      = Path(output_dir)
    out_results = output / "results"
    out_rawdata = output / "rawData"
    out_results.mkdir(parents=True, exist_ok=True)
    out_rawdata.mkdir(parents=True, exist_ok=True)

    _log(log_q, f"Output folder: {output}\n")

    # ── durations ──────────────────────────────────────────────────────────────
    _log(log_q, "=== Resolving durations ===")
    durations: list[float] = []
    for i, fp in enumerate(fpaths):
        dur = _resolve_duration(fp, log_q)
        durations.append(dur)
        _log(log_q, f"  [{i + 1}] {fp.name!r}:  {dur:.2f} s")
    _log(log_q, "")

    # Cumulative start time of each recording in merged time
    shifts: list[float] = [0.0]
    for i in range(len(fpaths) - 1):
        shifts.append(shifts[-1] + durations[i] + offsets[i])

    # ── CSVs ───────────────────────────────────────────────────────────────────
    _log(log_q, "=== Merging result CSVs ===")
    _merge_csvs(fpaths, durations, shifts, offsets, out_results, log_q)
    _log(log_q, "")

    # ── video ──────────────────────────────────────────────────────────────────
    _log(log_q, "=== Merging video ===")
    if skip_video:
        _log(log_q, "[video] Skipped by user.")
    else:
        _merge_videos(fpaths, durations, offsets, out_rawdata, log_q)
    _log(log_q, "")

    _log(log_q, "=== Complete ===")
    _log(log_q, f"Merged recording saved to:  {output}")
