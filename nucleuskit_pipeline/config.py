"""
Optional POV / ffmpeg settings for the Nucleus-Kit processing pipeline.

Precedence: merge JSON from the working directory (legacy filename, then canonical
filename), then ``extra_json_path``, then environment variables override each field.

Environment variables:
  HERMES_POV_DATA_ROOT — directory containing the UUID-named POV file
  HERMES_FFMPEG_DIR    — directory containing the ffmpeg executable
  HERMES_POV_SCREEN_ID — optional default screen / file id (no extension)

JSON keys (same semantics): pov_data_root, ffmpeg_dir, screen_id
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

DEFAULT_JSON_NAME = "nucleuskit_pipeline_config.json"
LEGACY_JSON_NAME = "hermes_standalone_config.json"


@dataclass
class PovSettings:
    data_root: Optional[str] = None
    ffmpeg_bin_dir: Optional[str] = None
    screen_id: Optional[str] = None


def _strip_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _merge_pov_dict(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if v is not None:
            out[k] = v
    return out


def resolve_pov_settings(extra_json_path: Optional[str] = None) -> PovSettings:
    """
    Merge optional JSON config from cwd (legacy then canonical), then extra_json_path,
    then apply env overrides.
    """
    merged: dict[str, Any] = {}
    legacy = os.path.join(os.getcwd(), LEGACY_JSON_NAME)
    if os.path.isfile(legacy):
        merged = _merge_pov_dict(merged, _load_json(legacy))
    cwd_json = os.path.join(os.getcwd(), DEFAULT_JSON_NAME)
    if os.path.isfile(cwd_json):
        merged = _merge_pov_dict(merged, _load_json(cwd_json))
    if extra_json_path and os.path.isfile(extra_json_path):
        merged = _merge_pov_dict(merged, _load_json(extra_json_path))

    data_root = _strip_str(merged.get("pov_data_root"))
    ffmpeg_dir = _strip_str(merged.get("ffmpeg_dir"))
    screen_id = _strip_str(merged.get("screen_id"))

    if _strip_str(os.environ.get("HERMES_POV_DATA_ROOT")):
        data_root = _strip_str(os.environ.get("HERMES_POV_DATA_ROOT"))
    if _strip_str(os.environ.get("HERMES_FFMPEG_DIR")):
        ffmpeg_dir = _strip_str(os.environ.get("HERMES_FFMPEG_DIR"))
    if _strip_str(os.environ.get("HERMES_POV_SCREEN_ID")):
        screen_id = _strip_str(os.environ.get("HERMES_POV_SCREEN_ID"))

    return PovSettings(data_root=data_root, ffmpeg_bin_dir=ffmpeg_dir, screen_id=screen_id)
