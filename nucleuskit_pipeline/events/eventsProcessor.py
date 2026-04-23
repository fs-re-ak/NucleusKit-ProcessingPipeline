"""
Events Processor

Seeds the playback_annotations.json file from the session's events.csv.

Each recorded event becomes a "point" annotation whose time value is its
offset in seconds from RECORDING_ONSET.  The file is written once: if it
already exists this step is skipped entirely, preserving any manual edits.

Author(s):
    RE-AK Technologies Inc.
    Spring 2026
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from pathlib import Path

from nucleuskit_pipeline.logging_utils import printError, printInfo, printWarning

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_EVENTS_CSV = "event.csv"
_OUTPUT_FILENAME = "playback_annotations.json"
_OUTPUT_SUBDIR = os.path.join("features", "events")

_ANNOTATIONS_SCHEMA_VERSION = 1

_IGNORED_EVENT_TYPES = frozenset({"RECORDING_ONSET", "RECIPE_CONFIG", "SYSTEM_CONFIG", "CONSENT_INFO"})
_RECORDING_ONSET = "RECORDING_ONSET"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _load_events_csv(path: str) -> list[tuple[float, str]] | None:
    """
    Read the first two columns (timestamp, event_name) from events.csv.

    Returns a list of (timestamp_seconds, event_name) tuples, or None when
    the file is missing or completely empty.
    """
    if not os.path.isfile(path):
        return None

    rows: list[tuple[float, str]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for raw_row in reader:
            if len(raw_row) < 2:
                continue
            ts_str = raw_row[0].strip()
            name = raw_row[1].strip()
            try:
                ts = float(ts_str)
            except ValueError:
                continue
            rows.append((ts, name))

    return rows if rows else None


def _build_annotations(rows: list[tuple[float, str]]) -> dict | None:
    """
    Convert raw event rows into the playback_annotations JSON dict.

    Returns None when RECORDING_ONSET is not found (cannot establish t=0).
    """
    onset_ts: float | None = None
    for ts, name in rows:
        if name == _RECORDING_ONSET:
            onset_ts = ts
            break

    if onset_ts is None:
        return None

    points: list[dict] = []
    for ts, name in rows:
        if name in _IGNORED_EVENT_TYPES:
            continue
        offset_s = ts - onset_ts
        points.append(
            {
                "id": str(uuid.uuid4()),
                "t": round(offset_s, 6),
                "label": name,
                "visible": True,
            }
        )

    return {
        "version": _ANNOTATIONS_SCHEMA_VERSION,
        "points": points,
        "zones": [],
    }


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def seedPlaybackAnnotations(recPath: str) -> None:
    """
    Seed features/events/playback_annotations.json from rawData/events.csv.

    If the output file already exists this function returns immediately so
    that manually-edited annotations are never overwritten.

    Args:
        recPath: Root directory of the recording session.
    """
    printInfo("[eventsProcessor] Seeding playback annotations")

    output_dir = Path(recPath) / _OUTPUT_SUBDIR
    output_path = output_dir / _OUTPUT_FILENAME

    if output_path.is_file():
        printInfo(f"[eventsProcessor] {_OUTPUT_FILENAME} already exists — skipping")
        return

    events_csv = os.path.join(recPath, "rawData", _EVENTS_CSV)
    rows = _load_events_csv(events_csv)

    if rows is None:
        printWarning(f"[eventsProcessor] {_EVENTS_CSV} not found or empty — skipping")
        return

    annotations = _build_annotations(rows)

    if annotations is None:
        printWarning(
            f"[eventsProcessor] RECORDING_ONSET not found in {_EVENTS_CSV} — "
            "cannot establish time zero, skipping"
        )
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(annotations, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        printError(f"[eventsProcessor] Could not write {_OUTPUT_FILENAME}: {exc}")
        return

    printInfo(
        f"[eventsProcessor] Wrote {len(annotations['points'])} annotation(s) "
        f"to {output_path}"
    )
