"""Canonical RMS channel names and flexible rename from legacy headers."""

from __future__ import annotations

import pandas as pd

from nucleuskit_toolkit.hermes.constants import (
    CHANNEL_NAMES,
    OUTPUT_RMS_COLUMNS,
    hardware_channel_index,
)

# Backward-compatible alias used by UI and channel-fixer imports.
CANONICAL_CHANNEL_NAMES = CHANNEL_NAMES

_CANONICAL_SET = set(CHANNEL_NAMES)


def _canonical_name(column: str) -> str | None:
    idx = hardware_channel_index(column)
    if idx is None:
        return None
    return CHANNEL_NAMES[idx]


def as_output_rms_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a canonical RMS frame to the names written on disk.

    Input columns are ``Timestamp`` plus electrode names (``AF8`` …).
    Output columns are ``Timestamp`` plus ``HEAD_R_RMS`` ….
    """
    rename = dict(zip(CHANNEL_NAMES, OUTPUT_RMS_COLUMNS))
    out = df.rename(columns=rename)
    return out[["Timestamp", *OUTPUT_RMS_COLUMNS]]


def _rename_channels(channel_cols: list[str]) -> dict[str, str]:
    seen_canon: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    for c in channel_cols:
        canon = _canonical_name(c)
        if canon is None:
            raise ValueError(
                f"Unknown RMS channel column {c!r}. "
                f"Use ch1–ch8, legacy names (e.g. HEAD_R), electrode names "
                f"{list(CHANNEL_NAMES)}, or output names {list(OUTPUT_RMS_COLUMNS)}."
            )
        if canon in seen_canon:
            raise ValueError(
                f"Ambiguous columns {seen_canon[canon]!r} and {c!r} both map to {canon!r}"
            )
        seen_canon[canon] = c
        rename_map[c] = canon
    return rename_map


def normalize_rms_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session-style CSV: Timestamp + 8 channels (legacy and/or canonical names).

    Timestamp may be named ``Timestamp`` (case variants) or the first column is used as time.
    Drops stray ``Unnamed:*`` columns from bad CSV exports.
    """
    out = df.copy()

    unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
    if unnamed:
        out = out.drop(columns=unnamed, errors="ignore")

    ts_src = _resolve_timestamp_column(out)
    out = out.rename(columns={ts_src: "Timestamp"})

    channel_cols = [c for c in out.columns if c != "Timestamp"]
    if len(channel_cols) != len(CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(CHANNEL_NAMES)} channel columns, got {len(channel_cols)}: "
            f"{channel_cols!r}"
        )

    out = out.rename(columns=_rename_channels(channel_cols))
    present = set(out.columns) - {"Timestamp"}
    if present != _CANONICAL_SET:
        raise ValueError(
            f"After rename, expected exactly {sorted(_CANONICAL_SET)}, got {sorted(present)}"
        )

    return out[["Timestamp"] + list(CHANNEL_NAMES)]


def normalize_rms_channels_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Training-matrix CSV: only channel columns (no Timestamp), legacy and/or canonical.

    Strips ``Unnamed:*`` index columns and optional ``Timestamp`` if present.
    Output columns are canonical and in fixed order.
    """
    out = df.copy()
    unnamed = [c for c in out.columns if str(c).startswith("Unnamed")]
    if unnamed:
        out = out.drop(columns=unnamed, errors="ignore")
    if "Timestamp" in out.columns:
        out = out.drop(columns=["Timestamp"])

    channel_cols = [c for c in out.columns]
    if len(channel_cols) != len(CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(CHANNEL_NAMES)} channel columns, got {len(channel_cols)}: "
            f"{channel_cols!r}"
        )

    out = out.rename(columns=_rename_channels(channel_cols))
    present = set(out.columns)
    if present != _CANONICAL_SET:
        raise ValueError(
            f"After rename, expected exactly {sorted(_CANONICAL_SET)}, got {sorted(present)}"
        )

    return out[list(CHANNEL_NAMES)]


def _resolve_timestamp_column(df: pd.DataFrame) -> str:
    for name in ("Timestamp", "timestamp", "TIME", "Time", "time"):
        if name in df.columns:
            return name
    return str(df.columns[0])
