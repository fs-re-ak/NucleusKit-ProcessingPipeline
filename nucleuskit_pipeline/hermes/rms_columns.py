"""Canonical RMS channel names and flexible rename from legacy headers."""

from __future__ import annotations

import pandas as pd

# Fixed order used by models (index 0..7).
CANONICAL_CHANNEL_NAMES: tuple[str, ...] = (
    "AF8",
    "AF7",
    "CHEEK_R",
    "CHEEK_L",
    "EAR_R",
    "AFz",
    "BROW_L",
    "NOSE",
)

_CANONICAL_SET = set(CANONICAL_CHANNEL_NAMES)

# Legacy labels -> canonical (keys match typical CSV headers).
_ALIAS_TO_CANONICAL: dict[str, str] = {
    "HEAD_R": "AF8",
    "HEAD_L": "AF7",
    "FOREHEAD_L": "AFz",
    "BROW": "BROW_L",
    **{n: n for n in CANONICAL_CHANNEL_NAMES},
}


def _rename_channels(channel_cols: list[str]) -> dict[str, str]:
    seen_canon: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    for c in channel_cols:
        if c not in _ALIAS_TO_CANONICAL:
            raise ValueError(
                f"Unknown RMS channel column {c!r}. "
                f"Use legacy names (e.g. HEAD_R) or canonical names {list(CANONICAL_CHANNEL_NAMES)}."
            )
        canon = _ALIAS_TO_CANONICAL[c]
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
    if len(channel_cols) != len(CANONICAL_CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(CANONICAL_CHANNEL_NAMES)} channel columns, got {len(channel_cols)}: "
            f"{channel_cols!r}"
        )

    out = out.rename(columns=_rename_channels(channel_cols))
    present = set(out.columns) - {"Timestamp"}
    if present != _CANONICAL_SET:
        raise ValueError(
            f"After rename, expected exactly {sorted(_CANONICAL_SET)}, got {sorted(present)}"
        )

    return out[["Timestamp"] + list(CANONICAL_CHANNEL_NAMES)]


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
    if len(channel_cols) != len(CANONICAL_CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(CANONICAL_CHANNEL_NAMES)} channel columns, got {len(channel_cols)}: "
            f"{channel_cols!r}"
        )

    out = out.rename(columns=_rename_channels(channel_cols))
    present = set(out.columns)
    if present != _CANONICAL_SET:
        raise ValueError(
            f"After rename, expected exactly {sorted(_CANONICAL_SET)}, got {sorted(present)}"
        )

    return out[list(CANONICAL_CHANNEL_NAMES)]


def _resolve_timestamp_column(df: pd.DataFrame) -> str:
    for name in ("Timestamp", "timestamp", "TIME", "Time", "time"):
        if name in df.columns:
            return name
    return str(df.columns[0])
