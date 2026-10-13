"""Constants for the Hermes EEG/EMG headset hardware."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Hardware acquisition
# ---------------------------------------------------------------------------
SAMPLING_RATE: int = 250
"""EEG/EMG acquisition rate in Hz."""

DISCONNECT_VALUE: float = 187500.0
"""Raw ADC saturation value indicating an electrode disconnect."""

# ---------------------------------------------------------------------------
# Channel layout (canonical electrode names, index 0..7)
# ---------------------------------------------------------------------------
CHANNEL_NAMES: tuple[str, ...] = (
    "AF8",
    "AF7",
    "CHEEK_R",
    "CHEEK_L",
    "EAR_R",
    "AFz",
    "BROW_L",
    "NOSE",
)

# Names written to emotion feature files. Same hardware order as CHANNEL_NAMES
# and as raw columns ch1..ch8.
OUTPUT_RMS_COLUMNS: tuple[str, ...] = (
    "HEAD_R_RMS",
    "HEAD_L_RMS",
    "CHEEK_R_RMS",
    "CHEEK_L_RMS",
    "EAR_R_RMS",
    "FOREHEAD_L_RMS",
    "BROW_RMS",
    "NOSE_RMS",
)

CHANNELS: dict[str, int] = {name: i for i, name in enumerate(CHANNEL_NAMES)}
N_CHANNELS: int = len(CHANNEL_NAMES)

# Accepted raw / legacy headers → hardware index (ch1 = AF8 = HEAD_R_RMS).
_RAW_COLUMN_INDEX: dict[str, int] = {}
for _i, _electrode in enumerate(CHANNEL_NAMES):
    _RAW_COLUMN_INDEX[_electrode.lower()] = _i
    _RAW_COLUMN_INDEX[f"ch{_i + 1}"] = _i
    _RAW_COLUMN_INDEX[OUTPUT_RMS_COLUMNS[_i].lower()] = _i
_RAW_COLUMN_INDEX.update(
    {
        "head_r": 0,
        "head_l": 1,
        "forehead_l": 5,
        "brow": 6,
    }
)


def hardware_channel_index(column_name: str) -> int | None:
    """Map a raw or legacy column header to hardware index 0..7.

    Accepts ``ch1``..``ch8``, electrode names (``AF8`` …), legacy labels
    (``HEAD_R`` …), and the output RMS names (``HEAD_R_RMS`` …).
    """
    return _RAW_COLUMN_INDEX.get(str(column_name).strip().lower())


class HermesConstants:
    """Backward-compatible namespace for existing ``HermesConstants.*`` imports."""

    SAMPLING_RATE = SAMPLING_RATE
    CHANNEL_NAMES = list(CHANNEL_NAMES)
    OUTPUT_RMS_COLUMNS = list(OUTPUT_RMS_COLUMNS)
    CHANNELS = CHANNELS
    N_CHANNELS = N_CHANNELS
    DISCONNECT_VALUE = DISCONNECT_VALUE
