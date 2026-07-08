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

CHANNELS: dict[str, int] = {name: i for i, name in enumerate(CHANNEL_NAMES)}
N_CHANNELS: int = len(CHANNEL_NAMES)


class HermesConstants:
    """Backward-compatible namespace for existing ``HermesConstants.*`` imports."""

    SAMPLING_RATE = SAMPLING_RATE
    CHANNEL_NAMES = list(CHANNEL_NAMES)
    CHANNELS = CHANNELS
    N_CHANNELS = N_CHANNELS
    DISCONNECT_VALUE = DISCONNECT_VALUE
