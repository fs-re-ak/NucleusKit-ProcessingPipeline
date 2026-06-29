"""
Hardware and experiment configuration for ModelV12.

All paths are expressed relative to the HermesEmotions directory so the
module is portable across machines.  Adjust ExperimentConfig.data_root and
ExperimentConfig.features_root if your layout differs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class HardwareConfig:
    """Physical constants tied to the Hermes headset hardware."""

    sampling_rate: int = 250
    """Acquisition rate in Hz."""

    n_channels: int = 8
    """Number of EEG/EMG channels."""

    channel_names: List[str] = field(
        default_factory=lambda: [
            "HEAD_R", "HEAD_L", "CHEEK_R", "CHEEK_L",
            "EAR_R", "FOREHEAD_L", "BROW", "NOSE",
        ]
    )
    """Ordered channel labels matching the column indices in the raw data file."""

    bandpass_low: float = 15.0
    """Lower bandpass cutoff (Hz) for EMG conditioning."""

    bandpass_high: float = 45.0
    """Upper bandpass cutoff (Hz) for EMG conditioning."""

    filter_order: int = 4
    """Butterworth filter order applied twice (double-pass causal)."""

    @property
    def sampling_period(self) -> float:
        return 1.0 / self.sampling_rate

    @property
    def rms_feature_names(self) -> List[str]:
        """Per-channel L2-normalised RMS column names (n_channels entries)."""
        return [f"{ch}_RMS" for ch in self.channel_names]

    @property
    def all_feature_names(self) -> List[str]:
        """All feature columns in the order returned by RMSExtractor.transform().

        Per-channel normalised RMS values followed by ``AVG_RMS`` (raw mean
        over channels, computed before L2 normalisation).
        """
        return self.rms_feature_names + ["AVG_RMS"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Dataset and windowing parameters for the V12 training run."""

    # ------------------------------------------------------------------
    # Paths (relative to the HermesEmotions directory)
    # ------------------------------------------------------------------
    data_root: str = os.path.join("data", "raw", "dataset_v7")
    """Directory containing one sub-folder per recording session."""

    features_root: str = os.path.join("data", "features")
    """Directory where featuresDf.csv and stats.csv will be written."""

    # ------------------------------------------------------------------
    # Recording filters
    # ------------------------------------------------------------------
    skip_prefixes: List[str] = field(default_factory=lambda: ["z"])
    """Recording folders whose name starts with any of these strings are skipped."""

    recordings_to_skip: List[str] = field(default_factory=list)
    """Explicit list of recording folder names to skip regardless of prefix."""

    # ------------------------------------------------------------------
    # Trial / windowing parameters
    # ------------------------------------------------------------------
    trial_length_sec: float = 6.0
    """Duration of a single emotion trial in seconds."""

    window_length_sec: float = 1.5
    """Duration of each non-overlapping analysis window within a trial."""

    @property
    def nb_windows(self) -> int:
        """Number of contiguous, non-overlapping windows per trial."""
        return int(self.trial_length_sec / self.window_length_sec)

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------
    normalize_rms: bool = True
    """Apply L2 normalization to each RMS feature vector."""

    # ------------------------------------------------------------------
    # Emotion labels
    # ------------------------------------------------------------------
    emotions: List[str] = field(
        default_factory=lambda: [
            "NEUTRAL", "HAPPY", "ANGER", "SURPRISE",
            "CONTEMPT-L", "CONTEMPT-R", "DISGUST",
            "FEAR", "SADNESS", "BLINKS", "JAW_CLENCH",
            "HEAD_UP", "HEAD_DOWN",
        ]
    )


# ---------------------------------------------------------------------------
# Module-level singletons used as defaults throughout the pipeline
# ---------------------------------------------------------------------------
DEFAULT_HW = HardwareConfig()
DEFAULT_EXP = ExperimentConfig()
