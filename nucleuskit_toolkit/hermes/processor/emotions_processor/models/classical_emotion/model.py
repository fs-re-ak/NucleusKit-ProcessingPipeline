"""
Adapter wrapping classical-emotion 2.5.0 as a NucleusKit EmotionModel.

The vendored release lives at
``models/classical-emotion-2.5.0/``.  Its ``exg_emotion`` sub-package is
added to ``sys.path`` on first load so the adapter can import
``exg_emotion.runtime.predictor.Predictor`` without requiring an installed
package.

Subject calibration contract
-----------------------------
The underlying model (``extended_nm_a`` feature set) performs per-subject
z-scoring of RMS and waveform-length features.  :meth:`set_subject_context`
**must** be called once with all valid windows from the current recording
before any :meth:`predict` call.  Calling :meth:`reset` between recordings
clears the calibration so no subject's stats bleed into the next.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from nucleuskit_toolkit.hermes.processor.emotions_processor.constants import EMOTION_COLUMNS
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.model import (
    EmotionModel,
    EmotionWindowResult,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

# Absolute path to the vendored classical-emotion 2.5.0 release directory.
_RELEASE_DIR: Path = (
    Path(__file__).parent.parent / "classical-emotion-2.5.0"
)

# Number of features produced by the extended_nm_a feature set.
_N_FEATURES: int = 25

# Mapping from model uppercase labels → title-case EMOTION_COLUMNS keys.
_LABEL_MAP: dict[str, str] = {col.upper(): col for col in EMOTION_COLUMNS}

# Hermes canonical channel names (index order from loadEXG).
_HERMES_CHANNELS: tuple[str, ...] = (
    "AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE",
)

# EAR_R maps to Temporal_EEG and is excluded from WL and MDF features.
_TEMPORAL_EEG_IDX: int = 4
_EMG_CHANNELS: tuple[str, ...] = tuple(
    c for i, c in enumerate(_HERMES_CHANNELS) if i != _TEMPORAL_EEG_IDX
)

# Ordered feature column names for emotionClassifierInputs.csv.
FEATURE_COLUMNS: list[str] = (
    [f"rms_{c}" for c in _HERMES_CHANNELS]   # 0-7:  per-channel RMS (z-scored)
    + [f"wl_{c}" for c in _EMG_CHANNELS]      # 8-14: waveform-length (z-scored)
    + [f"mdf_{c}" for c in _EMG_CHANNELS]     # 15-21: median frequency
    + ["zyg_asymmetry", "ant_corr", "nm_ratio_a"]  # 22-24: scale-free features
)


# ---------------------------------------------------------------------------
# sys.path helper
# ---------------------------------------------------------------------------

def _ensure_exg_emotion_on_path() -> None:
    """Insert the vendored release directory into ``sys.path`` if needed."""
    release_str = str(_RELEASE_DIR)
    if release_str not in sys.path:
        sys.path.insert(0, release_str)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ClassicalEmotionAdapter(EmotionModel):
    """EmotionModel backed by the classical-emotion 2.5.0 ``exg_emotion`` package.

    Loads ``pipeline.joblib`` from the vendored release directory.  Channel
    order from :func:`~...session.loadEXG` matches the model training order
    (Hermes hardware order = ``Supraorbital_L`` … ``Nasolabial``), so no
    permutation is applied.

    Parameters
    ----------
    predictor:
        A loaded ``exg_emotion.runtime.predictor.Predictor`` instance.
    """

    def __init__(self, predictor) -> None:  # noqa: ANN001
        self._predictor = predictor

    # ------------------------------------------------------------------ #
    # EmotionModel contract                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, release_dir: Optional[str] = None, **kwargs) -> "ClassicalEmotionAdapter":
        """Load classical-emotion 2.5.0 and return a ready adapter.

        Parameters
        ----------
        release_dir:
            Path to the ``classical-emotion-2.5.0`` release artifact directory.
            Defaults to the vendored copy bundled with this package.
        **kwargs:
            Ignored; accepted for registry-call compatibility.
        """
        _ensure_exg_emotion_on_path()
        from exg_emotion.runtime.predictor import Predictor  # type: ignore[import]

        rdir = Path(release_dir) if release_dir else _RELEASE_DIR
        predictor = Predictor.from_release_dir(rdir)
        return cls(predictor)

    def reset(self) -> None:
        """Clear per-subject calibration stats so the adapter is ready for a new recording."""
        inner = self._predictor._model
        inner._subject_mean = None
        inner._subject_std = None
        inner._subject_wl_mean = None
        inner._subject_wl_std = None

    def set_subject_context(self, windows: np.ndarray) -> None:
        """Calibrate per-subject RMS and waveform-length normalisation stats.

        Parameters
        ----------
        windows:
            Array of shape ``(n_windows, n_channels, n_samples)`` containing
            all valid, already-filtered windows from the current recording.
        """
        self._predictor.set_subject_context(windows)

    def predict(self, samples: np.ndarray) -> EmotionWindowResult:
        """Classify one filtered EXG window.

        Parameters
        ----------
        samples:
            Array of shape ``(n_channels, n_samples)`` = ``(8, 500)``,
            bandpass-filtered, no NLMS.

        Returns
        -------
        EmotionWindowResult
            Contains title-case probabilities (matching
            :data:`~...constants.EMOTION_COLUMNS`), raw per-channel RMS,
            and the 25-D feature vector.
        """
        prediction = self._predictor.predict(samples)

        # Map uppercase model labels → title-case EMOTION_COLUMNS keys.
        probabilities: dict[str, float] = {
            _LABEL_MAP.get(k, k.title()): float(v)
            for k, v in prediction.probabilities.items()
        }

        label_tc = _LABEL_MAP.get(prediction.label, prediction.label.title())
        confidence = float(probabilities.get(label_tc, 0.0))

        # Raw per-channel RMS (8 elements, no normalisation) for rmsSignals.csv.
        channel_rms = np.sqrt(np.mean(np.asarray(samples, dtype=np.float64) ** 2, axis=1))

        # 25-D feature vector for emotionClassifierInputs.csv.
        try:
            inner = self._predictor._model
            features = np.asarray(inner._window_features(samples)[0], dtype=np.float64)
        except Exception:
            features = np.full(_N_FEATURES, np.nan, dtype=np.float64)

        return EmotionWindowResult(
            probabilities=probabilities,
            label=label_tc,
            confidence=confidence,
            channel_rms=channel_rms,
            model_features=features,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        """Model family name as declared in ``manifest.json``."""
        return self._predictor.model_name

    @property
    def model_version(self) -> str:
        """Semantic version string as declared in ``manifest.json``."""
        return self._predictor.model_version

    @property
    def feature_columns(self) -> list[str]:
        """25 ordered feature column names (RMS + WL + MDF + scale-free)."""
        return FEATURE_COLUMNS

    @property
    def emotion_labels(self) -> list[str]:
        """All emotion labels this model can output (title-case)."""
        return list(EMOTION_COLUMNS)
