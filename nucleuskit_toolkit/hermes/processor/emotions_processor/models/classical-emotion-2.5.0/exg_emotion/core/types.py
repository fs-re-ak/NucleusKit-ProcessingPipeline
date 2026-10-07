"""Fundamental value types for the exg_emotion system.

These are the only types the production runtime needs to know about.
Training, evaluation, and release code share them as a common vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Signal input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalWindow:
    """A single pre-filtered, pre-windowed EXG epoch ready for classification.

    samples
        Float32 array of shape ``[n_channels, n_samples]``.  The caller is
        responsible for filtering and windowing; the model validates the shape
        against its own manifest but does not re-filter.
    channel_names
        Ordered channel labels matching the rows of *samples* (e.g.
        ``["Supraorbital_L", "Supraorbital_R", ...]``).  May be ``None`` when
        the caller cannot supply names; the model then skips name validation.
    sample_rate_hz
        Nominal sample rate in Hz.  Used for manifest validation only.
    """

    samples: object  # numpy ndarray [n_channels, n_samples]
    channel_names: Optional[Sequence[str]] = None
    sample_rate_hz: Optional[float] = None


# ---------------------------------------------------------------------------
# Classification output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmotionPrediction:
    """The result of a single forward pass through an EmotionModel.

    Fields are a strict superset of what callers need today so that adding
    SQI fields later does not break any existing call site.
    """

    label: str
    """Winning class label (e.g. ``"HAPPINESS"``)."""

    probabilities: Mapping[str, float]
    """Per-class probabilities; values sum to 1.0."""

    model_name: str
    model_version: str

    quality: Optional[float] = None
    """Signal quality index [0, 1]; ``None`` until SQI is implemented."""

    usable: bool = True
    """Whether the prediction should be acted upon.  When ``False``, the
    reason string explains why (e.g. "low signal quality")."""

    reason: Optional[str] = None
    """Human-readable explanation when ``usable`` is ``False``."""


# ---------------------------------------------------------------------------
# Experiment record
# ---------------------------------------------------------------------------


@dataclass
class Experiment:
    """Mutable record produced by a single training run.

    Written to ``experiments/exp-NNNN/experiment.json`` so every run is
    reproducible.  The split specification is embedded so results never
    depend on external state.
    """

    id: str
    """Zero-padded counter, e.g. ``"exp-0001"``."""

    model_family: str
    """Registered family name, e.g. ``"classical_v1"``."""

    config: dict
    """Full config snapshot as loaded from the YAML file."""

    dataset_version: str
    """Content-derived fingerprint of the dataset used for training."""

    split_spec: dict
    """Full split specification (strategy, ratios, seed, folds)."""

    git_commit: Optional[str]
    """Short SHA of HEAD at training time; ``None`` if not in a git repo."""

    metrics: Optional[dict] = None
    """Evaluation results populated after training + eval."""

    checkpoint_path: Optional[str] = None
    """Relative path to the serialized model artifact inside the experiment dir."""

    experiment_dir: Optional[str] = None
    """Absolute path to ``experiments/exp-NNNN/``."""


# ---------------------------------------------------------------------------
# Model release
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRelease:
    """Immutable record pointing to a released model artifact directory.

    The directory layout is:

    .. code-block:: text

        releases/<name>/<version>/
            manifest.json
            config.json
            labels.json
            metrics.json
            pipeline.joblib        (classical) or model.pt (neural)
            README.md
    """

    name: str
    version: str
    release_dir: str
    manifest: dict = field(default_factory=dict)
