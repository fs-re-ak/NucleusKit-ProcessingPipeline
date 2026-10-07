"""ClassicalEmotionModel — runtime wrapper for the sklearn-based artifact.

This module is the only thing the runtime layer needs to import.  It has no
dependency on training code, pandas, or YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import json
import numpy as np

from exg_emotion.core.interfaces import EmotionModel
from exg_emotion.core.types import EmotionPrediction
from exg_emotion.core.validation import validate_input
from exg_emotion.models.classical_v1.features import (
    FEATURE_SETS,
    _TEMPORAL_EEG_NAME,
    _DEFAULT_EMG_INDICES,
    _DEFAULT_ZYG_L,
    _DEFAULT_ZYG_R,
    _DEFAULT_ANT_A,
    _DEFAULT_ANT_B,
    _DEFAULT_ANT_C,
    _DEFAULT_ANT_D,
    _DEFAULT_NM_A_N,
    _DEFAULT_NM_A_M,
    _DEFAULT_NM_B_N,
    _DEFAULT_NM_B_M,
)

_EXTENDED_FAMILY = {"extended", "extended_nm_a", "extended_nm_a_lp", "extended_nm_b", "extended_nm_ab", "extended_nm_a_corr"}

_CONTEMPT_VARIANTS = ("CONTEMPT_LEFT", "CONTEMPT_RIGHT")
_CONTEMPT_OUT = "CONTEMPT"


def merge_contempt_output(
    label: str,
    probabilities: dict,
) -> tuple[str, dict]:
    """Collapse CONTEMPT_LEFT / CONTEMPT_RIGHT into CONTEMPT after classification.

    Training stays 9-class. This only rewrites the returned label and
    sums the two laterality probabilities.
    """
    if not any(k in probabilities for k in _CONTEMPT_VARIANTS):
        return label, dict(probabilities)
    merged: dict = {}
    mass = 0.0
    for key, value in probabilities.items():
        if key in _CONTEMPT_VARIANTS:
            mass += float(value)
        else:
            merged[key] = float(value)
    merged[_CONTEMPT_OUT] = mass
    out_label = _CONTEMPT_OUT if label in _CONTEMPT_VARIANTS else label
    return out_label, merged


def public_labels(internal_labels: list, merge: bool) -> list:
    """Map internal classifier labels to the labels ``predict()`` returns."""
    if not merge:
        return list(internal_labels)
    out: list = []
    seen = False
    for lbl in internal_labels:
        if lbl in _CONTEMPT_VARIANTS:
            if not seen:
                out.append(_CONTEMPT_OUT)
                seen = True
        else:
            out.append(lbl)
    return out


class ClassicalEmotionModel(EmotionModel):
    """EmotionModel backed by a joblib-serialised sklearn Pipeline.

    Do not instantiate directly.  Use :meth:`from_release_dir` or
    :meth:`from_artifact` in production.

    Parameters
    ----------
    pipeline:
        A fitted sklearn ``Pipeline`` that accepts a flat 1-D feature vector
        and exposes ``predict_proba``.
    labels:
        Ordered list of class labels matching the pipeline's class indices.
    name:
        Model family name (e.g. ``"classical-emotion"``).
    version:
        Semantic version string.
    expected_channels:
        Number of input channels declared in the manifest.
    expected_samples:
        Number of samples per window declared in the manifest.
    channel_names:
        Optional ordered channel names for strict channel-name validation.
    sample_rate_hz:
        Nominal sample rate; used for feature extraction and validation only.
    """

    def __init__(
        self,
        pipeline,  # sklearn Pipeline — no static import so runtime stays slim
        labels: List[str],
        name: str,
        version: str,
        expected_channels: int = 8,
        expected_samples: int = 500,
        channel_names: Optional[List[str]] = None,
        sample_rate_hz: float = 250.0,
        feature_set: str = "full",
        channel_indices: Optional[List[int]] = None,
        merge_contempt: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._labels = labels
        self._name = name
        self._version = version
        self._expected_channels = expected_channels
        self._expected_samples = expected_samples
        self._channel_names = channel_names
        self._sample_rate_hz = sample_rate_hz
        self._feature_set = feature_set
        # Optional channel sub-selection (0-based indices into the raw input)
        self._channel_indices = channel_indices
        self._merge_contempt = merge_contempt
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature_set '{feature_set}'. "
                f"Choose from: {list(FEATURE_SETS)}"
            )
        self._extract_fn = FEATURE_SETS[feature_set]
        # Subject calibration stats for rms_zscore (set via set_subject_context)
        self._subject_mean: Optional[np.ndarray] = None
        self._subject_std: Optional[np.ndarray] = None
        # Additional calibration stats for extended feature set (WL path)
        self._subject_wl_mean: Optional[np.ndarray] = None
        self._subject_wl_std: Optional[np.ndarray] = None
        self._rest_reference_ok: bool = True

        # Derived spatial indices for the extended feature set family.
        # All indices are into the full (post channel_indices) array —
        # for extended family models, channel_indices is always None so these
        # reference the standard 8-channel montage directly.
        if feature_set in _EXTENDED_FAMILY and channel_names:
            ch = list(channel_names)
            self._emg_indices: List[int] = [
                i for i, name in enumerate(ch) if name != _TEMPORAL_EEG_NAME
            ]
            self._zyg_l_idx: int = ch.index("Zygomatic_L")   if "Zygomatic_L"    in ch else _DEFAULT_ZYG_L
            self._zyg_r_idx: int = ch.index("Zygomatic_R")   if "Zygomatic_R"    in ch else _DEFAULT_ZYG_R
            self._ant_a_idx: int = ch.index("Glabella")       if "Glabella"       in ch else _DEFAULT_ANT_A
            self._ant_b_idx: int = ch.index("Supraorbital_L") if "Supraorbital_L" in ch else _DEFAULT_ANT_B
            self._ant_c_idx: int = ch.index("Supraorbital_R") if "Supraorbital_R" in ch else _DEFAULT_ANT_C
            self._ant_d_idx: int = ch.index("Temporal_R")     if "Temporal_R"     in ch else _DEFAULT_ANT_D
            # NM ratio channel indices — always derive both pairs so
            # extended_nm_ab can append A then B.
            self._nm_a_n_idx: int = ch.index("Glabella") if "Glabella" in ch else _DEFAULT_NM_A_N
            self._nm_a_m_indices: List[int] = [
                ch.index("Nasolabial") if "Nasolabial" in ch else _DEFAULT_NM_A_M[0]
            ]
            self._nm_b_n_idx: int = ch.index("Nasolabial") if "Nasolabial" in ch else _DEFAULT_NM_B_N
            self._nm_b_m_indices: List[int] = [
                ch.index("Zygomatic_L") if "Zygomatic_L" in ch else _DEFAULT_NM_B_M[0],
                ch.index("Zygomatic_R") if "Zygomatic_R" in ch else _DEFAULT_NM_B_M[1],
            ]
            # Single-variant aliases used by extended_nm_a / extended_nm_b
            if feature_set == "extended_nm_b":
                self._nm_n_idx: int = self._nm_b_n_idx
                self._nm_m_indices: List[int] = self._nm_b_m_indices
            else:
                self._nm_n_idx = self._nm_a_n_idx
                self._nm_m_indices = self._nm_a_m_indices
        else:
            self._emg_indices  = _DEFAULT_EMG_INDICES
            self._zyg_l_idx    = _DEFAULT_ZYG_L
            self._zyg_r_idx    = _DEFAULT_ZYG_R
            self._ant_a_idx    = _DEFAULT_ANT_A
            self._ant_b_idx    = _DEFAULT_ANT_B
            self._ant_c_idx    = _DEFAULT_ANT_C
            self._ant_d_idx    = _DEFAULT_ANT_D
            self._nm_a_n_idx   = _DEFAULT_NM_A_N
            self._nm_a_m_indices = list(_DEFAULT_NM_A_M)
            self._nm_b_n_idx   = _DEFAULT_NM_B_N
            self._nm_b_m_indices = list(_DEFAULT_NM_B_M)
            self._nm_n_idx     = _DEFAULT_NM_A_N
            self._nm_m_indices = list(_DEFAULT_NM_A_M)

    # ------------------------------------------------------------------
    # EmotionModel contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    # ------------------------------------------------------------------
    # Subject calibration (required for rms_zscore models)
    # ------------------------------------------------------------------

    def set_subject_context(self, windows: np.ndarray) -> None:
        """Calibrate per-channel normalisation stats from this subject's windows.

        Must be called before :meth:`predict` when ``feature_set`` is
        ``"rms_zscore"`` or ``"extended"``.

        Parameters
        ----------
        windows:
            Shape ``[n_windows, n_channels, n_samples]`` — all windows from
            the subject being classified.
        """
        from exg_emotion.models.classical_v1.features import compute_rms_batch, compute_wl_batch

        if self._feature_set == "extended_nm_a_lp":
            from exg_emotion.models.classical_v1.features import (
                compute_rms_batch,
                compute_wl_batch,
                overall_emg_power,
                fit_low_power_stats,
            )
            rms = compute_rms_batch(windows)
            wl = compute_wl_batch(windows[:, self._emg_indices, :])
            power = overall_emg_power(rms, self._emg_indices)
            self._subject_mean, self._subject_std, ok, n_ref = fit_low_power_stats(rms, power)
            self._subject_wl_mean, self._subject_wl_std, _, _ = fit_low_power_stats(wl, power)
            self._rest_reference_ok = ok
            if not ok:
                print(
                    f"[low-power] rest band may not be rest "
                    f"(p20/p50 high, n_ref={n_ref})"
                )
        elif self._feature_set in _EXTENDED_FAMILY:
            # RMS stats: all channels (no pre-slice)
            rms = compute_rms_batch(windows)                         # [n, 8]
            self._subject_mean = rms.mean(axis=0)
            self._subject_std  = rms.std(axis=0)
            # WL stats: EMG channels only
            wl = compute_wl_batch(windows[:, self._emg_indices, :])  # [n, 7]
            self._subject_wl_mean = wl.mean(axis=0)
            self._subject_wl_std  = wl.std(axis=0)
        else:
            if self._channel_indices is not None:
                windows = windows[:, self._channel_indices, :]
            rms = compute_rms_batch(windows)     # [n_windows, n_active_channels]
            self._subject_mean = rms.mean(axis=0)
            self._subject_std  = rms.std(axis=0)

    def _window_features(self, samples: np.ndarray) -> np.ndarray:
        validate_input(
            samples,
            expected_channels=self._expected_channels,
            expected_samples=self._expected_samples,
        )

        # Apply channel sub-selection before feature extraction.
        # Extended models always use the full channel array (channel_indices=None).
        s = samples[self._channel_indices, :] if self._channel_indices is not None else samples

        if self._feature_set == "rms_zscore":
            if self._subject_mean is None:
                raise RuntimeError(
                    "rms_zscore model requires subject calibration before inference. "
                    "Call set_subject_context(windows) with this subject's windows first."
                )
            rms      = self._extract_fn(s, self._sample_rate_hz)  # raw RMS [n_active_ch]
            features = (rms - self._subject_mean) / (self._subject_std + 1e-12)

        elif self._feature_set in _EXTENDED_FAMILY:
            if self._subject_mean is None or self._subject_wl_mean is None:
                raise RuntimeError(
                    f"{self._feature_set} model requires subject calibration before inference. "
                    "Call set_subject_context(windows) with this subject's windows first."
                )
            from exg_emotion.models.classical_v1.features import (
                compute_rms_batch,
                compute_wl_batch,
                _compute_mdf_channel,
                compute_asymmetry_batch,
                compute_antagonist_corr_batch,
                compute_nm_ratio_batch,
            )
            # s is the full 8-channel window; shape [n_ch, n_samples]
            window_batch = s[np.newaxis, :, :]  # [1, n_ch, n_samples]

            # Amplitude-sensitive: RMS (all ch) + WL (EMG ch), subject-normalised
            rms_raw = np.sqrt(np.mean(s.astype(np.float64) ** 2, axis=1))  # [8]
            rms_norm = (rms_raw - self._subject_mean) / (self._subject_std + 1e-12)

            wl_raw  = np.sum(np.abs(np.diff(s[self._emg_indices, :].astype(np.float64), axis=1)), axis=1)  # [7]
            wl_norm = (wl_raw - self._subject_wl_mean) / (self._subject_wl_std + 1e-12)

            # Scale-free: MDF (EMG ch)
            mdf = np.array([
                _compute_mdf_channel(s[ch, :], self._sample_rate_hz)
                for ch in self._emg_indices
            ], dtype=np.float64)  # [7]

            # Scale-free: asymmetry and antagonist correlation
            asym = compute_asymmetry_batch(window_batch, self._zyg_l_idx, self._zyg_r_idx)[0]  # [1]
            corr = compute_antagonist_corr_batch(window_batch, self._ant_a_idx, self._ant_b_idx)[0]  # [1]

            features = np.concatenate([rms_norm, wl_norm, mdf, asym, corr])  # [24]

            # NM ratio(s) appended for the nm variants
            if self._feature_set == "extended_nm_ab":
                nm_a = compute_nm_ratio_batch(window_batch, self._nm_a_n_idx, self._nm_a_m_indices)[0]
                nm_b = compute_nm_ratio_batch(window_batch, self._nm_b_n_idx, self._nm_b_m_indices)[0]
                features = np.concatenate([features, nm_a, nm_b])  # [26]
            elif self._feature_set in ("extended_nm_a", "extended_nm_a_lp", "extended_nm_b"):
                nm = compute_nm_ratio_batch(window_batch, self._nm_n_idx, self._nm_m_indices)[0]  # [1]
                features = np.concatenate([features, nm])  # [25]
            if self._feature_set == "extended_nm_a_corr":
                nm = compute_nm_ratio_batch(window_batch, self._nm_n_idx, self._nm_m_indices)[0]
                corr_sr = compute_antagonist_corr_batch(window_batch, self._ant_a_idx, self._ant_c_idx)[0]
                corr_tr = compute_antagonist_corr_batch(window_batch, self._ant_a_idx, self._ant_d_idx)[0]
                features = np.concatenate([features, nm, corr_sr, corr_tr])  # [27]

        else:
            features = self._extract_fn(s, self._sample_rate_hz)

        features_2d = features[np.newaxis, :]  # [1, n_features]
        return features_2d

    def _to_prediction(self, label: str, probabilities: dict) -> EmotionPrediction:
        if self._merge_contempt:
            label, probabilities = merge_contempt_output(label, probabilities)
        return EmotionPrediction(
            label=label,
            probabilities=probabilities,
            model_name=self._name,
            model_version=self._version,
        )

    def predict(self, samples: np.ndarray) -> EmotionPrediction:
        """Classify one EXG window."""
        features_2d = self._window_features(samples)
        proba = self._pipeline.predict_proba(features_2d)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = self._labels[pred_idx]
        return self._to_prediction(pred_label, dict(zip(self._labels, proba.tolist())))

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_release_dir(cls, release_dir: str | Path) -> "ClassicalEmotionModel":
        """Load a released model from its artifact directory.

        Expects the standard layout::

            releases/<name>/<version>/
                manifest.json
                labels.json
                pipeline.joblib
        """
        import joblib

        release_dir = Path(release_dir)
        manifest = json.loads((release_dir / "manifest.json").read_text())
        pipeline = joblib.load(release_dir / "pipeline.joblib")

        # sklearn fits classifiers with alphabetically-sorted classes_.
        # That ordering must be used as the label list so that predict_proba
        # column indices align with self._labels indices.  labels.json is a
        # human-readable cross-check but must never override clf.classes_.
        clf = pipeline.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "classes_"):
            labels = [str(c) for c in clf.classes_]
        else:
            labels = json.loads((release_dir / "labels.json").read_text())

        input_spec = manifest.get("input", {})
        ch_names = input_spec.get("channels")

        # Read feature_set and channel selection from config.json
        feature_set    = "full"
        channel_indices: Optional[List[int]] = None
        has_laterals = (
            "CONTEMPT_LEFT" in labels and "CONTEMPT_RIGHT" in labels
        )
        # Production default: merge laterality after classification when the
        # 9-way head was trained with CONTEMPT_LEFT / CONTEMPT_RIGHT.
        merge_contempt = has_laterals
        config_path = release_dir / "config.json"
        if config_path.exists():
            config_data    = json.loads(config_path.read_text())
            feature_set    = config_data.get("feature_set", "full")
            in_channels    = config_data.get("input_channels", [])
            drop_channels  = config_data.get("drop_channels", [])
            if "merge_contempt" in config_data:
                merge_contempt = bool(config_data["merge_contempt"])
            # Extended family models manage their own channel selection internally;
            # never set channel_indices for them.
            if feature_set not in _EXTENDED_FAMILY and drop_channels and in_channels:
                drop_set       = set(drop_channels)
                channel_indices = [
                    i for i, ch in enumerate(in_channels) if ch not in drop_set
                ]

        return cls(
            pipeline=pipeline,
            labels=labels,
            name=manifest["name"],
            version=manifest["version"],
            expected_channels=len(ch_names) if ch_names else 8,
            expected_samples=input_spec.get("window_samples", 500),
            channel_names=ch_names,
            sample_rate_hz=float(input_spec.get("sample_rate_hz", 250)),
            feature_set=feature_set,
            channel_indices=channel_indices,
            merge_contempt=merge_contempt,
        )


# Stage-1 gate labels and leaf groups for the hierarchical classifier.
GATE_NEUTRAL = "NEUTRAL"
GATE_SMILE = "SMILE"          # happiness + contempt L/R
GATE_NEGATIVE = "NEGATIVE"    # anger, surprise, fear, sadness, disgust

SMILE_LEAVES = ("HAPPINESS", "CONTEMPT_LEFT", "CONTEMPT_RIGHT")
NEGATIVE_LEAVES = ("ANGER", "SURPRISE", "FEAR", "SADNESS", "DISGUST")

_LEAF_TO_GATE = {
    "NEUTRAL": GATE_NEUTRAL,
    **{lbl: GATE_SMILE for lbl in SMILE_LEAVES},
    **{lbl: GATE_NEGATIVE for lbl in NEGATIVE_LEAVES},
}


class HierarchicalClassicalModel(ClassicalEmotionModel):
    """Three-head logreg: 3-way gate, then smile/contempt and negative specialists.

    Stage 1 predicts NEUTRAL / SMILE / NEGATIVE.
    Stage 2a (SMILE) predicts HAPPINESS / CONTEMPT_LEFT / CONTEMPT_RIGHT.
    Stage 2b (NEGATIVE) predicts ANGER / SURPRISE / FEAR / SADNESS / DISGUST.
    """

    def __init__(
        self,
        stage1,
        stage2a,
        stage2b,
        leaf_labels: List[str],
        **kwargs,
    ) -> None:
        super().__init__(pipeline=stage1, labels=list(leaf_labels), **kwargs)
        self._stage1 = stage1
        self._stage2a = stage2a
        self._stage2b = stage2b
        self._checkpoint = {
            "architecture": "hierarchical",
            "stage1": stage1,
            "stage2a": stage2a,
            "stage2b": stage2b,
        }

    def predict(self, samples: np.ndarray) -> EmotionPrediction:
        features_2d = self._window_features(samples)
        p1 = self._stage1.predict_proba(features_2d)[0]
        gate_classes = list(self._stage1.named_steps["classifier"].classes_)
        gate_proba = dict(zip(gate_classes, p1.tolist()))
        gate = gate_classes[int(np.argmax(p1))]

        leaf_probs = {lbl: 0.0 for lbl in self._labels}

        if gate == GATE_NEUTRAL:
            leaf_probs["NEUTRAL"] = gate_proba.get(GATE_NEUTRAL, 0.0)
            pred_label = "NEUTRAL"
        elif gate == GATE_SMILE:
            p2 = self._stage2a.predict_proba(features_2d)[0]
            c2 = list(self._stage2a.named_steps["classifier"].classes_)
            gmass = gate_proba.get(GATE_SMILE, 0.0)
            for cls, p in zip(c2, p2.tolist()):
                if cls in leaf_probs:
                    leaf_probs[cls] = gmass * p
            pred_label = c2[int(np.argmax(p2))]
        else:
            p2 = self._stage2b.predict_proba(features_2d)[0]
            c2 = list(self._stage2b.named_steps["classifier"].classes_)
            gmass = gate_proba.get(GATE_NEGATIVE, 0.0)
            for cls, p in zip(c2, p2.tolist()):
                if cls in leaf_probs:
                    leaf_probs[cls] = gmass * p
            pred_label = c2[int(np.argmax(p2))]

        # Keep NEUTRAL probability from the gate as well
        if "NEUTRAL" in leaf_probs and gate != GATE_NEUTRAL:
            leaf_probs["NEUTRAL"] = gate_proba.get(GATE_NEUTRAL, 0.0)

        return self._to_prediction(pred_label, leaf_probs)
