"""
Two-stage classifier inference — ModelV12.

V12 extends the V10 TwoStageClassifier with an artefact-rejection gate that
runs *before* Stage 1.  Windows whose AVG_RMS (the last feature column)
exceeds a pre-computed threshold are immediately returned as Neutral with an
artefact flag, without consulting Stage 1 or Stage 2.

The threshold is loaded from ``artefact_config.json`` in the classifier
directory.  If that file is absent the gate is disabled, preserving full
backward compatibility with V10 artifact directories.

V12 also supports an optional **post-artefact cooldown**: when an artefact
is detected, the next ``cooldown_windows`` windows are automatically rejected
as Neutral (artefact) regardless of their AVG_RMS.  This prevents windows
immediately following a glitch — whose amplitude may have partially decayed
but whose signal is still corrupt — from being mistakenly classified as
emotional expressions.  The cooldown is opt-in (default 0) so the stateless
behavior is preserved when not needed.

Usage
-----
  from inference import TwoStageClassifier

  # Stateless (default — identical to V10 behaviour)
  clf = TwoStageClassifier.load("data/features/2026-04-22_ModelV12/classifier")

  # Stateful with 2-window post-artefact cooldown
  clf = TwoStageClassifier.load("…/classifier", cooldown_windows=2)

  label, confidence = clf.predict(x)           # x: 1-D np.ndarray of features
  proba_dict = clf.predict_proba(x)            # {label: probability}
  is_art = clf.is_artefact(x)                  # bool (threshold only, no state)

  # Reset cooldown state between recordings
  clf.reset_artefact_state()
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np


class TwoStageClassifier:
    """Stateless two-stage emotion classifier with an artefact-rejection gate.

    Gate (V12)  — AVG_RMS pre-check before any LDA inference.
    Stage 1     — Binary Neutral vs. Active shrinkage LDA.
    Stage 2     — Emotion-class shrinkage LDA + Mahalanobis KNN ensemble.

    Parameters are loaded from ``config.json`` and optionally
    ``artefact_config.json`` at construction.
    """

    NEUTRAL_LABEL: str = "Neutral"
    ARTEFACT_LABEL: str = "Neutral"   # artefact windows are reported as Neutral

    def __init__(
        self,
        stage1,
        stage2_lda,
        stage2_knn,
        stage2_cov: np.ndarray,
        feature_columns: List[str],
        label_encoder,
        config: dict,
        artefact_config: Optional[dict] = None,
        cooldown_windows: int = 0,
        threshold_override: Optional[float] = None,
    ) -> None:
        self._stage1 = stage1
        self._stage2_lda = stage2_lda
        self._stage2_knn = stage2_knn
        self._stage2_cov = stage2_cov
        self._feature_columns = feature_columns
        self._label_encoder = label_encoder

        self.theta: float = float(config.get("theta", 0.5))
        self.lda_weight: float = float(config.get("lda_ensemble_weight", 0.6))
        self.knn_weight: float = float(config.get("knn_ensemble_weight", 0.4))
        self._emotion_classes: np.ndarray = np.asarray(
            config.get("stage2_classes", self._stage2_lda.classes_)
        )
        self._active_col_idx: int = list(self._stage1.classes_).index(1)

        # Artefact gate — None means disabled (V10 compatibility)
        if artefact_config is not None:
            self.artefact_threshold: Optional[float] = float(
                artefact_config["threshold"]
            )
            self._artefact_percentile: float = float(
                artefact_config.get("percentile", 95.0)
            )
            self._artefact_scale: float = float(
                artefact_config.get("scale", 1.0)
            )
        else:
            self.artefact_threshold = None
            self._artefact_percentile = 95.0
            self._artefact_scale = 1.0

        # threshold_override lets callers raise/lower the gate at runtime
        # without retraining or editing artefact_config.json.
        if threshold_override is not None:
            self.artefact_threshold = float(threshold_override)

        # Index of AVG_RMS in the feature vector (always the last column)
        self._avg_rms_idx: int = len(feature_columns) - 1

        # Post-artefact cooldown — number of windows to reject after an artefact.
        # 0 = stateless (default); > 0 = stateful, requires reset_artefact_state()
        # between independent recordings.
        self.cooldown_windows: int = max(0, int(cooldown_windows))
        self._artefact_cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        classifier_dir: str,
        cooldown_windows: int = 0,
        threshold_override: Optional[float] = None,
    ) -> "TwoStageClassifier":
        """Load a TwoStageClassifier from a serialised artifact directory.

        Parameters
        ----------
        classifier_dir:
            Path to the ``classifier/`` folder produced by ``train_and_save()``.
            Required files: stage1_neutral_detector.pkl, stage2_lda.pkl,
            stage2_knn.pkl, stage2_cov.npy, feature_columns.json, config.json.
            Optional V12 file: artefact_config.json (gate disabled if absent).
        cooldown_windows:
            Number of windows to reject after an artefact (post-artefact cooldown).
            0 (default) = stateless, no cooldown.  > 0 = stateful; call
            ``reset_artefact_state()`` between independent recordings.
        threshold_override:
            If provided, replaces the AVG_RMS threshold from ``artefact_config.json``
            at runtime.  Use this to tune the gate without retraining (e.g. set to
            35.0 when the training-set percentile is too conservative for a given
            subject or session).  Has no effect when the gate is disabled.

        Raises
        ------
        FileNotFoundError
            If any required artifact is missing.
        """
        required = [
            "stage1_neutral_detector.pkl",
            "stage2_lda.pkl",
            "stage2_knn.pkl",
            "stage2_cov.npy",
            "feature_columns.json",
            "config.json",
        ]
        for fname in required:
            fpath = os.path.join(classifier_dir, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    f"Required artifact '{fname}' not found in {classifier_dir}"
                )

        stage1 = joblib.load(os.path.join(classifier_dir, "stage1_neutral_detector.pkl"))
        stage2_lda = joblib.load(os.path.join(classifier_dir, "stage2_lda.pkl"))
        stage2_knn = joblib.load(os.path.join(classifier_dir, "stage2_knn.pkl"))
        stage2_cov = np.load(os.path.join(classifier_dir, "stage2_cov.npy"))

        with open(os.path.join(classifier_dir, "feature_columns.json")) as fh:
            feature_columns: List[str] = json.load(fh)

        with open(os.path.join(classifier_dir, "config.json")) as fh:
            config: dict = json.load(fh)

        le_path = os.path.join(classifier_dir, "label_encoder.pkl")
        label_encoder = joblib.load(le_path) if os.path.isfile(le_path) else None

        # V12 artefact config — optional
        artefact_path = os.path.join(classifier_dir, "artefact_config.json")
        artefact_config: Optional[dict] = None
        if os.path.isfile(artefact_path):
            with open(artefact_path) as fh:
                artefact_config = json.load(fh)

        return cls(
            stage1=stage1,
            stage2_lda=stage2_lda,
            stage2_knn=stage2_knn,
            stage2_cov=stage2_cov,
            feature_columns=feature_columns,
            label_encoder=label_encoder,
            config=config,
            artefact_config=artefact_config,
            cooldown_windows=cooldown_windows,
            threshold_override=threshold_override,
        )

    # ------------------------------------------------------------------
    # Artefact gate helper
    # ------------------------------------------------------------------

    def is_artefact(self, x: np.ndarray) -> bool:
        """Return True if *x* exceeds the artefact threshold.

        Always returns False when the gate is disabled (V10 artifacts).
        """
        if self.artefact_threshold is None:
            return False
        x2d = self._prepare_input(x)
        return bool(x2d[0, self._avg_rms_idx] > self.artefact_threshold)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def infer(self, x: np.ndarray) -> Tuple[Dict[str, float], str, float]:
        """Single forward pass: class probabilities, discrete label, and label confidence.

        Use this when you need both ``predict_proba`` and ``predict`` outputs for the
        same window.  Calling those methods separately would apply the artefact gate
        twice and break ``cooldown_windows`` state.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,) or (1, n_features)

        Returns
        -------
        proba_dict : dict
            Same mapping as :meth:`predict_proba`.
        label : str
            Same label as :meth:`predict`.
        confidence : float
            Same confidence as :meth:`predict`.
        """
        x2d = self._prepare_input(x)

        # --- Artefact gate (stateful when cooldown_windows > 0) ---
        if self.artefact_threshold is not None:
            if x2d[0, self._avg_rms_idx] > self.artefact_threshold:
                self._artefact_cooldown_remaining = self.cooldown_windows
                result: Dict[str, float] = {self.NEUTRAL_LABEL: 1.0}
                for label in self._emotion_classes:
                    result[str(label)] = 0.0
                return result, self.ARTEFACT_LABEL, 1.0
            if self._artefact_cooldown_remaining > 0:
                self._artefact_cooldown_remaining -= 1
                result = {self.NEUTRAL_LABEL: 1.0}
                for label in self._emotion_classes:
                    result[str(label)] = 0.0
                return result, self.ARTEFACT_LABEL, 1.0

        s1_proba = self._stage1.predict_proba(x2d)[0]
        neutral_col = list(self._stage1.classes_).index(0)
        p_neutral = float(s1_proba[neutral_col])
        p_active = float(s1_proba[self._active_col_idx])

        ens = self._stage2_ensemble(x2d)
        scaled_ens = ens * p_active

        result = {self.NEUTRAL_LABEL: p_neutral}
        for label, prob in zip(self._emotion_classes, scaled_ens):
            result[str(label)] = float(prob)

        if p_active < self.theta:
            return result, self.NEUTRAL_LABEL, float(1.0 - p_active)

        best_idx = int(np.argmax(ens))
        return (
            result,
            str(self._emotion_classes[best_idx]),
            float(ens[best_idx]),
        )

    def predict(self, x: np.ndarray) -> Tuple[str, float]:
        """Predict the emotion label for a single feature vector.

        The artefact gate is checked first.  If it fires, ``("Neutral", 1.0)``
        is returned immediately without invoking Stage 1 or Stage 2.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,) or (1, n_features)

        Returns
        -------
        label : str
        confidence : float
        """
        _, label, confidence = self.infer(x)
        return label, confidence

    def predict_proba(self, x: np.ndarray) -> Dict[str, float]:
        """Return a probability distribution over all labels (including Neutral).

        For artefact windows every probability mass is placed on Neutral.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,) or (1, n_features)

        Returns
        -------
        dict mapping label → probability
        """
        proba_dict, _, _ = self.infer(x)
        return proba_dict

    def predict_from_dict(
        self, feature_dict: Dict[str, float]
    ) -> Tuple[str, float]:
        """Convenience wrapper: accepts a dict keyed by feature name."""
        x = np.array(
            [feature_dict.get(col, 0.0) for col in self._feature_columns],
            dtype=float,
        )
        return self.predict(x)

    # ------------------------------------------------------------------
    # Batch API
    # ------------------------------------------------------------------

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels and confidences for a batch of samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of str, shape (n_samples,)
        confidences : np.ndarray of float, shape (n_samples,)
        """
        self._validate_shape(X)

        labels = np.empty(len(X), dtype=object)
        confidences = np.empty(len(X), dtype=float)

        # When cooldown is active, process sequentially so the counter
        # propagates correctly across windows.
        if self.cooldown_windows > 0:
            for i, row in enumerate(X):
                labels[i], confidences[i] = self.predict(row)
            return labels.astype(str), confidences

        # --- Vectorized path (stateless, cooldown_windows == 0) ---
        if self.artefact_threshold is not None:
            artefact_mask = X[:, self._avg_rms_idx] > self.artefact_threshold
            labels[artefact_mask] = self.ARTEFACT_LABEL
            confidences[artefact_mask] = 1.0
        else:
            artefact_mask = np.zeros(len(X), dtype=bool)

        remaining = ~artefact_mask
        if not remaining.any():
            return labels.astype(str), confidences

        X_rem = X[remaining]

        # --- Stage 1 ---
        s1_proba = self._stage1.predict_proba(X_rem)
        p_active = s1_proba[:, self._active_col_idx]

        neutral_mask_rem = p_active < self.theta
        neutral_idx = np.where(remaining)[0][neutral_mask_rem]
        labels[neutral_idx] = self.NEUTRAL_LABEL
        confidences[neutral_idx] = 1.0 - p_active[neutral_mask_rem]

        active_mask_rem = ~neutral_mask_rem
        if active_mask_rem.any():
            X_act = X_rem[active_mask_rem]
            ens = self._stage2_ensemble_batch(X_act)
            best_idx = np.argmax(ens, axis=1)
            active_idx = np.where(remaining)[0][active_mask_rem]
            labels[active_idx] = self._emotion_classes[best_idx]
            confidences[active_idx] = ens[np.arange(len(X_act)), best_idx]

        return labels.astype(str), confidences

    def predict_batch_detailed(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Like predict_batch but also returns artefact mask and Stage 1 p_active.

        Returns
        -------
        labels : ndarray of str, shape (n_samples,)
        confidences : ndarray of float, shape (n_samples,)
        artefact_mask : ndarray of bool, shape (n_samples,)
        p_active : ndarray of float, shape (n_samples,)
            Stage 1 active probability.  Set to 0.0 for artefact windows.
        """
        self._validate_shape(X)

        labels = np.empty(len(X), dtype=object)
        confidences = np.empty(len(X), dtype=float)
        p_active_out = np.zeros(len(X), dtype=float)

        # When cooldown is active, process sequentially so the counter
        # propagates correctly across windows.  Stage 1 p_active is not
        # available for cooldown-rejected windows (left at 0.0).
        if self.cooldown_windows > 0:
            artefact_mask = np.zeros(len(X), dtype=bool)
            for i, row in enumerate(X):
                x2d = self._prepare_input(row)
                avg_rms = x2d[0, self._avg_rms_idx]

                # Determine artefact status BEFORE calling predict(), which
                # mutates _artefact_cooldown_remaining.  Checking the label
                # would be wrong because ARTEFACT_LABEL == "Neutral", making
                # legitimate Stage 1 Neutral predictions indistinguishable.
                is_art = (
                    self.artefact_threshold is not None
                    and (
                        avg_rms > self.artefact_threshold
                        or self._artefact_cooldown_remaining > 0
                    )
                )

                label, conf = self.predict(row)
                labels[i] = label
                confidences[i] = conf
                artefact_mask[i] = is_art
                if not is_art:
                    p_active_out[i] = float(
                        self._stage1.predict_proba(x2d)[0, self._active_col_idx]
                    )
            return labels.astype(str), confidences, artefact_mask, p_active_out

        # --- Vectorized path (stateless, cooldown_windows == 0) ---
        if self.artefact_threshold is not None:
            artefact_mask = X[:, self._avg_rms_idx] > self.artefact_threshold
        else:
            artefact_mask = np.zeros(len(X), dtype=bool)

        labels[artefact_mask] = self.ARTEFACT_LABEL
        confidences[artefact_mask] = 1.0

        remaining = ~artefact_mask
        if remaining.any():
            X_rem = X[remaining]
            s1_proba = self._stage1.predict_proba(X_rem)
            p_active_rem = s1_proba[:, self._active_col_idx]
            p_active_out[remaining] = p_active_rem

            neutral_mask_rem = p_active_rem < self.theta
            neutral_idx = np.where(remaining)[0][neutral_mask_rem]
            labels[neutral_idx] = self.NEUTRAL_LABEL
            confidences[neutral_idx] = 1.0 - p_active_rem[neutral_mask_rem]

            active_mask_rem = ~neutral_mask_rem
            if active_mask_rem.any():
                X_act = X_rem[active_mask_rem]
                ens = self._stage2_ensemble_batch(X_act)
                best_idx = np.argmax(ens, axis=1)
                active_idx = np.where(remaining)[0][active_mask_rem]
                labels[active_idx] = self._emotion_classes[best_idx]
                confidences[active_idx] = ens[np.arange(len(X_act)), best_idx]

        return labels.astype(str), confidences, artefact_mask, p_active_out

    # ------------------------------------------------------------------
    # Artefact state management
    # ------------------------------------------------------------------

    def reset_artefact_state(self) -> None:
        """Reset the post-artefact cooldown counter.

        Call this between independent recordings when ``cooldown_windows > 0``
        to prevent the tail-end cooldown of one session from bleeding into the
        next.  Has no effect when ``cooldown_windows == 0``.
        """
        self._artefact_cooldown_remaining = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_columns(self) -> List[str]:
        """Ordered list of feature column names expected at inference time."""
        return list(self._feature_columns)

    @property
    def emotion_classes(self) -> List[str]:
        """Stage 2 emotion class labels (excludes Neutral)."""
        return list(self._emotion_classes)

    @property
    def all_classes(self) -> List[str]:
        """All output labels (Neutral + emotion classes)."""
        return [self.NEUTRAL_LABEL] + self.emotion_classes

    @property
    def artefact_gate_enabled(self) -> bool:
        """True if an artefact threshold is loaded and active."""
        return self.artefact_threshold is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self._validate_shape(x)
        return x

    def _validate_shape(self, X: np.ndarray) -> None:
        expected = len(self._feature_columns)
        if X.shape[1] != expected:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected} columns "
                f"({self._feature_columns}), got {X.shape[1]}."
            )

    def _stage2_ensemble(self, x2d: np.ndarray) -> np.ndarray:
        lda_p = self._stage2_lda.predict_proba(x2d)[0]
        knn_p = self._stage2_knn.predict_proba(x2d)[0]
        knn_p_aligned = self._align_proba_1d(knn_p, self._stage2_knn.classes_)
        return self.lda_weight * lda_p + self.knn_weight * knn_p_aligned

    def _stage2_ensemble_batch(self, X: np.ndarray) -> np.ndarray:
        lda_p = self._stage2_lda.predict_proba(X)
        knn_p = self._stage2_knn.predict_proba(X)
        knn_p_aligned = self._align_proba_batch(knn_p, self._stage2_knn.classes_)
        return self.lda_weight * lda_p + self.knn_weight * knn_p_aligned

    def _align_proba_1d(
        self, knn_proba: np.ndarray, knn_classes: np.ndarray
    ) -> np.ndarray:
        aligned = np.zeros(len(self._emotion_classes))
        src_map = {cls: i for i, cls in enumerate(knn_classes)}
        for j, cls in enumerate(self._emotion_classes):
            if cls in src_map:
                aligned[j] = knn_proba[src_map[cls]]
        return aligned

    def _align_proba_batch(
        self, knn_proba: np.ndarray, knn_classes: np.ndarray
    ) -> np.ndarray:
        aligned = np.zeros((knn_proba.shape[0], len(self._emotion_classes)))
        src_map = {cls: i for i, cls in enumerate(knn_classes)}
        for j, cls in enumerate(self._emotion_classes):
            if cls in src_map:
                aligned[:, j] = knn_proba[:, src_map[cls]]
        return aligned

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        gate = (
            f"artefact_threshold={self.artefact_threshold:.4f}"
            if self.artefact_threshold is not None
            else "artefact_gate=disabled"
        )
        return (
            f"TwoStageClassifier("
            f"theta={self.theta}, "
            f"lda_weight={self.lda_weight}, "
            f"knn_weight={self.knn_weight}, "
            f"n_features={len(self._feature_columns)}, "
            f"emotion_classes={self.emotion_classes}, "
            f"{gate}"
            f")"
        )
