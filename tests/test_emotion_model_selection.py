"""Discovery, preprocess specs, import isolation, and incremental skip logic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from nucleuskit_toolkit.hermes.processor.emotions_processor.interface.streaming import (
    StreamingWindowSource,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.models import (
    DEFAULT_MODEL,
    DEFAULT_RELEASE_KEY,
    available_model_keys,
    canonicalize_model_key,
    get_model,
    list_models,
    load_preprocess_spec,
    resolve_model,
)
from nucleuskit_toolkit.hermes.processor.emotions_processor.session import (
    outputs_match_requested_model,
)

_MODELS_DIR = (
    Path(__file__).parent.parent
    / "nucleuskit_toolkit"
    / "hermes"
    / "processor"
    / "emotions_processor"
    / "models"
)
RELEASE_25 = _MODELS_DIR / "classical-emotion-2.5.0"
RELEASE_26 = _MODELS_DIR / "classical-emotion-2.6.0"


class TestDiscovery:
    def test_list_models_includes_both_releases(self):
        keys = {info.key for info in list_models()}
        assert "classical-emotion-2.5.0" in keys
        assert "classical-emotion-2.6.0" in keys

    def test_list_models_display_labels(self):
        by_key = {info.key: info for info in list_models()}
        assert by_key["classical-emotion-2.5.0"].display_label == "classical-emotion 2.5.0"
        assert by_key["classical-emotion-2.5.0"].version == "2.5.0"
        assert by_key["classical-emotion-2.6.0"].display_label == "classical-emotion 2.6.0"
        assert by_key["classical-emotion-2.6.0"].version == "2.6.0"

    def test_alias_resolves_to_2_5_0(self):
        info = resolve_model(DEFAULT_MODEL)
        assert info.key == DEFAULT_RELEASE_KEY
        assert info.version == "2.5.0"
        assert canonicalize_model_key(None) == DEFAULT_RELEASE_KEY
        assert canonicalize_model_key("classical-emotion") == DEFAULT_RELEASE_KEY

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown emotion model"):
            get_model("not-a-real-model")
        available = available_model_keys()
        assert DEFAULT_MODEL in available
        assert "classical-emotion-2.5.0" in available
        assert "classical-emotion-2.6.0" in available


class TestPreprocessSpec:
    def test_2_5_0_falls_back_to_15_40_no_notch(self):
        spec = load_preprocess_spec(RELEASE_25)
        assert spec.used_fallback is True
        assert spec.bandpass_low == 15.0
        assert spec.bandpass_high == 40.0
        assert spec.notch_hz is None

    def test_2_6_0_uses_notch_and_20_110(self):
        spec = load_preprocess_spec(RELEASE_26)
        assert spec.used_fallback is False
        assert spec.bandpass_low == 20.0
        assert spec.bandpass_high == 110.0
        assert spec.notch_hz == 60.0

    def test_missing_config_falls_back(self, tmp_path):
        spec = load_preprocess_spec(tmp_path)
        assert spec.used_fallback is True
        assert spec.bandpass_low == 15.0
        assert spec.notch_hz is None


class TestStreamingNotch:
    def test_notch_changes_filtered_windows(self):
        rng = np.random.default_rng(0)
        n = 250 * 6
        t = np.arange(n, dtype=np.float64) / 250.0
        eeg = rng.normal(scale=1.0, size=(n, 8))
        eeg = eeg + 20.0 * np.sin(2 * np.pi * 60.0 * t)[:, None]
        ts = t

        src_default = StreamingWindowSource(eeg, timestamps=ts)
        src_26 = StreamingWindowSource(
            eeg,
            timestamps=ts,
            bandpass_low=20.0,
            bandpass_high=110.0,
            notch_hz=60.0,
        )
        assert src_default.notch_hz is None
        assert src_26.notch_hz == 60.0
        win_default = next(iter(src_default)).samples
        win_26 = next(iter(src_26)).samples
        assert win_default is not None and win_26 is not None
        assert win_default.shape == (8, 500)
        assert win_26.shape == (8, 500)
        assert not np.allclose(win_default, win_26)


class TestSkipLogic:
    def _write_info(self, folder: Path, registry_key: str, version: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model_info.json").write_text(
            json.dumps(
                {
                    "model_name": "classical-emotion",
                    "model_version": version,
                    "registry_key": registry_key,
                }
            ),
            encoding="utf-8",
        )

    def test_legacy_alias_matches_default(self, tmp_path):
        self._write_info(tmp_path, "classical-emotion", "2.5.0")
        assert outputs_match_requested_model(str(tmp_path), None)
        assert outputs_match_requested_model(str(tmp_path), "classical-emotion")
        assert outputs_match_requested_model(str(tmp_path), "classical-emotion-2.5.0")

    def test_missing_info_treated_as_2_5_0(self, tmp_path):
        assert outputs_match_requested_model(str(tmp_path), "classical-emotion")
        assert not outputs_match_requested_model(str(tmp_path), "classical-emotion-2.6.0")

    def test_mismatch_triggers_recompute(self, tmp_path):
        self._write_info(tmp_path, "classical-emotion-2.5.0", "2.5.0")
        assert not outputs_match_requested_model(str(tmp_path), "classical-emotion-2.6.0")

    def test_matching_2_6_0_skips(self, tmp_path):
        self._write_info(tmp_path, "classical-emotion-2.6.0", "2.6.0")
        assert outputs_match_requested_model(str(tmp_path), "classical-emotion-2.6.0")
        assert not outputs_match_requested_model(str(tmp_path), "classical-emotion")

    def test_unreadable_info_is_mismatch(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "model_info.json").write_text("{not-json", encoding="utf-8")
        assert not outputs_match_requested_model(str(tmp_path), "classical-emotion")


@pytest.mark.skipif(
    not (RELEASE_25 / "pipeline.joblib").exists()
    or not (RELEASE_26 / "pipeline.joblib").exists(),
    reason="classical-emotion 2.5.0/2.6.0 pipeline.joblib not found",
)
class TestImportIsolation:
    def test_load_2_6_after_2_5_uses_2_6_package(self):
        model_25 = get_model("classical-emotion-2.5.0")
        assert model_25.model_version == "2.5.0"
        assert model_25.preprocess_spec.bandpass_high == 40.0
        assert model_25.emotions_csv_uses_argmax is False

        model_26 = get_model("classical-emotion-2.6.0")
        assert model_26.model_version == "2.6.0"
        assert model_26.preprocess_spec.notch_hz == 60.0
        assert model_26.preprocess_spec.bandpass_high == 110.0
        assert model_26.emotions_csv_uses_argmax is True

        import exg_emotion

        assert "classical-emotion-2.6.0" in Path(exg_emotion.__file__).as_posix()
