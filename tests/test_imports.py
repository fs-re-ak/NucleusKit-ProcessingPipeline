"""Smoke test: package imports."""

from __future__ import annotations

import importlib

import pytest


def test_import_package_version() -> None:
    import nucleuskit_toolkit

    assert nucleuskit_toolkit.__version__


def test_import_main_main() -> None:
    from nucleuskit_toolkit.__main__ import main

    assert callable(main)


def test_import_pipeline() -> None:
    from nucleuskit_toolkit.pipeline import NucleusKitProcessingToolkit

    pipe = NucleusKitProcessingToolkit(None)
    assert pipe.processingSteps[:3] == ["prepare", "video_rotation", "metainfo"]
    assert pipe.processingSteps[-1] == "gps"


@pytest.mark.parametrize(
    "module",
    [
        "channel_fixer_page",
        "channel_gain_page",
        "eeg_regression_page",
        "offline_page",
        "playback_page",
        "ppg_fixer_page",
        "realtime_viewer_page",
        "revert_original_page",
    ],
)
def test_import_gui_pages(module: str) -> None:
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    importlib.import_module(f"nucleuskit_toolkit.ui.pages.{module}")
