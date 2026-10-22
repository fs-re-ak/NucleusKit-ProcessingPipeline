"""Process page layout (no run summary) and Playback series filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from nucleuskit_toolkit.ui import plot_style  # noqa: E402
from nucleuskit_toolkit.ui.theme import apply_theme  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    apply_theme(instance, "dark")
    return instance


def test_process_page_has_no_run_summary_and_run_sits_in_step_2(app) -> None:
    from nucleuskit_toolkit.ui.pages.offline_page import OfflinePage

    page = OfflinePage()
    assert not hasattr(page, "_summary_card")
    assert page._run.parent() is page._options_card or page._options_card.isAncestorOf(page._run)
    assert not page._run.isEnabled()


def test_running_keeps_cancel_clickable(app) -> None:
    from nucleuskit_toolkit.ui.pages.offline_page import OfflinePage

    page = OfflinePage()
    page._set_processing(True)
    assert page._run.isEnabled()
    assert not page._opt_nlms.isEnabled() and not page._emotion_model.isEnabled()
    page._set_processing(False)
    assert page._opt_nlms.isEnabled()


def test_playback_shows_only_series_with_data(app, tmp_path) -> None:
    from nucleuskit_toolkit.ui.pages.playback_page import MetricsPlotWidget

    t = np.arange(0, 10, 0.5)
    df = pd.DataFrame({"Timestamp": t, "Neutral": 0.5, "Happiness": 0.3, "Anger": 0.0,
                       "Surprise": np.nan, "Contempt": 0.0, "Disgust": 0.0, "Fear": 0.2, "Sadness": 0.0})
    path = tmp_path / "Emotions.csv"
    df.to_csv(path, index=False)

    widget = MetricsPlotWidget()
    assert widget.load_csv(str(path)) is None
    shown = [name for name, _cb, _filled in widget._series]
    assert shown == ["Neutral", "Happiness", "Fear"]
    for curve, name in zip(widget._curves, shown):
        assert curve.opts["pen"].color().name().upper() == plot_style.DARK.emotion(name)


def test_playback_reports_a_file_without_data(app, tmp_path) -> None:
    from nucleuskit_toolkit.ui.pages.playback_page import MetricsPlotWidget

    path = tmp_path / "Emotions.csv"
    pd.DataFrame({"Timestamp": [0.0, 0.5], "Neutral": 0.0, "Happiness": 0.0}).to_csv(path, index=False)
    assert MetricsPlotWidget().load_csv(str(path)) == "Emotions.csv has no data to plot."
