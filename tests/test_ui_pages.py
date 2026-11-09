"""Process page layout (no run summary) and Playback series."""

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


@pytest.fixture
def page(app):
    """An OfflinePage whose inspector thread is stopped afterwards.

    The app stops it on aboutToQuit, which never fires in tests; a thread still running at exit makes Qt abort.
    """
    from nucleuskit_toolkit.ui.pages.offline_page import OfflinePage

    instance = OfflinePage()
    yield instance
    instance._stop_inspector()


def test_process_page_has_no_run_summary_and_run_sits_in_step_2(page) -> None:
    assert not hasattr(page, "_summary_card")
    assert page._run.parent() is page._options_card or page._options_card.isAncestorOf(page._run)
    assert not page._run.isEnabled()


def test_running_keeps_cancel_clickable(page) -> None:
    page._set_processing(True)
    assert page._run.isEnabled()
    assert not page._opt_nlms.isEnabled() and not page._emotion_model.isEnabled()
    page._set_processing(False)
    assert page._opt_nlms.isEnabled()


def test_playback_keeps_every_emotion_column(app, tmp_path) -> None:
    """All-zero and all-NaN columns are plotted too (flat line or gap), Neutral first, in their named colours."""
    from nucleuskit_toolkit.ui.pages.playback_page import MetricsPlotWidget
    from reak_core.models import EMOTION_COLUMNS

    t = np.arange(0, 10, 0.5)
    df = pd.DataFrame({"Timestamp": t, "Neutral": 0.5, "Happiness": 0.3, "Anger": 0.0,
                       "Surprise": np.nan, "Contempt": 0.0, "Disgust": 0.0, "Fear": 0.2, "Sadness": 0.0})
    path = tmp_path / "Emotions.csv"
    df.to_csv(path, index=False)

    widget = MetricsPlotWidget()
    assert widget.load_csv(str(path)) is None
    shown = [name for name, _cb, _filled in widget._series]
    assert shown == ["Neutral"] + [c for c in EMOTION_COLUMNS if c != "Neutral" and c in df.columns]
    for curve, name in zip(widget._curves, shown):
        assert curve.opts["pen"].color().name().upper() == plot_style.DARK.emotion(name)


def test_playback_loads_a_file_with_only_zeros(app, tmp_path) -> None:
    from nucleuskit_toolkit.ui.pages.playback_page import MetricsPlotWidget

    path = tmp_path / "Emotions.csv"
    pd.DataFrame({"Timestamp": [0.0, 0.5], "Neutral": 0.0, "Happiness": 0.0}).to_csv(path, index=False)
    widget = MetricsPlotWidget()
    assert widget.load_csv(str(path)) is None
    assert [name for name, _cb, _filled in widget._series] == ["Neutral", "Happiness"]
