"""Theme-aware plot colours: palettes per theme and live recolouring of the real-time plots."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from nucleuskit_toolkit.ui import plot_style  # noqa: E402
from nucleuskit_toolkit.ui.theme import apply_theme  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_palettes_are_complete_and_distinct() -> None:
    for colors in (plot_style.LIGHT, plot_style.DARK):
        assert len(colors.emotions) == len(plot_style.EMOTION_ORDER) == len(set(colors.emotions))
        assert len(set(colors.xyz)) == 3 and len(set(colors.pair)) == 2
        assert colors.neutral not in colors.emotions
        assert colors.emotion("Neutral") == colors.neutral
        assert colors.emotion("Fear") != colors.emotion("Contempt")


def test_plot_colors_follow_the_theme(app) -> None:
    apply_theme(app, "dark")
    assert plot_style.plot_colors() is plot_style.DARK
    apply_theme(app, "light")
    assert plot_style.plot_colors() is plot_style.LIGHT


def test_realtime_plots_recolour_on_theme_change(app) -> None:
    from nucleuskit_toolkit.ui.realtime_eeg_plot import RealtimeEegPlot
    from nucleuskit_toolkit.ui.realtime_hermes_motion_plot import RealtimeHermesMotionPlot

    apply_theme(app, "dark")
    eeg = RealtimeEegPlot()
    motion = RealtimeHermesMotionPlot()
    assert eeg._curves[0].opts["pen"].color() == QColor(plot_style.DARK.ink)

    apply_theme(app, "light")
    eeg.refresh_theme()
    motion.refresh_theme()
    assert eeg._curves[0].opts["pen"].color() == QColor(plot_style.LIGHT.ink)
    assert [c.opts["pen"].color() for c in motion._curves] == [QColor(x) for x in plot_style.LIGHT.xyz]
    assert eeg._win.backgroundBrush().color() == QColor(plot_style.LIGHT.background)


def test_realtime_plots_cannot_be_zoomed_or_panned(app) -> None:
    from nucleuskit_toolkit.ui.realtime_eeg_plot import RealtimeEegPlot
    from nucleuskit_toolkit.ui.realtime_hermes_motion_plot import RealtimeHermesMotionPlot
    from nucleuskit_toolkit.ui.realtime_shimmer_plot import RealtimeShimmerPlot

    for widget in (RealtimeEegPlot(), RealtimeHermesMotionPlot(), RealtimeShimmerPlot()):
        vb = widget._plot.getViewBox()
        assert vb.state["mouseEnabled"] == [False, False]
        assert vb.menu is None or not widget._plot.menuEnabled()
