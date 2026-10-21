"""Theme-aware styling for the pyqtgraph plots (real-time, playback, tools).

Plots sit on the card surface of the current theme, with muted axes, a faint grid and text in text colours (never a
series colour). Series colours come from fixed, ordered sets chosen for each theme and checked for contrast against
the plot surface and for separation under colour-vision deficiency:

- ``xyz``: X / Y / Z traces (motion). Red / teal / amber: red-amber is in the colour-blind "secondary encoding"
  band, so these plots always show a legend.
- ``emotions``: the seven emotions in ``EMOTION_ORDER`` (fixed order, never cycled); Neutral is a neutral grey.
- ``pair``: two compared groups (blue / red).
- ``ink``: a single colour for stacked signals identified by their row label (EEG channels).

Each plot widget calls the helpers below when it is built and again from its ``refresh_theme()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import pyqtgraph as pg
from PySide6.QtGui import QColor

from nucleuskit_toolkit.ui.theme import current_palette

EMOTION_ORDER = ("Happiness", "Anger", "Surprise", "Contempt", "Disgust", "Fear", "Sadness")


@dataclass(frozen=True)
class PlotColors:
    background: str
    axis: str
    text: str
    ink: str
    accent: str
    grid_alpha: float
    xyz: tuple[str, str, str]
    emotions: tuple[str, ...]
    neutral: str
    pair: tuple[str, str]

    def emotion(self, name: str) -> str:
        if name in EMOTION_ORDER:
            return self.emotions[EMOTION_ORDER.index(name)]
        return self.neutral


LIGHT = PlotColors(
    background="#FFFFFF",
    axis="#5B6579",
    text="#10131C",
    ink="#1F2633",
    accent="#4F85EE",
    grid_alpha=0.10,
    xyz=("#e34948", "#1baf7a", "#eda100"),
    emotions=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"),
    neutral="#8A93A6",
    pair=("#2a78d6", "#e34948"),
)

DARK = PlotColors(
    background="#1C2230",
    axis="#AAB6CC",
    text="#EEF2F8",
    ink="#DCE3EE",
    accent="#4F85EE",
    grid_alpha=0.12,
    xyz=("#e5484d", "#199e70", "#c98500"),
    emotions=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
    neutral="#8D97AB",
    pair=("#3987e5", "#e66767"),
)


def plot_colors() -> PlotColors:
    """Plot colours of the theme applied last."""
    return DARK if current_palette().name == "dark" else LIGHT


def pen(color: str, width: float = 1.5) -> pg.QtGui.QPen:
    p = pg.mkPen(color, width=width)
    p.setCosmetic(True)
    return p


def translucent(color: str, alpha: int) -> QColor:
    c = QColor(color)
    c.setAlpha(alpha)
    return c


def style_view(view) -> None:  # type: ignore[no-untyped-def]
    """Background of a GraphicsLayoutWidget / PlotWidget."""
    view.setBackground(plot_colors().background)


def style_plot(plot: pg.PlotItem, *, grid: bool = True) -> None:
    """Axes, tick labels, grid and title of one PlotItem in the current theme."""
    c = plot_colors()
    for name in ("left", "bottom", "right", "top"):
        axis = plot.getAxis(name)
        axis.setPen(pen(c.axis, 1))
        axis.setTextPen(c.axis)
        axis.setTickPen(pen(c.axis, 1))
        label = axis.labelText
        if label:
            axis.setLabel(label, units=axis.labelUnits, color=c.axis)
    plot.showGrid(x=grid, y=grid, alpha=c.grid_alpha)
    title = plot.titleLabel.text
    if title:
        set_title(plot, title)
    legend = plot.legend
    if legend is not None:
        style_legend(legend)


def lock_view(plot: pg.PlotItem) -> None:
    """No zoom, pan or context menu: the view keeps its default scale (real-time plots)."""
    plot.setMenuEnabled(False)
    plot.hideButtons()
    plot.getViewBox().setMouseEnabled(x=False, y=False)


def set_title(plot: pg.PlotItem, text: str) -> None:
    plot.setTitle(text, color=plot_colors().text, size="11pt")


def set_label(plot: pg.PlotItem, axis: str, text: str, units: str | None = None) -> None:
    plot.setLabel(axis, text, units=units, color=plot_colors().axis)


def add_legend(plot: pg.PlotItem) -> pg.LegendItem:
    legend = plot.addLegend(offset=(10, 10))
    style_legend(legend)
    return legend


def style_legend(legend: pg.LegendItem) -> None:
    c = plot_colors()
    legend.setLabelTextColor(c.text)
    legend.setBrush(translucent(c.background, 210))
    legend.setPen(pen(c.axis, 0.5))
