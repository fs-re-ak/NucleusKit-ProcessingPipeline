"""Theme-aware styling for the pyqtgraph plots (real-time, playback, tools).

Plots sit on the card surface of the current theme, with muted axes, a faint grid and text in text colours (never a
series colour). Series colours come from fixed, ordered sets chosen for each theme and checked for contrast against
the plot surface and for separation under colour-vision deficiency:

- ``xyz``: X / Y / Z traces (motion). Red / teal / amber: red-amber is in the colour-blind "secondary encoding"
  band, so these plots always show a legend.
- ``emotions``: one colour per name of ``EMOTION_ORDER`` (emotions and affective metrics, fixed order, never
  cycled), so a series keeps its colour whatever else the file contains. Dark uses the product palette as given;
  light darkens the pale ones (Happiness, Fear, Surprise, Attention, Contempt) to 3:1 on white, same hue.
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

EMOTION_ORDER = (
    "Neutral", "Happiness", "Anger", "Surprise", "Contempt", "Disgust", "Fear", "Sadness",
    "Love", "Distress", "Attention", "Cognitive", "Engagement",
)


@dataclass(frozen=True)
class PlotColors:
    background: str
    axis: str
    text: str
    ink: str
    accent: str
    grid_alpha: float
    xyz: tuple[str, str, str]
    emotions: tuple[str, ...]  # aligned with EMOTION_ORDER
    pair: tuple[str, str]

    @property
    def neutral(self) -> str:
        return self.emotion("Neutral")

    def emotion(self, name: str) -> str:
        """Colour of a named emotion or metric; unknown names get the Neutral grey."""
        return self.emotions[EMOTION_ORDER.index(name if name in EMOTION_ORDER else "Neutral")]


LIGHT = PlotColors(
    background="#FFFFFF",
    axis="#5B6579",
    text="#10131C",
    ink="#1F2633",
    accent="#4F85EE",
    grid_alpha=0.10,
    xyz=("#e34948", "#1baf7a", "#eda100"),
    #        Neutral    Happiness  Anger      Surprise   Contempt   Disgust    Fear       Sadness
    emotions=("#7A8391", "#A99300", "#C85263", "#8A91B9", "#239ADE", "#6E8551", "#7D94B8", "#2F67B8",
              # Love    Distress   Attention  Cognitive  Engagement
              "#E35F8A", "#6B352A", "#D87A4D", "#AF68A4", "#A73A41"),
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
    #        Neutral    Happiness  Anger      Surprise   Contempt   Disgust    Fear       Sadness
    emotions=("#7A8391", "#D9C34A", "#C85263", "#B7BEE7", "#2A9FE3", "#6E8551", "#C7D4E9", "#2F67B8",
              # Love    Distress   Attention  Cognitive  Engagement
              "#E35F8A", "#6B352A", "#E68659", "#AF68A4", "#A73A41"),
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
