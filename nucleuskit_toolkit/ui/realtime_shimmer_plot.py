"""Real-time Shimmer wristband plot (PySide6 + pyqtgraph): IMU stack or single EDA/PPG."""

from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from nucleuskit_toolkit.shimmer.constants import ShimmerConstants
from nucleuskit_toolkit.ui.plot_style import add_legend, pen, plot_colors, set_label, set_title, style_plot, style_view

# Combo box display string (user-requested exact label)
SHIMMER_COMBO_LABEL_IMU = "IMU (x,y,z)"


class RealtimeShimmerPlot(QWidget):
    """
    Scrolls Shimmer streams: combo selects IMU (x,y,z with vertical offset), EDA, or PPG.
    Y-axis autoscales to visible data.
    """

    def __init__(self, parent: QWidget | None = None, window_seconds: float = 5.0) -> None:
        super().__init__(parent)
        self._fs = float(ShimmerConstants.SAMPLING_RATE)
        self._buf_len = max(32, int(self._fs * window_seconds))
        self._imu_offsets = np.array([0.0, 100.0, 200.0], dtype=np.float64)

        self._combo = QComboBox()
        self._combo.addItems([SHIMMER_COMBO_LABEL_IMU, "EDA", "PPG"])
        self._combo.currentIndexChanged.connect(self._on_mode_changed)

        top = QHBoxLayout()
        top.addWidget(QLabel("Display:"))
        top.addWidget(self._combo, stretch=1)

        self._win = pg.GraphicsLayoutWidget(title="Real-time Shimmer")
        self._plot = self._win.addPlot()
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        set_label(self._plot, "bottom", "Time", units="s")

        self._legend = add_legend(self._plot)
        self._curves_imu = [self._plot.plot(name=axis, antialias=True) for axis in "XYZ"]
        self._curve_scalar = self._plot.plot(antialias=True)

        self._buf_imu = [np.zeros(self._buf_len) for _ in range(3)]
        self._buf_scalar = np.zeros(self._buf_len)
        # Seconds before now of each buffer sample (the newest sample is at 0).
        self._time = (np.arange(self._buf_len) - (self._buf_len - 1)) / self._fs
        self.refresh_theme()

        self._plot.enableAutoRange(axis="y", enable=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self._win, stretch=1)

        self._on_mode_changed(self._combo.currentIndex())

    def refresh_theme(self) -> None:
        """Re-apply the plot colours of the current theme."""
        c = plot_colors()
        style_view(self._win)
        style_plot(self._plot)
        for curve, color in zip(self._curves_imu, c.xyz):
            curve.setPen(pen(color, 2))
        self._curve_scalar.setPen(pen(c.accent, 2))

    def _on_mode_changed(self, index: int) -> None:
        if index == 0:
            set_title(self._plot, SHIMMER_COMBO_LABEL_IMU)
            set_label(self._plot, "left", "IMU (offset stacked)")
            for c in self._curves_imu:
                c.setVisible(True)
            self._curve_scalar.setVisible(False)
        elif index == 1:
            set_title(self._plot, "EDA")
            set_label(self._plot, "left", "EDA")
            for c in self._curves_imu:
                c.setVisible(False)
            self._curve_scalar.setVisible(True)
        else:
            set_title(self._plot, "PPG")
            set_label(self._plot, "left", "PPG")
            for c in self._curves_imu:
                c.setVisible(False)
            self._curve_scalar.setVisible(True)
        self._legend.setVisible(index == 0)
        self._autorange_y()

    def _autorange_y(self) -> None:
        visible = [c for c in self._plot.listDataItems() if c.isVisible()]
        if visible:
            self._plot.enableAutoRange(axis="y", enable=True)
            self._plot.autoRange(items=visible)

    def clear_buffers(self) -> None:
        for j in range(3):
            self._buf_imu[j].fill(0.0)
            self._curves_imu[j].setData(self._time, self._buf_imu[j] + self._imu_offsets[j])
        self._buf_scalar.fill(0.0)
        self._curve_scalar.setData(self._time, self._buf_scalar)
        self._autorange_y()

    def add_sample(self, imu_x: float, imu_y: float, imu_z: float, eda: float, ppg: float) -> None:
        """Append one sample row (decoded numeric values)."""
        mode = self._combo.currentIndex()

        if mode == 0:
            vals = (float(imu_x), float(imu_y), float(imu_z))
            for j in range(3):
                buf = self._buf_imu[j]
                buf[:] = np.roll(buf, -1)
                buf[-1] = vals[j]
                self._curves_imu[j].setData(self._time, buf + self._imu_offsets[j])
        else:
            buf = self._buf_scalar
            buf[:] = np.roll(buf, -1)
            buf[-1] = float(eda if mode == 1 else ppg)
            self._curve_scalar.setData(self._time, buf)

        self._autorange_y()
