"""Real-time Hermes 9-axis motion (PySide6 + pyqtgraph): accel / gyro / compass groups."""

from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

pg.setConfigOptions(foreground="k")

LABEL_ACCEL = "Accelerometer (x,y,z)"
LABEL_GYRO = "Gyroscope (x,y,z)"
LABEL_COMPASS = "Compass (x,y,z)"

# Match EEG scrolling window length (250 Hz * 5 s); motion notify rate varies.
BUF_LEN = 250 * 5


class RealtimeHermesMotionPlot(QWidget):
    """
    Scrolls Hermes motion: combo selects accelerometer, gyroscope, or compass (magnetometer).
    Three x/y/z traces with vertical offset; y-axis autoscales.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buf_len = BUF_LEN
        self._offsets = np.array([0.0, 100.0, 200.0], dtype=np.float64)

        self._buf_acc = [np.zeros(self._buf_len) for _ in range(3)]
        self._buf_gyr = [np.zeros(self._buf_len) for _ in range(3)]
        self._buf_mag = [np.zeros(self._buf_len) for _ in range(3)]

        self._combo = QComboBox()
        self._combo.addItems([LABEL_ACCEL, LABEL_GYRO, LABEL_COMPASS])
        self._combo.currentIndexChanged.connect(self._on_mode_changed)

        top = QHBoxLayout()
        top.addWidget(QLabel("Motion:"))
        top.addWidget(self._combo, stretch=1)

        self._win = pg.GraphicsLayoutWidget(title="Hermes motion")
        self._plot = self._win.addPlot(title=LABEL_ACCEL)
        self._plot.showGrid(x=True, y=True)
        self._plot.setLabel("bottom", "Time (samples)")

        pens = [pg.intColor(i, hues=3, values=2) for i in range(3)]
        self._curves = [self._plot.plot(pen=pens[i]) for i in range(3)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self._win, stretch=1)

        self._on_mode_changed(self._combo.currentIndex())

    def _active_buffers(self) -> list[np.ndarray]:
        idx = self._combo.currentIndex()
        if idx == 0:
            return self._buf_acc
        if idx == 1:
            return self._buf_gyr
        return self._buf_mag

    def _on_mode_changed(self, index: int) -> None:
        if index == 0:
            self._plot.setTitle(LABEL_ACCEL)
            self._plot.setLabel("left", "Acceleration (g)")
        elif index == 1:
            self._plot.setTitle(LABEL_GYRO)
            self._plot.setLabel("left", "Angular rate (dps)")
        else:
            self._plot.setTitle(LABEL_COMPASS)
            self._plot.setLabel("left", "Magnetometer (gauss)")
        self._redraw_active()
        self._autorange_y()

    def _redraw_active(self) -> None:
        bufs = self._active_buffers()
        for j in range(3):
            self._curves[j].setData(bufs[j] + self._offsets[j])

    def _autorange_y(self) -> None:
        visible = [c for c in self._plot.listDataItems() if c.isVisible()]
        if visible:
            self._plot.enableAutoRange(axis="y", enable=True)
            self._plot.autoRange(items=visible)

    def clear_buffers(self) -> None:
        for group in (self._buf_acc, self._buf_gyr, self._buf_mag):
            for j in range(3):
                group[j].fill(0.0)
        self._redraw_active()
        self._autorange_y()

    def add_motion_sample(
        self,
        ax: float,
        ay: float,
        az: float,
        gx: float,
        gy: float,
        gz: float,
        cx: float,
        cy: float,
        cz: float,
    ) -> None:
        """Append one motion sample (same units as HermesBleProxy.motion_process output)."""
        vals_acc = (float(ax), float(ay), float(az))
        vals_gyr = (float(gx), float(gy), float(gz))
        vals_mag = (float(cx), float(cy), float(cz))

        for j in range(3):
            for buf, val in (
                (self._buf_acc[j], vals_acc[j]),
                (self._buf_gyr[j], vals_gyr[j]),
                (self._buf_mag[j], vals_mag[j]),
            ):
                buf[:] = np.roll(buf, -1)
                buf[-1] = val

        bufs = self._active_buffers()
        for j in range(3):
            b = bufs[j]
            self._curves[j].setData(b + self._offsets[j])
        self._autorange_y()
