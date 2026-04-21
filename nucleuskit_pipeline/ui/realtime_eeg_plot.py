"""Real-time EEG plot (PySide6 + pyqtgraph), adapted from the legacy example visualizer."""

from __future__ import annotations

import os

# Prefer PySide6 before pyqtgraph loads a different Qt binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget
from scipy.signal import butter, lfilter, lfilter_zi

from nucleuskit_pipeline.hermes.HermesConstants import HermesConstants

pg.setConfigOptions(foreground="k")


def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")


class RealtimeEegPlot(QWidget):
    """Scrolls multi-channel EEG with optional bandpass (same defaults as the legacy example)."""

    def __init__(
        self,
        parent: QWidget | None = None,
        nb_channels: int = 8,
        sampling_rate: float = 250.0,
        apply_filter: bool = True,
        lowcut: float = 15.0,
        highcut: float = 30.0,
        order: int = 2,
    ) -> None:
        super().__init__(parent)
        self._nb_channels = nb_channels
        self._sampling_rate = sampling_rate
        self._apply_filter = apply_filter
        self._ear_r_re_reference = False
        self._ear_r_ch = HermesConstants.CHANNELS["EAR_R"]

        self._win = pg.GraphicsLayoutWidget(title="Real-time EEG")
        self._plot = self._win.addPlot(title="EEG Channels")
        self._plot.showGrid(x=True, y=True)
        self._plot.setLabel("left", "Amplitude (uV)")
        self._plot.setLabel("bottom", "Time (samples)")

        self._curves = [self._plot.plot(pen=pg.intColor(i)) for i in range(nb_channels)]

        buffer_size = int(sampling_rate * 5)
        self._data_buffers = [np.zeros(buffer_size) for _ in range(nb_channels)]
        self._channel_offsets = [100.0 * i for i in range(nb_channels)]

        if apply_filter:
            b, a = _butter_bandpass(lowcut, highcut, fs=sampling_rate, order=order)
            self._b, self._a = b, a
            self._filter_states = [lfilter_zi(b, a) * 0 for _ in range(nb_channels)]
        else:
            self._b = self._a = None
            self._filter_states = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._win, stretch=1)

    def set_ear_r_re_reference(self, enabled: bool) -> None:
        """Subtract EAR_R/2 from all channels before filtering (matches offline loadEXG re_reference)."""
        self._ear_r_re_reference = bool(enabled)
        if self._apply_filter and self._b is not None and self._a is not None:
            self._filter_states = [lfilter_zi(self._b, self._a) * 0 for _ in range(self._nb_channels)]

    def clear_buffers(self) -> None:
        for j in range(self._nb_channels):
            self._data_buffers[j].fill(0.0)
            self._curves[j].setData(self._data_buffers[j] + self._channel_offsets[j])

    def add_samples(self, samples: np.ndarray) -> None:
        """Append rows of shape (n, nb_channels) to the scrolling buffers."""
        if not isinstance(samples, np.ndarray) or samples.ndim != 2 or samples.shape[1] != self._nb_channels:
            return

        for i in range(samples.shape[0]):
            sample = np.array(samples[i], dtype=np.float64, copy=True)
            if self._ear_r_re_reference:
                ref_signal = sample[self._ear_r_ch] / 2.0
                sample -= ref_signal

            if self._apply_filter and self._b is not None and self._a is not None:
                filtered_sample = np.zeros(self._nb_channels)
                for j in range(self._nb_channels):
                    y, self._filter_states[j] = lfilter(
                        self._b, self._a, [float(sample[j])], zi=self._filter_states[j]
                    )
                    filtered_sample[j] = float(y[0])
            else:
                filtered_sample = sample

            for j in range(self._nb_channels):
                buf = self._data_buffers[j]
                buf[:] = np.roll(buf, -1)
                buf[-1] = filtered_sample[j]
                self._curves[j].setData(buf + self._channel_offsets[j])
