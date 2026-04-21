"""Real-time EEG plot (PySide6 + pyqtgraph), adapted from the legacy example visualizer."""

from __future__ import annotations

import os
import threading

# Prefer PySide6 before pyqtgraph loads a different Qt binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
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

        # Thread-safe accumulation buffer: BLE worker threads write here; the
        # QTimer flushes it on the GUI thread at a fixed display rate (~30 FPS).
        # This breaks the Qt event-queue backpressure that builds up when
        # add_samples is connected via a queued signal to a high-rate producer.
        self._pending_lock = threading.Lock()
        self._pending_batches: list[np.ndarray] = []

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(33)  # ~30 FPS
        self._flush_timer.timeout.connect(self._flush_pending)
        self._flush_timer.start()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._win, stretch=1)

    def set_ear_r_re_reference(self, enabled: bool) -> None:
        """Subtract EAR_R/2 from all channels before filtering (matches offline loadEXG re_reference)."""
        self._ear_r_re_reference = bool(enabled)
        if self._apply_filter and self._b is not None and self._a is not None:
            self._filter_states = [lfilter_zi(self._b, self._a) * 0 for _ in range(self._nb_channels)]

    def enqueue_samples(self, samples: np.ndarray) -> None:
        """Thread-safe entry point for producer threads (BLE worker, etc.).

        Accumulates batches without touching Qt objects.  The QTimer calls
        _flush_pending() on the GUI thread to drain and render them.
        """
        if not isinstance(samples, np.ndarray) or samples.ndim != 2 or samples.shape[1] != self._nb_channels:
            return
        with self._pending_lock:
            self._pending_batches.append(samples)

    def _flush_pending(self) -> None:
        """Called on the GUI thread at ~30 Hz.  Drains all pending batches in one render pass."""
        with self._pending_lock:
            if not self._pending_batches:
                return
            batches = self._pending_batches
            self._pending_batches = []

        combined = np.concatenate(batches, axis=0)
        self.add_samples(combined)

    def clear_buffers(self) -> None:
        for j in range(self._nb_channels):
            self._data_buffers[j].fill(0.0)
            self._curves[j].setData(self._data_buffers[j] + self._channel_offsets[j])

    def add_samples(self, samples: np.ndarray) -> None:
        """Append rows of shape (n, nb_channels) to the scrolling buffers.

        Processes the entire batch at once: one lfilter call per channel, one
        np.roll per channel, and one setData per channel — regardless of how
        many samples are in the batch.  This keeps the GUI thread load constant
        with respect to batch size and avoids the O(n*channels) paint events
        that the previous per-sample loop generated.
        """
        if not isinstance(samples, np.ndarray) or samples.ndim != 2 or samples.shape[1] != self._nb_channels:
            return

        n = samples.shape[0]
        batch = np.array(samples, dtype=np.float64)

        if self._ear_r_re_reference:
            ref = batch[:, self._ear_r_ch : self._ear_r_ch + 1] / 2.0
            batch -= ref

        buf_len = len(self._data_buffers[0])

        for j in range(self._nb_channels):
            col = batch[:, j]

            if self._apply_filter and self._b is not None and self._a is not None:
                col, self._filter_states[j] = lfilter(self._b, self._a, col, zi=self._filter_states[j])

            buf = self._data_buffers[j]
            if n >= buf_len:
                buf[:] = col[-buf_len:]
            else:
                buf[:-n] = buf[n:]
                buf[-n:] = col

            self._curves[j].setData(buf + self._channel_offsets[j])
