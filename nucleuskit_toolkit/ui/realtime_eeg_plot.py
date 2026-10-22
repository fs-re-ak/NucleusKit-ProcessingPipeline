"""Real-time EEG plot (PySide6 + pyqtgraph), adapted from the legacy example visualizer.

Stacked display: one row per channel in a single ink colour, channel names on the left axis, time in seconds
(the last ``window`` seconds, 0 = now) and a 100 µV scale bar. Colours follow the app theme (``plot_style``).
"""

from __future__ import annotations

import os
import threading

# Prefer PySide6 before pyqtgraph loads a different Qt binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget
from reak_core.eeg.nlms import (
    CausalNLMSFilter,
    build_default_predictor_mask,
)
from scipy.signal import butter, lfilter, lfilter_zi

from nucleuskit_toolkit.hermes.constants import HermesConstants
from nucleuskit_toolkit.ui.plot_style import lock_view, pen, plot_colors, set_label, style_plot, style_view

ROW_SPACING_UV = 100.0
SCALE_BAR_UV = 100.0
WINDOW_S = 5.0


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
        self._nlms_filter: CausalNLMSFilter | None = None
        self._nlms_predictor_mask = build_default_predictor_mask(nb_channels)

        self._win = pg.GraphicsLayoutWidget(title="Real-time EEG")
        self._plot = self._win.addPlot()
        lock_view(self._plot)

        buffer_size = int(sampling_rate * WINDOW_S)
        self._data_buffers = [np.zeros(buffer_size) for _ in range(nb_channels)]
        # Seconds before now of each buffer sample (the newest sample is at 0).
        self._time = (np.arange(buffer_size) - (buffer_size - 1)) / sampling_rate
        self._curves = [self._plot.plot(antialias=True) for _ in range(nb_channels)]
        for curve in self._curves:
            curve.setClipToView(True)

        # Top-to-bottom on screen: AF8 … NOSE (index 0 at top, index 7 at bottom).
        self._channel_offsets = [
            ROW_SPACING_UV * (nb_channels - 1 - j) for j in range(nb_channels)
        ]
        channel_names = HermesConstants.CHANNEL_NAMES[:nb_channels]
        self._plot.getAxis("left").setTicks(
            [[(self._channel_offsets[j], name) for j, name in enumerate(channel_names)], []]
        )
        self._plot.getAxis("left").setWidth(64)
        set_label(self._plot, "bottom", "Time", units="s")
        self._plot.setXRange(-WINDOW_S, 0.55, padding=0)
        self._plot.setLimits(xMin=-WINDOW_S, xMax=0.55)

        # Scale bar on the right of "now", spanning the bottom row.
        bar_x = 0.18
        self._scale_bar = self._plot.plot(
            [bar_x, bar_x], [-SCALE_BAR_UV / 2, SCALE_BAR_UV / 2], antialias=True
        )
        self._scale_label = pg.TextItem(f"{SCALE_BAR_UV:.0f} µV", anchor=(0.0, 0.5))
        self._scale_label.setPos(bar_x + 0.05, 0.0)
        self._plot.addItem(self._scale_label)
        self.refresh_theme()

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

    def refresh_theme(self) -> None:
        """Re-apply the plot colours of the current theme."""
        c = plot_colors()
        style_view(self._win)
        style_plot(self._plot, grid=True)
        self._plot.showGrid(x=True, y=False, alpha=c.grid_alpha)
        for curve in self._curves:
            curve.setPen(pen(c.ink, 1.2))
        self._scale_bar.setPen(pen(c.text, 2))
        self._scale_label.setColor(c.axis)

    def set_ear_r_re_reference(self, enabled: bool) -> None:
        """Subtract EAR_R/2 from all channels before filtering (matches offline loadEXG re_reference)."""
        self._ear_r_re_reference = bool(enabled)
        if self._apply_filter and self._b is not None and self._a is not None:
            self._filter_states = [lfilter_zi(self._b, self._a) * 0 for _ in range(self._nb_channels)]

    def set_nlms_filter(self, enabled: bool) -> None:
        """Enable or disable the causal NLMS adaptive decorrelation filter.

        Enabling creates a fresh CausalNLMSFilter and resets the bandpass
        filter states so the transition is clean.  Disabling resets and
        discards the filter instance.
        """
        if enabled:
            self._nlms_filter = CausalNLMSFilter(
                n_channels=self._nb_channels,
                fs=self._sampling_rate,
                predictor_mask=self._nlms_predictor_mask.copy(),
            )
        else:
            if self._nlms_filter is not None:
                self._nlms_filter.reset()
            self._nlms_filter = None
        if self._apply_filter and self._b is not None and self._a is not None:
            self._filter_states = [lfilter_zi(self._b, self._a) * 0 for _ in range(self._nb_channels)]

    def nlms_predictor_mask(self) -> np.ndarray:
        """Return a copy of the current NLMS predictor connectivity mask."""
        return self._nlms_predictor_mask.copy()

    def set_nlms_predictor_mask(self, mask: np.ndarray) -> None:
        """Update predictor connectivity; restart NLMS if currently enabled."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self._nb_channels, self._nb_channels):
            raise ValueError(
                f"predictor mask must be ({self._nb_channels}, {self._nb_channels}), "
                f"got {mask.shape}"
            )
        self._nlms_predictor_mask = mask.copy()
        if self._nlms_filter is not None:
            self.set_nlms_filter(True)

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
            self._curves[j].setData(self._time, self._data_buffers[j] + self._channel_offsets[j])

    def add_samples(self, samples: np.ndarray) -> None:
        """Append rows of shape (n, nb_channels) to the scrolling buffers.

        Processing is split into three sequential passes so that NLMS (a
        cross-channel operation) can be inserted cleanly between the per-channel
        bandpass and the display buffer update:

          Pass 1 — optional EAR_R re-reference + per-channel bandpass
          Pass 2 — optional NLMS adaptive decorrelation (cross-channel)
          Pass 3 — scroll display buffers and repaint curves
        """
        if not isinstance(samples, np.ndarray) or samples.ndim != 2 or samples.shape[1] != self._nb_channels:
            return

        n = samples.shape[0]
        batch = np.array(samples, dtype=np.float64)

        # Pass 1: re-reference then bandpass per channel
        if self._ear_r_re_reference:
            ref = batch[:, self._ear_r_ch : self._ear_r_ch + 1] / 2.0
            batch -= ref

        filtered = np.empty_like(batch)
        for j in range(self._nb_channels):
            col = batch[:, j]
            if self._apply_filter and self._b is not None and self._a is not None:
                col, self._filter_states[j] = lfilter(self._b, self._a, col, zi=self._filter_states[j])
            filtered[:, j] = col

        # Pass 2: NLMS adaptive decorrelation (optional, cross-channel)
        if self._nlms_filter is not None:
            filtered = self._nlms_filter.push_batch(filtered)

        # Pass 3: update scrolling display buffers
        buf_len = len(self._data_buffers[0])
        for j in range(self._nb_channels):
            col = filtered[:, j]
            buf = self._data_buffers[j]
            if n >= buf_len:
                buf[:] = col[-buf_len:]
            else:
                buf[:-n] = buf[n:]
                buf[-n:] = col
            self._curves[j].setData(self._time, buf + self._channel_offsets[j])
