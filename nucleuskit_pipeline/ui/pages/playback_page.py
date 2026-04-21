"""Session playback: optional video (rawData/video.mp4) + emotions plot (results/Emotions.csv)."""

from __future__ import annotations

import os
from pathlib import Path

# Prefer PySide6 before pyqtgraph loads a different Qt binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.hermes.processor.emotionsProcessor import EMOTION_COLUMNS

PLAYBACK_EMOTION_COLORS: dict[str, str] = {
    "Neutral": "#a0a0a0",
    "Anger": "#ab3852",
    "Happiness": "#e0ca3c",
    "Sadness": "#00798c",
    "Fear": "#0f3899",
    "Disgust": "#426d40",
    "Surprise": "#aba1f4",
    "Contempt": "#0f3899",
}

_Z_NEUTRAL = -10.0
_Z_EMOTION_BASE = 0.0
_Z_PLAYHEAD = 100.0


def _emotion_checkbox_stylesheet(hex_color: str) -> str:
    """QCheckBox: label + square indicator use the series color (acts as legend)."""
    return (
        "QCheckBox {"
        f" color: {hex_color};"
        " spacing: 6px;"
        " font-weight: 600;"
        "}"
        "QCheckBox::indicator {"
        " width: 14px;"
        " height: 14px;"
        " border-radius: 3px;"
        " border: 1px solid palette(mid);"
        "}"
        "QCheckBox::indicator:unchecked {"
        " background: palette(base);"
        "}"
        "QCheckBox::indicator:checked {"
        f" background: {hex_color};"
        f" border: 1px solid {hex_color};"
        "}"
    )


def _ordered_playback_columns(plot_cols: list[str]) -> list[str]:
    """Neutral first (drawn behind), then EMOTION_COLUMNS order, then any extras."""
    seen: set[str] = set()
    out: list[str] = []
    if "Neutral" in plot_cols:
        out.append("Neutral")
        seen.add("Neutral")
    for c in EMOTION_COLUMNS:
        if c in plot_cols and c not in seen:
            out.append(c)
            seen.add(c)
    for c in plot_cols:
        if c not in seen:
            out.append(c)
    return out


def _session_playback_paths(session_dir: str) -> tuple[str, str]:
    root = Path(session_dir).expanduser().resolve()
    video = root / "rawData" / "video.mp4"
    emotions = root / "results" / "Emotions.csv"
    return str(video), str(emotions)


def _playback_preflight(session_dir: str) -> str | None:
    _video, emotions = _session_playback_paths(session_dir)
    if not Path(emotions).is_file():
        return f"Missing emotions file:\n{emotions}"
    return None


def _session_has_video(session_dir: str) -> bool:
    video, _ = _session_playback_paths(session_dir)
    return Path(video).is_file()


class EmotionsPlotWidget(QWidget):
    """Filled probability areas vs time with a vertical playhead and per-series toggles."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._t_min = 0.0
        self._t_max = 1.0
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setLabel("left", "Probability")
        self._plot.setYRange(0.0, 1.0, padding=0.02)
        self._curves: list[pg.PlotDataItem] = []
        self._playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen("w", width=2))
        self._playhead.setZValue(_Z_PLAYHEAD)

        self._toggle_bar = QWidget()
        self._toggle_layout = QHBoxLayout(self._toggle_bar)
        self._toggle_layout.setContentsMargins(4, 4, 4, 0)
        self._toggle_layout.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, stretch=1)
        layout.addWidget(self._toggle_bar)

    def _clear_toggle_widgets(self) -> None:
        while self._toggle_layout.count():
            item = self._toggle_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def load_csv(self, path: str) -> str | None:
        """Load Emotions.csv; returns error message or None on success."""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return f"Could not read {path}:\n{e}"

        if "Timestamp" not in df.columns:
            return "Emotions.csv must contain a Timestamp column."

        for c in self._curves:
            self._plot.removeItem(c)
        self._curves.clear()
        self._clear_toggle_widgets()
        try:
            self._plot.removeItem(self._playhead)
        except Exception:
            pass
        leg = self._plot.plotItem.legend
        if leg is not None:
            try:
                scene = leg.scene()
                if scene is not None:
                    scene.removeItem(leg)
            except Exception:
                pass
            self._plot.plotItem.legend = None

        ts = df["Timestamp"].to_numpy(dtype=np.float64)
        if ts.size == 0:
            return "Emotions.csv has no rows."

        self._t_min = float(np.nanmin(ts))
        self._t_max = float(np.nanmax(ts))

        plot_cols = [c for c in EMOTION_COLUMNS if c in df.columns]
        if not plot_cols:
            plot_cols = [c for c in df.columns if c != "Timestamp"]

        plot_cols = _ordered_playback_columns(plot_cols)
        x = ts

        emotion_idx = 0
        for col in plot_cols:
            y = df[col].to_numpy(dtype=np.float64)
            hex_color = PLAYBACK_EMOTION_COLORS.get(col)
            if hex_color is not None:
                qcol = QColor(hex_color)
                pen = pg.mkPen(qcol, width=1.5)
                brush = pg.mkBrush(qcol)
                style_hex = hex_color
            else:
                ic = pg.intColor(emotion_idx, values=1)
                pen = pg.mkPen(ic, width=1.5)
                brush = pg.mkBrush(ic)
                style_hex = QColor(ic).name(QColor.NameFormat.HexRgb)
                emotion_idx += 1

            curve = self._plot.plot(
                x,
                y,
                pen=pen,
                brush=brush,
                fillLevel=0.0,
            )
            if col == "Neutral":
                curve.setZValue(_Z_NEUTRAL)
            else:
                curve.setZValue(_Z_EMOTION_BASE + 0.1 * float(len(self._curves)))
            self._curves.append(curve)

            cb = QCheckBox(col)
            cb.setChecked(True)
            cb.setStyleSheet(_emotion_checkbox_stylesheet(style_hex))
            cb.toggled.connect(lambda checked, c=curve: c.setVisible(checked))
            self._toggle_layout.addWidget(cb)

        self._toggle_layout.addStretch(1)

        self._plot.setXRange(self._t_min, self._t_max, padding=0.02)
        self._plot.addItem(self._playhead)
        self.set_playhead_seconds(self._t_min)
        return None

    def set_playhead_seconds(self, t: float) -> None:
        t_clamped = max(self._t_min, min(self._t_max, float(t)))
        self._playhead.setPos(t_clamped)

    def time_range(self) -> tuple[float, float]:
        return self._t_min, self._t_max


class PlaybackPage(QWidget):
    go_main_menu = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)

        self._video = QVideoWidget(self)
        self._player.setVideoOutput(self._video)

        self._emotions = EmotionsPlotWidget(self)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._playback_bottom = QWidget()
        bottom_layout = QVBoxLayout(self._playback_bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)
        bottom_layout.addWidget(self._emotions, stretch=1)
        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider, stretch=1)
        bottom_layout.addLayout(slider_row)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self._video)
        self._splitter.addWidget(self._playback_bottom)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)

        self._back = QPushButton("Main menu")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_main_menu.emit)

        self._open = QPushButton("Open session…")
        self._open.clicked.connect(self._browse_session)

        self._play = QPushButton("Play")
        self._play.setEnabled(False)
        self._play.clicked.connect(self._toggle_play)

        self._path_label = QLabel("")
        self._path_label.setProperty("role", "hint")
        self._path_label.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(self._back)
        top.addStretch(1)
        top.addWidget(self._open)
        top.addWidget(self._play)

        body = QVBoxLayout(self)
        body.setContentsMargins(12, 12, 12, 12)
        body.addLayout(top)
        body.addWidget(self._path_label)
        body.addWidget(self._splitter, stretch=1)

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        self._has_video = False
        self._metrics_span_ms = 0
        self._metrics_position_ms = 0
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(50)
        self._metrics_timer.timeout.connect(self._on_metrics_tick)

    def _playhead_from_elapsed_ms(self, elapsed_ms: int) -> None:
        t_min, _t_max = self._emotions.time_range()
        t = t_min + max(0, elapsed_ms) / 1000.0
        self._emotions.set_playhead_seconds(t)

    def _set_video_pane_visible(self, visible: bool) -> None:
        self._video.setVisible(visible)
        if visible:
            self._splitter.setSizes([400, 400])
        else:
            self._splitter.setSizes([0, 10_000])

    def _stop_metrics_playback(self) -> None:
        self._metrics_timer.stop()

    def _on_metrics_tick(self) -> None:
        self._metrics_position_ms = min(self._metrics_position_ms + self._metrics_timer.interval(), self._metrics_span_ms)
        self._slider.blockSignals(True)
        self._slider.setValue(self._metrics_position_ms)
        self._slider.blockSignals(False)
        self._playhead_from_elapsed_ms(self._metrics_position_ms)
        if self._metrics_position_ms >= self._metrics_span_ms:
            self._metrics_timer.stop()
            self._play.setText("Play")

    def _browse_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select session folder")
        if not path:
            return
        err = _playback_preflight(path)
        if err:
            QMessageBox.warning(self, "Playback", err)
            return
        self._load_session(path)

    def _load_session(self, session_dir: str) -> None:
        video_path, csv_path = _session_playback_paths(session_dir)

        err = self._emotions.load_csv(csv_path)
        if err:
            QMessageBox.warning(self, "Playback", err)
            return

        self._stop_metrics_playback()
        t_min, t_max = self._emotions.time_range()
        span_s = max(0.0, t_max - t_min)
        self._metrics_span_ms = int(round(span_s * 1000.0))

        self._has_video = _session_has_video(session_dir)
        self._path_label.setText(f"Session: {session_dir}")

        self._player.stop()
        if self._has_video:
            self._set_video_pane_visible(True)
            url = QUrl.fromLocalFile(str(Path(video_path).resolve()))
            self._player.setSource(url)
        else:
            self._player.setSource(QUrl())
            self._set_video_pane_visible(False)

        self._metrics_position_ms = 0
        self._play.setEnabled(True)
        self._slider.setEnabled(True)
        self._slider.blockSignals(True)
        self._slider.setValue(0)
        if self._has_video:
            self._slider.setRange(0, 0)
        else:
            self._slider.setRange(0, max(0, self._metrics_span_ms))
        self._slider.blockSignals(False)

        if not self._has_video:
            self._playhead_from_elapsed_ms(0)
            self._play.setText("Play")

    def _toggle_play(self) -> None:
        if self._has_video:
            if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._player.pause()
            else:
                self._player.play()
            return
        if self._metrics_timer.isActive():
            self._metrics_timer.stop()
            self._play.setText("Play")
        else:
            if self._metrics_position_ms >= self._metrics_span_ms:
                self._metrics_position_ms = 0
                self._slider.blockSignals(True)
                self._slider.setValue(0)
                self._slider.blockSignals(False)
                self._playhead_from_elapsed_ms(0)
            self._metrics_timer.start()
            self._play.setText("Pause")

    def _on_playback_state_changed(self) -> None:
        if not self._has_video:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._play.setText("Pause")
        else:
            self._play.setText("Play")

    def _on_duration_changed(self, duration_ms: int) -> None:
        if not self._has_video:
            return
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, duration_ms))
        self._slider.blockSignals(False)

    def _on_position_changed(self, position_ms: int) -> None:
        if not self._has_video:
            return
        self._slider.blockSignals(True)
        self._slider.setValue(position_ms)
        self._slider.blockSignals(False)
        self._playhead_from_elapsed_ms(position_ms)

    def _on_slider_moved(self, position_ms: int) -> None:
        if self._has_video:
            self._player.setPosition(position_ms)
        else:
            if self._metrics_timer.isActive():
                self._metrics_timer.stop()
                self._play.setText("Play")
            self._metrics_position_ms = position_ms
        self._playhead_from_elapsed_ms(position_ms)

    def _on_slider_released(self) -> None:
        if not self._slider.isEnabled():
            return
        ms = self._slider.value()
        if self._has_video:
            self._player.setPosition(ms)
        else:
            self._metrics_position_ms = ms
        self._playhead_from_elapsed_ms(ms)
