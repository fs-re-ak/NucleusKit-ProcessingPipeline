"""Session playback: optional video (rawData/video.mp4) + metrics plot (results/*.csv)."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

# Prefer PySide6 before pyqtgraph loads a different Qt binding.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QIcon, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
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
_Z_ANNOTATION = 50.0
_Z_PLAYHEAD = 100.0

PLAYBACK_ANNOTATIONS_FILENAME = "playback_annotations.json"
ANNOTATIONS_SCHEMA_VERSION = 1

# Pens for annotation graphics
_PEN_POINT = pg.mkPen("#88ccee", width=2)
_PEN_POINT_SEL = pg.mkPen("#ffcc66", width=3)
_PEN_ZONE = pg.mkPen("#88ccee", width=1)
_PEN_ZONE_SEL = pg.mkPen("#ffcc66", width=2)
_BRUSH_ZONE = pg.mkBrush(50, 130, 220, 130)
_BRUSH_ZONE_SEL = pg.mkBrush(255, 190, 70, 160)


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


_METRIC_CONFIGS: dict[str, dict] = {
    "emotions": {
        "file": "Emotions.csv",
        "label": "Emotions",
        "y_label": "Probability",
        "y_range": (0.0, 1.0),
        "filled": True,
    },
    "arousal": {
        "file": "Arousal.csv",
        "label": "Arousal",
        "y_label": "EDA (z-score)",
        "y_range": None,
        "filled": False,
    },
    "cognition": {
        "file": "Cognition.csv",
        "label": "Cognition",
        "y_label": "Index",
        "y_range": None,
        "filled": False,
    },
    "heartdynamics": {
        "file": "HeartDynamics.csv",
        "label": "Heart Dynamics",
        "y_label": "HRV Metrics",
        "y_range": None,
        "filled": False,
    },
}

_METRIC_ORDER = ["emotions", "arousal", "cognition", "heartdynamics"]


def _session_playback_paths(session_dir: str) -> tuple[str, str]:
    root = Path(session_dir).expanduser().resolve()
    video = root / "rawData" / "video.mp4"
    emotions = root / "results" / "Emotions.csv"
    return str(video), str(emotions)


def _available_metrics(session_dir: str) -> list[str]:
    """Return metric keys (in display order) whose result CSV exists."""
    root = Path(session_dir).expanduser().resolve()
    return [
        key for key in _METRIC_ORDER
        if (root / "results" / _METRIC_CONFIGS[key]["file"]).is_file()
    ]


def _metric_csv_path(session_dir: str, metric_key: str) -> str:
    root = Path(session_dir).expanduser().resolve()
    return str(root / "results" / _METRIC_CONFIGS[metric_key]["file"])


def playback_events_dir(session_dir: str) -> Path:
    return Path(session_dir).expanduser().resolve() / "features" / "events"


def playback_annotations_path(session_dir: str) -> Path:
    return playback_events_dir(session_dir) / PLAYBACK_ANNOTATIONS_FILENAME


def playback_annotations_results_path(session_dir: str) -> Path:
    return Path(session_dir).expanduser().resolve() / "results" / PLAYBACK_ANNOTATIONS_FILENAME


def _playback_preflight(session_dir: str) -> str | None:
    available = _available_metrics(session_dir)
    if not available:
        files = ", ".join(cfg["file"] for cfg in _METRIC_CONFIGS.values())
        return f"No result files found in results/.\nExpected at least one of: {files}"
    return None


def _session_has_video(session_dir: str) -> bool:
    video, _ = _session_playback_paths(session_dir)
    return Path(video).is_file()


def _clamp_t(t: float, t_min: float, t_max: float) -> float:
    return max(t_min, min(t_max, float(t)))


def _fmt_ms(ms: int) -> str:
    """Format milliseconds as M:SS (no zero-padding for minutes)."""
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


@dataclass
class _PointAnn:
    ann_id: str
    t: float
    label: str
    visible: bool = True


@dataclass
class _ZoneAnn:
    ann_id: str
    t0: float
    t1: float
    label: str
    visible: bool = True


class PlaybackAnnotationManager(QObject):
    """Point events (vertical lines) and zones (linear regions) on the plot."""

    changed = Signal()
    selection_changed = Signal(object)  # emits str ann_id or None

    def __init__(self, plot: pg.PlotWidget, get_time_bounds: callable) -> None:
        super().__init__(plot)
        self._plot = plot
        self._get_time_bounds = get_time_bounds
        self._points: dict[str, _PointAnn] = {}
        self._zones: dict[str, _ZoneAnn] = {}
        self._line_items: dict[str, pg.InfiniteLine] = {}
        self._region_items: dict[str, pg.LinearRegionItem] = {}
        self._selected_id: str | None = None
        self._tool_mode = "navigate"
        self._zone_pending_t0: float | None = None

    def set_tool_mode(self, mode: str) -> None:
        self._tool_mode = mode
        self._zone_pending_t0 = None

    def zone_pending_hint(self) -> str | None:
        if self._tool_mode != "zone":
            return None
        if self._zone_pending_t0 is None:
            return "Click start time, then end time."
        return "Click end time."

    @staticmethod
    def _set_region_pen(reg: pg.LinearRegionItem, pen: pg.QtGui.QPen) -> None:
        """LinearRegionItem has no setPen(); set it on each border line."""
        for line in reg.lines:
            line.setPen(pen)

    def _configure_zone_item(self, reg: pg.LinearRegionItem) -> None:
        reg.setZValue(_Z_ANNOTATION)
        self._set_region_pen(reg, _PEN_ZONE)
        reg.setBrush(_BRUSH_ZONE)
        reg.setSpan(0.0, 1.0)

    def set_visible(self, ann_id: str, visible: bool) -> None:
        if ann_id in self._points:
            self._points[ann_id].visible = visible
            self._line_items[ann_id].setVisible(visible)
        elif ann_id in self._zones:
            self._zones[ann_id].visible = visible
            self._region_items[ann_id].setVisible(visible)

    def clear(self) -> None:
        for it in list(self._line_items.values()):
            self._plot.removeItem(it)
        for it in list(self._region_items.values()):
            self._plot.removeItem(it)
        self._points.clear()
        self._zones.clear()
        self._line_items.clear()
        self._region_items.clear()
        self._selected_id = None
        self._zone_pending_t0 = None
        self.selection_changed.emit(None)

    def selected_id(self) -> str | None:
        return self._selected_id

    def select(self, ann_id: str | None) -> None:
        if self._selected_id == ann_id:
            if ann_id is not None:
                self._apply_selection_style()
            return
        self._selected_id = ann_id
        self._apply_selection_style()
        self.selection_changed.emit(ann_id)

    def _apply_selection_style(self) -> None:
        sid = self._selected_id
        for aid, line in self._line_items.items():
            line.setPen(_PEN_POINT_SEL if aid == sid else _PEN_POINT)
        for aid, reg in self._region_items.items():
            if aid == sid:
                self._set_region_pen(reg, _PEN_ZONE_SEL)
                reg.setBrush(_BRUSH_ZONE_SEL)
            else:
                self._set_region_pen(reg, _PEN_ZONE)
                reg.setBrush(_BRUSH_ZONE)

    def _bounds(self) -> tuple[float, float]:
        return self._get_time_bounds()

    def handle_scene_click(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = ev.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        vb = self._plot.getViewBox()
        t = float(vb.mapSceneToView(pos).x())
        t_min, t_max = self._bounds()
        t = _clamp_t(t, t_min, t_max)

        if self._tool_mode == "event":
            self._add_point(t)
        elif self._tool_mode == "zone":
            if self._zone_pending_t0 is None:
                self._zone_pending_t0 = t
            else:
                t0, t1 = self._zone_pending_t0, t
                self._zone_pending_t0 = None
                if abs(t1 - t0) < 1e-6:
                    return
                if t0 > t1:
                    t0, t1 = t1, t0
                self._add_zone(t0, t1)

    def _add_point(self, t: float) -> None:
        aid = str(uuid.uuid4())
        n = len(self._points) + 1
        ann = _PointAnn(aid, t, f"Event {n}")
        self._points[aid] = ann
        line = pg.InfiniteLine(pos=t, angle=90, movable=True, pen=_PEN_POINT)
        line.setZValue(_Z_ANNOTATION)
        line.setVisible(ann.visible)
        setattr(line, "_ann_id", aid)
        setattr(line, "_ann_kind", "point")
        self._line_items[aid] = line
        self._plot.addItem(line)
        self._wire_point_line(line, aid)
        self.select(aid)
        self.changed.emit()

    def _wire_point_line(self, line: pg.InfiniteLine, aid: str) -> None:
        sig = getattr(line, "sigPositionChangeFinished", None) or line.sigPositionChanged

        def _on_pos(_=None) -> None:
            t_min, t_max = self._bounds()
            nt = _clamp_t(float(line.value()), t_min, t_max)
            if abs(nt - float(line.value())) > 1e-12:
                line.blockSignals(True)
                line.setPos(nt)
                line.blockSignals(False)
            if aid in self._points:
                self._points[aid].t = nt
                self.changed.emit()

        sig.connect(_on_pos)

    def _add_zone(self, t0: float, t1: float) -> None:
        aid = str(uuid.uuid4())
        n = len(self._zones) + 1
        ann = _ZoneAnn(aid, t0, t1, f"Zone {n}")
        self._zones[aid] = ann
        reg = pg.LinearRegionItem(
            values=(t0, t1),
            orientation="vertical",
            movable=True,
            bounds=self._bounds(),
        )
        setattr(reg, "_ann_id", aid)
        setattr(reg, "_ann_kind", "zone")
        self._region_items[aid] = reg
        self._plot.addItem(reg)
        self._configure_zone_item(reg)
        reg.setVisible(ann.visible)
        self._wire_zone_region(reg, aid)
        self.select(aid)
        self.changed.emit()

    def _wire_zone_region(self, reg: pg.LinearRegionItem, aid: str) -> None:
        sig = getattr(reg, "sigRegionChangeFinished", None) or reg.sigRegionChanged

        def _on_reg() -> None:
            t_min, t_max = self._bounds()
            r0, r1 = reg.getRegion()
            r0, r1 = _clamp_t(r0, t_min, t_max), _clamp_t(r1, t_min, t_max)
            if r0 > r1:
                r0, r1 = r1, r0
            cur = reg.getRegion()
            if abs(cur[0] - r0) > 1e-9 or abs(cur[1] - r1) > 1e-9:
                reg.blockSignals(True)
                reg.setRegion((r0, r1))
                reg.blockSignals(False)
            if aid in self._zones:
                self._zones[aid].t0 = r0
                self._zones[aid].t1 = r1
                self.changed.emit()

        sig.connect(_on_reg)

    def delete_selected(self) -> None:
        if self._selected_id is None:
            return
        aid = self._selected_id
        self.select(None)
        if aid in self._points:
            self._plot.removeItem(self._line_items.pop(aid))
            del self._points[aid]
        elif aid in self._zones:
            self._plot.removeItem(self._region_items.pop(aid))
            del self._zones[aid]
        self.changed.emit()

    def row_snapshot(self) -> list[tuple[str, str, str, str, str, bool]]:
        """Rows: ann_id, type, label, start, end, visible."""
        rows: list[tuple[str, str, str, str, str, bool]] = []
        for aid, p in sorted(self._points.items(), key=lambda kv: kv[1].t):
            rows.append((aid, "Event", p.label, f"{p.t:.6g}", "", p.visible))
        for aid, z in sorted(self._zones.items(), key=lambda kv: kv[1].t0):
            rows.append((aid, "Zone", z.label, f"{z.t0:.6g}", f"{z.t1:.6g}", z.visible))
        return rows

    def set_label(self, ann_id: str, label: str) -> None:
        if ann_id in self._points:
            self._points[ann_id].label = label
        elif ann_id in self._zones:
            self._zones[ann_id].label = label

    def set_point_time(self, ann_id: str, t: float) -> None:
        if ann_id not in self._points:
            return
        t_min, t_max = self._bounds()
        t = _clamp_t(t, t_min, t_max)
        self._points[ann_id].t = t
        line = self._line_items[ann_id]
        line.blockSignals(True)
        line.setPos(t)
        line.blockSignals(False)

    def set_zone_times(self, ann_id: str, t0: float, t1: float) -> None:
        if ann_id not in self._zones:
            return
        t_min, t_max = self._bounds()
        t0, t1 = _clamp_t(t0, t_min, t_max), _clamp_t(t1, t_min, t_max)
        if t0 > t1:
            t0, t1 = t1, t0
        self._zones[ann_id].t0 = t0
        self._zones[ann_id].t1 = t1
        reg = self._region_items[ann_id]
        reg.blockSignals(True)
        reg.setRegion((t0, t1))
        reg.blockSignals(False)

    def to_json_dict(self) -> dict:
        return {
            "version": ANNOTATIONS_SCHEMA_VERSION,
            "points": [
                {"id": p.ann_id, "t": p.t, "label": p.label, "visible": p.visible} for p in self._points.values()
            ],
            "zones": [
                {"id": z.ann_id, "t0": z.t0, "t1": z.t1, "label": z.label, "visible": z.visible}
                for z in self._zones.values()
            ],
        }

    def load_from_json_dict(self, data: dict) -> str | None:
        """Returns error message or None."""
        self.clear()
        if not isinstance(data, dict):
            self.changed.emit()
            return "Invalid annotations file (not an object)."
        pts = data.get("points", [])
        zns = data.get("zones", [])
        if not isinstance(pts, list) or not isinstance(zns, list):
            self.changed.emit()
            return "Invalid annotations file (points/zones)."
        t_min, t_max = self._bounds()
        for p in pts:
            if not isinstance(p, dict):
                continue
            try:
                aid = str(p.get("id") or uuid.uuid4())
                t = float(p["t"])
                label = str(p.get("label") or "Event")
            except (KeyError, TypeError, ValueError):
                continue
            vis = p.get("visible") is not False
            t = _clamp_t(t, t_min, t_max)
            self._points[aid] = _PointAnn(aid, t, label, visible=vis)
            line = pg.InfiniteLine(pos=t, angle=90, movable=True, pen=_PEN_POINT)
            line.setZValue(_Z_ANNOTATION)
            line.setVisible(vis)
            setattr(line, "_ann_id", aid)
            self._line_items[aid] = line
            self._plot.addItem(line)
            self._wire_point_line(line, aid)
        for z in zns:
            if not isinstance(z, dict):
                continue
            try:
                aid = str(z.get("id") or uuid.uuid4())
                t0 = float(z["t0"])
                t1 = float(z["t1"])
                label = str(z.get("label") or "Zone")
            except (KeyError, TypeError, ValueError):
                continue
            vis = z.get("visible") is not False
            t0, t1 = _clamp_t(t0, t_min, t_max), _clamp_t(t1, t_min, t_max)
            if t0 > t1:
                t0, t1 = t1, t0
            if abs(t1 - t0) < 1e-9:
                continue
            self._zones[aid] = _ZoneAnn(aid, t0, t1, label, visible=vis)
            reg = pg.LinearRegionItem(
                values=(t0, t1),
                orientation="vertical",
                movable=True,
                bounds=(t_min, t_max),
            )
            setattr(reg, "_ann_id", aid)
            self._region_items[aid] = reg
            self._plot.addItem(reg)
            self._configure_zone_item(reg)
            reg.setVisible(vis)
            self._wire_zone_region(reg, aid)
        self._apply_selection_style()
        self.changed.emit()
        return None


class MetricsPlotWidget(QWidget):
    """Time-series metrics plot with a vertical playhead and per-series toggles."""

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

        vb = self._plot.getViewBox()
        vb.sigRangeChanged.connect(self._on_view_range_changed)
        self._trim_context_menu()

        self._toggle_bar = QWidget()
        self._toggle_layout = QHBoxLayout(self._toggle_bar)
        self._toggle_layout.setContentsMargins(4, 4, 4, 0)
        self._toggle_layout.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, stretch=1)
        layout.addWidget(self._toggle_bar)

    def plot_widget(self) -> pg.PlotWidget:
        return self._plot

    def _trim_context_menu(self) -> None:
        """Keep only 'View All' in the right-click context menu."""
        vb = self._plot.getViewBox()
        for action in list(vb.menu.actions()):
            if "all" not in action.text().lower():
                vb.menu.removeAction(action)
        # Remove "Plot Options" (PlotItem ctrl menu)
        self._plot.plotItem.ctrlMenu = None
        # Remove "Export…" (GraphicsScene level)
        self._plot.scene().contextMenu = []

    def _on_view_range_changed(self, *_args) -> None:
        """Keep visible X inside [t_min, t_max] and width <= span."""
        if self._t_max <= self._t_min + 1e-15:
            return
        vb = self._plot.getViewBox()
        lo, hi = self._t_min, self._t_max
        span = hi - lo
        try:
            x0, x1 = vb.viewRange()[0]
        except Exception:
            return
        w = x1 - x0
        changed = False
        if w > span + 1e-9:
            cx = (x0 + x1) * 0.5
            x0, x1 = cx - span * 0.5, cx + span * 0.5
            changed = True
        if x0 < lo - 1e-12:
            sh = lo - x0
            x0 += sh
            x1 += sh
            changed = True
        if x1 > hi + 1e-12:
            sh = x1 - hi
            x0 -= sh
            x1 -= sh
            changed = True
        if x0 < lo:
            x0 = lo
            changed = True
        if x1 > hi:
            x1 = hi
            changed = True
        if x1 - x0 > span:
            x1 = hi
            x0 = lo
            changed = True
        if not changed:
            return
        vb.blockSignals(True)
        try:
            vb.setXRange(x0, x1, padding=0)
        finally:
            vb.blockSignals(False)

    def _apply_x_view_limits(self) -> None:
        vb = self._plot.getViewBox()
        lo, hi = self._t_min, self._t_max
        span = max(hi - lo, 1e-9)
        min_w = min(max(span * 0.001, 1e-4), span * 0.5)
        vb.setLimits(xMin=lo, xMax=hi, minXRange=min_w, maxXRange=span * 1.0001)

    def _clear_toggle_widgets(self) -> None:
        while self._toggle_layout.count():
            item = self._toggle_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def load_csv(
        self,
        path: str,
        *,
        y_label: str = "Probability",
        y_range: tuple[float, float] | None = (0.0, 1.0),
        filled: bool = True,
    ) -> str | None:
        """Load a metrics CSV (must have a Timestamp column). Returns error or None."""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return f"Could not read {path}:\n{e}"

        filename = Path(path).name
        if "Timestamp" not in df.columns:
            return f"{filename} must contain a Timestamp column."

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
            return f"{filename} has no rows."

        self._t_min = float(np.nanmin(ts))
        self._t_max = float(np.nanmax(ts))

        # Prefer emotion column order when emotion columns are present.
        emotion_cols = [c for c in EMOTION_COLUMNS if c in df.columns]
        if emotion_cols:
            plot_cols = _ordered_playback_columns(emotion_cols)
        else:
            plot_cols = [c for c in df.columns if c != "Timestamp"]

        self._plot.setLabel("left", y_label)

        if y_range is not None:
            self._plot.setYRange(y_range[0], y_range[1], padding=0.02)
        else:
            all_vals = np.concatenate([df[c].to_numpy(dtype=np.float64) for c in plot_cols])
            valid = all_vals[np.isfinite(all_vals)]
            if valid.size > 0:
                lo, hi = float(np.nanmin(valid)), float(np.nanmax(valid))
                span = max(hi - lo, 1e-3)
                pad = span * 0.08
                self._plot.setYRange(lo - pad, hi + pad, padding=0)

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

            plot_kwargs: dict = {"pen": pen}
            if filled:
                plot_kwargs["brush"] = brush
                plot_kwargs["fillLevel"] = 0.0

            curve = self._plot.plot(x, y, **plot_kwargs)
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

        self._apply_x_view_limits()
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

        self._metrics_plot = MetricsPlotWidget(self)
        self._ann_mgr = PlaybackAnnotationManager(
            self._metrics_plot.plot_widget(),
            self._metrics_plot.time_range,
        )
        self._ann_mgr.changed.connect(self._on_annotations_changed)
        self._ann_mgr.selection_changed.connect(self._on_ann_selection_changed)

        self._plot_scene = self._metrics_plot.plot_widget().scene()

        self._current_metric: str | None = None
        self._metric_buttons: dict[str, QRadioButton] = {}
        self._plot_scene.sigMouseClicked.connect(self._on_plot_scene_clicked)

        self._session_dir: str | None = None
        self._annotations_dirty = False
        self._suppress_ann_changed_dirty = False
        self._table_refreshing = False

        self._tool_nav = QRadioButton("Navigate")
        self._tool_event = QRadioButton("Event")
        self._tool_zone = QRadioButton("Zone")
        self._tool_nav.setChecked(True)
        self._tool_group = QButtonGroup(self)
        self._tool_group.addButton(self._tool_nav)
        self._tool_group.addButton(self._tool_event)
        self._tool_group.addButton(self._tool_zone)
        self._tool_group.buttonClicked.connect(self._on_tool_button_clicked)

        self._zone_hint = QLabel("")
        self._zone_hint.setProperty("role", "hint")

        self._save_ann = QPushButton("Save annotations")
        self._save_ann.setEnabled(False)
        self._save_ann.clicked.connect(self._save_annotations_file)

        self._del_ann = QPushButton("Delete selected")
        self._del_ann.setEnabled(False)
        self._del_ann.setProperty("secondary", True)
        self._del_ann.clicked.connect(self._delete_selected_annotation)

        tool_row = QHBoxLayout()
        tool_row.addWidget(QLabel("Tool:"))
        tool_row.addWidget(self._tool_nav)
        tool_row.addWidget(self._tool_event)
        tool_row.addWidget(self._tool_zone)
        tool_row.addWidget(self._zone_hint, stretch=1)
        tool_row.addWidget(self._del_ann)
        tool_row.addWidget(self._save_ann)

        self._ann_table = QTableWidget(0, 6)
        self._ann_table.setHorizontalHeaderLabels(["", "Type", "Label", "Start (s)", "End (s)", "Id"])
        self._ann_table.hideColumn(5)
        self._ann_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._ann_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._ann_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._ann_table.itemChanged.connect(self._on_table_item_changed)
        hdr = self._ann_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._ann_table.setColumnWidth(0, 36)

        ann_panel = QWidget()
        ann_layout = QVBoxLayout(ann_panel)
        ann_layout.setContentsMargins(0, 0, 0, 0)
        ann_layout.addWidget(QLabel("Annotations"))
        ann_layout.addWidget(self._ann_table, stretch=1)

        self._metric_bar = QWidget()
        self._metric_layout = QHBoxLayout(self._metric_bar)
        self._metric_layout.setContentsMargins(0, 2, 0, 2)
        self._metric_layout.setSpacing(8)

        plot_split = QSplitter(Qt.Orientation.Horizontal)
        plot_split.addWidget(self._metrics_plot)
        plot_split.addWidget(ann_panel)
        plot_split.setStretchFactor(0, 3)
        plot_split.setStretchFactor(1, 1)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._time_label.setStyleSheet("font-variant-numeric: tabular-nums; min-width: 90px;")

        self._playback_bottom = QWidget()
        bottom_layout = QVBoxLayout(self._playback_bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)
        bottom_layout.addLayout(tool_row)
        bottom_layout.addWidget(self._metric_bar)
        bottom_layout.addWidget(plot_split, stretch=1)
        slider_row = QHBoxLayout()
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._time_label)
        bottom_layout.addLayout(slider_row)

        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self._video)
        self._splitter.addWidget(self._playback_bottom)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)

        self._back = QPushButton("Main menu")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self._on_main_menu_clicked)

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

        QShortcut(QKeySequence.StandardKey.Delete, self, self._delete_selected_annotation)

        self._refresh_tool_hint()

    def _on_main_menu_clicked(self) -> None:
        if not self._maybe_discard_dirty():
            return
        self._session_dir = None
        self._ann_mgr.clear()
        self._refresh_annotation_table()
        self._save_ann.setEnabled(False)
        self.go_main_menu.emit()

    def _on_tool_button_clicked(self, btn: QAbstractButton) -> None:
        if btn == self._tool_nav:
            mode = "navigate"
        elif btn == self._tool_event:
            mode = "event"
        else:
            mode = "zone"
        self._ann_mgr.set_tool_mode(mode)
        self._refresh_tool_hint()

    def _refresh_tool_hint(self) -> None:
        h = self._ann_mgr.zone_pending_hint()
        self._zone_hint.setText(h or "")

    def _on_plot_scene_clicked(self, ev) -> None:
        if self._tool_nav.isChecked():
            return
        self._ann_mgr.handle_scene_click(ev)
        self._refresh_tool_hint()

    def _on_annotations_changed(self) -> None:
        self._refresh_annotation_table()
        self._refresh_tool_hint()
        if self._suppress_ann_changed_dirty:
            return
        self._annotations_dirty = True
        self._save_ann.setEnabled(True)

    def _on_ann_selection_changed(self, ann_id: str | None) -> None:
        self._del_ann.setEnabled(ann_id is not None)
        if ann_id is None:
            self._ann_table.blockSignals(True)
            self._ann_table.clearSelection()
            self._ann_table.blockSignals(False)
            return
        self._select_table_row_for_id(ann_id)

    def _select_table_row_for_id(self, ann_id: str) -> None:
        for r in range(self._ann_table.rowCount()):
            it = self._ann_table.item(r, 5)
            if it is not None and it.text() == ann_id:
                self._ann_table.blockSignals(True)
                self._ann_table.selectRow(r)
                self._ann_table.blockSignals(False)
                return

    def _make_visibility_toggle(self, ann_id: str, visible: bool) -> QToolButton:
        btn = QToolButton()
        btn.setCheckable(True)
        btn.setChecked(visible)
        btn.setFixedSize(32, 24)
        btn.setAutoRaise(True)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        ic = QIcon.fromTheme("view-visible")
        if not ic.isNull():
            btn.setIcon(ic)
        else:
            btn.setText("\U0001f441")
        btn.toggled.connect(lambda checked, a=ann_id: self._on_annotation_visibility_toggled(a, checked))
        self._update_visibility_button_tooltip(btn, visible)
        return btn

    def _update_visibility_button_tooltip(self, btn: QToolButton, visible: bool) -> None:
        btn.setToolTip("Click to hide on plot" if visible else "Click to show on plot")

    def _on_annotation_visibility_toggled(self, ann_id: str, visible: bool) -> None:
        self._ann_mgr.set_visible(ann_id, visible)
        snd = self.sender()
        if isinstance(snd, QToolButton):
            self._update_visibility_button_tooltip(snd, visible)
        self._annotations_dirty = True
        self._save_ann.setEnabled(True)

    def _refresh_annotation_table(self) -> None:
        keep_sel = self._ann_mgr.selected_id()
        self._table_refreshing = True
        self._ann_table.blockSignals(True)
        try:
            for r in range(self._ann_table.rowCount()):
                w = self._ann_table.cellWidget(r, 0)
                if w is not None:
                    self._ann_table.removeCellWidget(r, 0)
                    w.deleteLater()
            rows = self._ann_mgr.row_snapshot()
            self._ann_table.setRowCount(len(rows))
            for i, (_aid, typ, label, start_s, end_s, vis) in enumerate(rows):
                self._ann_table.setCellWidget(i, 0, self._make_visibility_toggle(_aid, vis))
                self._ann_table.setItem(i, 1, QTableWidgetItem(typ))
                lab = QTableWidgetItem(label)
                lab.setFlags(lab.flags() | Qt.ItemFlag.ItemIsEditable)
                self._ann_table.setItem(i, 2, lab)
                st = QTableWidgetItem(start_s)
                en = QTableWidgetItem(end_s)
                if typ == "Event":
                    st.setFlags(st.flags() | Qt.ItemFlag.ItemIsEditable)
                    en.setFlags(en.flags() & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    st.setFlags(st.flags() | Qt.ItemFlag.ItemIsEditable)
                    en.setFlags(en.flags() | Qt.ItemFlag.ItemIsEditable)
                self._ann_table.setItem(i, 3, st)
                self._ann_table.setItem(i, 4, en)
                id_item = QTableWidgetItem(_aid)
                id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._ann_table.setItem(i, 5, id_item)
            if keep_sel is not None:
                for r in range(self._ann_table.rowCount()):
                    it = self._ann_table.item(r, 5)
                    if it is not None and it.text() == keep_sel:
                        self._ann_table.selectRow(r)
                        break
        finally:
            self._ann_table.blockSignals(False)
            self._table_refreshing = False
        self._del_ann.setEnabled(self._ann_mgr.selected_id() is not None)

    def _on_table_selection_changed(self) -> None:
        if self._table_refreshing:
            return
        row = self._ann_table.currentRow()
        if row < 0:
            self._ann_mgr.select(None)
            self._del_ann.setEnabled(False)
            return
        it = self._ann_table.item(row, 5)
        if it is None:
            self._del_ann.setEnabled(False)
            return
        self._ann_mgr.select(it.text())
        self._del_ann.setEnabled(True)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._table_refreshing:
            return
        row = item.row()
        id_item = self._ann_table.item(row, 5)
        if id_item is None:
            return
        aid = id_item.text()
        col = item.column()
        typ_item = self._ann_table.item(row, 1)
        typ = typ_item.text() if typ_item else ""
        if col == 2:
            self._ann_mgr.set_label(aid, item.text())
        elif col == 3:
            try:
                t = float(item.text())
            except ValueError:
                self._refresh_annotation_table()
                return
            if typ == "Event":
                self._ann_mgr.set_point_time(aid, t)
            else:
                en = self._ann_table.item(row, 4)
                try:
                    t1 = float(en.text()) if en and en.text() else t
                except ValueError:
                    self._refresh_annotation_table()
                    return
                self._ann_mgr.set_zone_times(aid, t, t1)
        elif col == 4 and typ == "Zone":
            st = self._ann_table.item(row, 3)
            try:
                t0 = float(st.text()) if st else 0.0
                t1 = float(item.text())
            except ValueError:
                self._refresh_annotation_table()
                return
            self._ann_mgr.set_zone_times(aid, t0, t1)
        if col in (3, 4):
            self._refresh_annotation_table()
            self._select_table_row_for_id(aid)
        self._annotations_dirty = True
        self._save_ann.setEnabled(True)

    def _delete_selected_annotation(self) -> None:
        self._ann_mgr.delete_selected()
        self._annotations_dirty = True
        self._save_ann.setEnabled(True)

    def _save_annotations_file(self) -> None:
        if self._session_dir is None:
            return
        payload = json.dumps(self._ann_mgr.to_json_dict(), indent=2)
        primary = playback_annotations_path(self._session_dir)
        mirror = playback_annotations_results_path(self._session_dir)
        try:
            playback_events_dir(self._session_dir).mkdir(parents=True, exist_ok=True)
            primary.write_text(payload, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, "Playback", f"Could not save annotations:\n{e}")
            return
        try:
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_text(payload, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, "Playback", f"Annotations saved to features/events but could not mirror to results/:\n{e}")
        self._annotations_dirty = False
        self._save_ann.setEnabled(False)

    def _load_annotations_file(self) -> None:
        if self._session_dir is None:
            return
        path = playback_annotations_path(self._session_dir)
        self._suppress_ann_changed_dirty = True
        try:
            if not path.is_file():
                self._ann_mgr.clear()
            else:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as e:
                    QMessageBox.warning(self, "Playback", f"Could not read annotations file:\n{e}")
                    self._ann_mgr.clear()
                else:
                    err = self._ann_mgr.load_from_json_dict(data)
                    if err:
                        QMessageBox.warning(self, "Playback", err)
        finally:
            self._suppress_ann_changed_dirty = False
            self._refresh_annotation_table()
            self._annotations_dirty = False
            self._save_ann.setEnabled(False)

    def _maybe_discard_dirty(self) -> bool:
        if not self._annotations_dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Playback")
        box.setText("Save changes to playback annotations?")
        save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setIcon(QMessageBox.Icon.Question)
        box.exec()
        clicked = box.clickedButton()
        if clicked is None or clicked == cancel_btn:
            return False
        if clicked == save_btn:
            self._save_annotations_file()
            if self._annotations_dirty:
                return False
        else:
            self._annotations_dirty = False
            self._save_ann.setEnabled(False)
        return True

    def _update_time_label(self, position_ms: int, duration_ms: int) -> None:
        self._time_label.setText(f"{_fmt_ms(position_ms)} / {_fmt_ms(duration_ms)}")

    def _playhead_from_elapsed_ms(self, elapsed_ms: int) -> None:
        t_min, _t_max = self._metrics_plot.time_range()
        t = t_min + max(0, elapsed_ms) / 1000.0
        self._metrics_plot.set_playhead_seconds(t)

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
        self._update_time_label(self._metrics_position_ms, self._metrics_span_ms)
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
        if self._session_dir is not None and Path(path).resolve() != Path(self._session_dir).resolve():
            if not self._maybe_discard_dirty():
                return
        self._load_session(path)

    def _load_session(self, session_dir: str) -> None:
        video_path, _ = _session_playback_paths(session_dir)

        available = _available_metrics(session_dir)
        preferred = "emotions" if "emotions" in available else available[0]
        cfg = _METRIC_CONFIGS[preferred]
        csv_path = _metric_csv_path(session_dir, preferred)

        err = self._metrics_plot.load_csv(
            csv_path,
            y_label=cfg["y_label"],
            y_range=cfg.get("y_range"),
            filled=cfg.get("filled", False),
        )
        if err:
            QMessageBox.warning(self, "Playback", err)
            return

        self._ann_mgr.clear()
        self._refresh_annotation_table()
        self._session_dir = str(Path(session_dir).expanduser().resolve())
        self._current_metric = preferred
        self._populate_metric_selector(available, selected=preferred)
        self._load_annotations_file()

        self._stop_metrics_playback()
        t_min, t_max = self._metrics_plot.time_range()
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

        self._update_time_label(0, self._metrics_span_ms if not self._has_video else 0)

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
        self._update_time_label(self._player.position(), duration_ms)

    def _on_position_changed(self, position_ms: int) -> None:
        if not self._has_video:
            return
        self._slider.blockSignals(True)
        self._slider.setValue(position_ms)
        self._slider.blockSignals(False)
        self._playhead_from_elapsed_ms(position_ms)
        self._update_time_label(position_ms, self._player.duration())

    def _on_slider_moved(self, position_ms: int) -> None:
        if self._has_video:
            self._player.setPosition(position_ms)
            duration_ms = self._player.duration()
        else:
            if self._metrics_timer.isActive():
                self._metrics_timer.stop()
                self._play.setText("Play")
            self._metrics_position_ms = position_ms
            duration_ms = self._metrics_span_ms
        self._playhead_from_elapsed_ms(position_ms)
        self._update_time_label(position_ms, duration_ms)

    def _on_slider_released(self) -> None:
        if not self._slider.isEnabled():
            return
        ms = self._slider.value()
        if self._has_video:
            self._player.setPosition(ms)
        else:
            self._metrics_position_ms = ms
        self._playhead_from_elapsed_ms(ms)

    # ------------------------------------------------------------------
    # Metric selector
    # ------------------------------------------------------------------

    def _populate_metric_selector(self, available: list[str], selected: str | None = None) -> None:
        """Rebuild the metric selector bar showing only available result files."""
        while self._metric_layout.count():
            item = self._metric_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._metric_buttons.clear()

        if not available:
            return

        lbl = QLabel("Metric:")
        self._metric_layout.addWidget(lbl)

        for key in _METRIC_ORDER:
            if key not in available:
                continue
            cfg = _METRIC_CONFIGS[key]
            btn = QRadioButton(cfg["label"])
            btn.setChecked(key == selected)
            btn.toggled.connect(
                lambda checked, k=key: self._on_metric_button_toggled(k, checked)
            )
            self._metric_buttons[key] = btn
            self._metric_layout.addWidget(btn)

        self._metric_layout.addStretch(1)

    def _on_metric_button_toggled(self, key: str, checked: bool) -> None:
        if not checked or key == self._current_metric:
            return
        self._switch_metric(key)

    def _switch_metric(self, key: str) -> None:
        """Load a different metric CSV into the plot without reloading the session."""
        if self._session_dir is None:
            return
        cfg = _METRIC_CONFIGS[key]
        path = _metric_csv_path(self._session_dir, key)
        err = self._metrics_plot.load_csv(
            path,
            y_label=cfg["y_label"],
            y_range=cfg.get("y_range"),
            filled=cfg.get("filled", False),
        )
        if err:
            QMessageBox.warning(self, "Playback", err)
            # Restore the previous button selection
            prev = self._current_metric
            if prev and prev in self._metric_buttons:
                self._metric_buttons[prev].blockSignals(True)
                self._metric_buttons[prev].setChecked(True)
                self._metric_buttons[prev].blockSignals(False)
            return
        self._current_metric = key

        if not self._has_video:
            t_min, t_max = self._metrics_plot.time_range()
            span_s = max(0.0, t_max - t_min)
            self._metrics_span_ms = int(round(span_s * 1000.0))
            self._stop_metrics_playback()
            self._metrics_position_ms = 0
            self._slider.blockSignals(True)
            self._slider.setValue(0)
            self._slider.setRange(0, max(0, self._metrics_span_ms))
            self._slider.blockSignals(False)
            self._playhead_from_elapsed_ms(0)
            self._play.setText("Play")
