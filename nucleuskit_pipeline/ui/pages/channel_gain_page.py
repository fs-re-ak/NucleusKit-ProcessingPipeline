"""Per-channel RMS gain and zero adjustment with blue vs red distribution plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.hermes.processor.rms_original_ops import (
    append_operation,
    ensure_baseline_snapshot,
    original_rms_csv,
    working_rms_csv,
)
from nucleuskit_pipeline.hermes.processorDevelopment.channel_fixer_release.channel_fixer.rms_columns import (
    CANONICAL_CHANNEL_NAMES,
    normalize_rms_dataframe,
)
from nucleuskit_pipeline.ui.offline_job import rms_features_preflight


def _quartile_stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"),) * 5
    qs = np.quantile(a, (0.0, 0.25, 0.5, 0.75, 1.0))
    return tuple(float(x) for x in qs)


def _apply_gain_zero(df: pd.DataFrame, gains: list[float], zeros: list[float]) -> pd.DataFrame:
    out = df.copy()
    for i, name in enumerate(CANONICAL_CHANNEL_NAMES):
        out[name] = out[name].astype(float) * gains[i] + zeros[i]
    return out


class ChannelGainPage(QWidget):
    go_tools_menu = Signal()

    _MAX_SCATTER = 1800
    _X_LEFT = -0.35
    _X_RIGHT = 0.75

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._back = QPushButton("Back")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_tools_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._back)

        self._folder: str | None = None
        self._work_df: pd.DataFrame | None = None
        self._blue_df: pd.DataFrame | None = None

        self._gains = [1.0] * 8
        self._zeros = [0.0] * 8
        self._spin_gains: list[QDoubleSpinBox] = []
        self._spin_zeros: list[QDoubleSpinBox] = []

        self._browse = QPushButton("Select session folder…")
        self._browse.clicked.connect(self._browse_session)

        self._reload = QPushButton("Reload RMS")
        self._reload.clicked.connect(self._reload_data)

        self._apply = QPushButton("Apply to rmsSignals.csv")
        self._apply.clicked.connect(self._apply_clicked)

        self._hint = QLabel(
            "Blue = frozen baseline (original/rmsSignals.csv if present, else current file on load). "
            "Red = preview after gain/zero. Zoom and pan on the plot affect the vertical axis only."
        )
        self._hint.setWordWrap(True)

        self._channel_combo = QComboBox()
        for name in CANONICAL_CHANNEL_NAMES:
            self._channel_combo.addItem(name)
        self._channel_combo.currentIndexChanged.connect(lambda _i: self._refresh_preview())

        ctrl_col = QVBoxLayout()
        ctrl_col.addWidget(self._browse)
        ctrl_col.addWidget(self._reload)
        ctrl_col.addWidget(self._apply)
        ctrl_col.addWidget(self._hint)
        ctrl_col.addWidget(QLabel("Plot channel:"))
        ctrl_col.addWidget(self._channel_combo)

        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_inner = QWidget()
        form_layout = QVBoxLayout(form_inner)
        for i, name in enumerate(CANONICAL_CHANNEL_NAMES):
            box = QGroupBox(name)
            fl = QFormLayout(box)
            sg = QDoubleSpinBox()
            sg.setRange(0.001, 1000.0)
            sg.setDecimals(4)
            sg.setValue(1.0)
            sg.setSingleStep(0.05)
            sz = QDoubleSpinBox()
            sz.setRange(-1e6, 1e6)
            sz.setDecimals(4)
            sz.setValue(0.0)

            def _mk_gain_handler(ii: int):
                def _h(v: float) -> None:
                    self._gains[ii] = v
                    self._refresh_preview()

                return _h

            def _mk_zero_handler(ii: int):
                def _h(v: float) -> None:
                    self._zeros[ii] = v
                    self._refresh_preview()

                return _h

            sg.valueChanged.connect(_mk_gain_handler(i))
            sz.valueChanged.connect(_mk_zero_handler(i))
            fl.addRow("Gain", sg)
            fl.addRow("Zero", sz)
            self._spin_gains.append(sg)
            self._spin_zeros.append(sz)
            form_layout.addWidget(box)
        form_layout.addStretch(1)
        form_scroll.setWidget(form_inner)

        self._table = QTableWidget()
        self._table.setColumnCount(11)
        self._table.setHorizontalHeaderLabels(
            [
                "Channel",
                "min B",
                "min R",
                "Q1 B",
                "Q1 R",
                "Med B",
                "Med R",
                "Q3 B",
                "Q3 R",
                "max B",
                "max R",
            ]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setRowCount(8)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("left", "RMS")
        self._plot.setLabel("bottom", "")
        self._plot.showGrid(x=False, y=True)
        self._plot.setMinimumHeight(280)
        self._plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=False, y=True)
        vb.sigRangeChanged.connect(self._on_plot_range_changed)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self._table)
        right_split.addWidget(self._plot)
        right_split.setStretchFactor(0, 0)
        right_split.setStretchFactor(1, 1)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.addLayout(ctrl_col)
        left_l.addWidget(form_scroll, 1)
        main_split.addWidget(left_w)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(main_split, 1)

    def _on_plot_range_changed(self, *_args) -> None:
        """Keep horizontal range fixed; only Y may change (zoom/pan)."""
        vb = self._plot.getViewBox()
        x0, x1 = self._X_LEFT, self._X_RIGHT
        try:
            xr = vb.viewRange()[0]
        except Exception:
            return
        if abs(xr[0] - x0) < 1e-6 and abs(xr[1] - x1) < 1e-6:
            return
        vb.blockSignals(True)
        try:
            vb.setXRange(x0, x1, padding=0)
        finally:
            vb.blockSignals(False)

    def _browse_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select session folder")
        if path:
            self._folder = path
            self._reload_data()

    def _reload_data(self) -> None:
        if not self._folder:
            QMessageBox.warning(self, "Session", "Select a session folder first.")
            return
        err = rms_features_preflight(self._folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return
        work_path = working_rms_csv(self._folder)
        self._work_df = normalize_rms_dataframe(pd.read_csv(str(work_path)))
        orig_path = original_rms_csv(self._folder)
        if orig_path.is_file():
            self._blue_df = normalize_rms_dataframe(pd.read_csv(str(orig_path)))
        else:
            self._blue_df = self._work_df.copy()
        for i in range(8):
            self._spin_gains[i].blockSignals(True)
            self._spin_zeros[i].blockSignals(True)
            self._spin_gains[i].setValue(1.0)
            self._spin_zeros[i].setValue(0.0)
            self._spin_gains[i].blockSignals(False)
            self._spin_zeros[i].blockSignals(False)
            self._gains[i] = 1.0
            self._zeros[i] = 0.0
        self._refresh_preview()

    def _preview_df(self) -> pd.DataFrame | None:
        if self._work_df is None:
            return None
        return _apply_gain_zero(self._work_df, self._gains, self._zeros)

    def _refresh_preview(self) -> None:
        if self._work_df is None or self._blue_df is None:
            self._table.setRowCount(0)
            self._plot.clear()
            return
        red_df = self._preview_df()
        if red_df is None:
            return

        self._table.setRowCount(8)
        for i, ch_name in enumerate(CANONICAL_CHANNEL_NAMES):
            bcol = self._blue_df[ch_name].to_numpy(dtype=float)
            rcol = red_df[ch_name].to_numpy(dtype=float)
            bstats = _quartile_stats(bcol)
            rstats = _quartile_stats(rcol)
            self._table.setItem(i, 0, QTableWidgetItem(ch_name))
            for j in range(5):
                self._table.setItem(i, 1 + 2 * j, QTableWidgetItem(f"{bstats[j]:.5g}"))
                self._table.setItem(i, 2 + 2 * j, QTableWidgetItem(f"{rstats[j]:.5g}"))

        idx = self._channel_combo.currentIndex()
        if idx < 0:
            idx = 0
        name = CANONICAL_CHANNEL_NAMES[idx]

        bcol = self._blue_df[name].to_numpy(dtype=float)
        rcol = red_df[name].to_numpy(dtype=float)

        self._plot.clear()
        self._plot.setTitle(name)
        n = len(bcol)
        k = min(n, self._MAX_SCATTER)
        if k < 1:
            self._plot.setXRange(self._X_LEFT, self._X_RIGHT, padding=0.02)
            return

        rng = np.random.default_rng(42)
        idx_s = rng.choice(n, size=k, replace=False)
        vb = bcol[idx_s]
        vr = rcol[idx_s]
        xb = rng.uniform(-0.2, 0.2, k)
        xr = 0.45 + rng.uniform(-0.2, 0.2, k)
        sp_b = pg.ScatterPlotItem(
            x=xb,
            y=vb,
            pen=None,
            brush=pg.mkBrush(30, 120, 255, 100),
            size=5,
        )
        sp_r = pg.ScatterPlotItem(
            x=xr,
            y=vr,
            pen=None,
            brush=pg.mkBrush(255, 60, 60, 100),
            size=5,
        )
        self._plot.addItem(sp_b)
        self._plot.addItem(sp_r)
        vb_plot = self._plot.getViewBox()
        vb_plot.blockSignals(True)
        try:
            vb_plot.setXRange(self._X_LEFT, self._X_RIGHT, padding=0.02)
            y_all = np.concatenate([vb, vr])
            y_all = y_all[np.isfinite(y_all)]
            if y_all.size > 0:
                y_min = float(np.min(y_all))
                y_max = float(np.max(y_all))
                pad = max((y_max - y_min) * 0.05, 1e-9)
                vb_plot.setYRange(y_min - pad, y_max + pad, padding=0.02)
        finally:
            vb_plot.blockSignals(False)

    def _apply_clicked(self) -> None:
        if not self._folder or self._work_df is None:
            QMessageBox.warning(self, "Session", "Load a session first.")
            return
        err = rms_features_preflight(self._folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return
        out = _apply_gain_zero(self._work_df, self._gains, self._zeros)
        work_path = working_rms_csv(self._folder)
        try:
            ensure_baseline_snapshot(self._folder)
            out.to_csv(work_path, index=False)
            parts = [
                f"{CANONICAL_CHANNEL_NAMES[i]}:g={self._gains[i]:.4g},z={self._zeros[i]:.4g}"
                for i in range(8)
            ]
            msg = "gain_adjust " + ";".join(parts)
            if len(msg) > 500:
                msg = msg[:497] + "..."
            append_operation(self._folder, msg)
        except OSError as e:
            QMessageBox.critical(self, "Apply", str(e))
            return
        self._reload_data()
        QMessageBox.information(self, "Gain", "Applied to rmsSignals.csv.")
