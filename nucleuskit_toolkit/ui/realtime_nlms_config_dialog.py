"""Dialog for configuring NLMS predictor connectivity in the real-time viewer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from nucleuskit_toolkit.hermes.constants import HermesConstants
from nucleuskit_toolkit.hermes.realtime.nlms_filter import build_default_predictor_mask


class NlmsPredictorConfigDialog(QDialog):
    """8×8 checkbox grid: rows = target channel, columns = predictor channel."""

    def __init__(
        self,
        mask: np.ndarray,
        parent=None,
        n_channels: int = 8,
    ) -> None:
        super().__init__(parent)
        self._n_channels = n_channels
        self._channel_names = HermesConstants.CHANNEL_NAMES[:n_channels]
        self.setWindowTitle("NLMS predictor matrix")
        self.setMinimumWidth(640)

        hint = QLabel(
            "Rows: channel being cleaned.  Columns: channels used as predictors.\n"
            "Checked = predictor enabled.  Diagonal is always disabled."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")

        grid = QGridLayout()
        grid.addWidget(QLabel("Target \\ Predictor"), 0, 0)
        for j, name in enumerate(self._channel_names):
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, 0, j + 1)

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n_channels, n_channels):
            mask = build_default_predictor_mask(n_channels)

        self._checkboxes: list[list[QCheckBox]] = []
        for i, target_name in enumerate(self._channel_names):
            row_label = QLabel(target_name)
            grid.addWidget(row_label, i + 1, 0)
            row_boxes: list[QCheckBox] = []
            for j in range(n_channels):
                cb = QCheckBox()
                cb.setChecked(bool(mask[i, j]))
                cb.setEnabled(i != j)
                cb.setToolTip(f"{self._channel_names[j]} → {target_name}")
                grid.addWidget(cb, i + 1, j + 1, alignment=Qt.AlignmentFlag.AlignCenter)
                row_boxes.append(cb)
            self._checkboxes.append(row_boxes)

        reset_btn = QPushButton("Reset to default")
        reset_btn.clicked.connect(self._reset_to_default)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addLayout(grid)
        layout.addLayout(btn_row)

    def _reset_to_default(self) -> None:
        mask = build_default_predictor_mask(self._n_channels)
        for i in range(self._n_channels):
            for j in range(self._n_channels):
                self._checkboxes[i][j].setChecked(bool(mask[i, j]))

    def predictor_mask(self) -> np.ndarray:
        """Return the configured (n, n) bool mask."""
        mask = np.zeros((self._n_channels, self._n_channels), dtype=bool)
        for i in range(self._n_channels):
            for j in range(self._n_channels):
                if i != j:
                    mask[i, j] = self._checkboxes[i][j].isChecked()
        return mask
