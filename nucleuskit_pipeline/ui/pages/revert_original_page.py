"""Revert working RMS CSV to the frozen baseline in ``features/emotions/original/``."""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.hermes.processor.rms_ops import (
    operations_log_path,
    original_rms_csv,
    revert_working_to_original,
)
from nucleuskit_pipeline.ui.offline_job import rms_features_preflight


class RevertOriginalPage(QWidget):
    go_tools_menu = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._back = QPushButton("Back")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_tools_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._back)

        self._session = QLabel()
        self._session.setWordWrap(True)
        self._session.setText("No session selected.")

        self._browse = QPushButton("Select session folder…")
        self._browse.clicked.connect(self._browse_session)

        self._revert = QPushButton("Revert to original")
        self._revert.clicked.connect(self._revert_clicked)

        sess_box = QGroupBox("Session")
        sg = QVBoxLayout(sess_box)
        sg.addWidget(self._browse)
        sg.addWidget(self._session)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("operations.log tail (after revert)…")

        log_box = QGroupBox("Recent operations log")
        lg = QVBoxLayout(log_box)
        lg.addWidget(self._log)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(sess_box)
        layout.addWidget(self._revert)
        layout.addWidget(log_box, 1)

        self._folder: str | None = None

    def _browse_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select session folder")
        if path:
            self._folder = path
            self._session.setText(path)
            self._refresh_log_tail()

    def _refresh_log_tail(self) -> None:
        if not self._folder:
            self._log.clear()
            return
        log_path = operations_log_path(self._folder)
        if not os.path.isfile(log_path):
            self._log.clear()
            return
        try:
            with open(log_path, encoding="utf-8") as fh:
                lines = fh.readlines()
            tail = "".join(lines[-80:])
            self._log.setPlainText(tail)
        except OSError:
            self._log.setPlainText("(Could not read log.)")

    def _revert_clicked(self) -> None:
        if not self._folder:
            QMessageBox.warning(self, "Session", "Please select a session folder.")
            return
        err = rms_features_preflight(self._folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return
        ori = original_rms_csv(self._folder)
        if not ori.is_file():
            QMessageBox.warning(
                self,
                "Revert",
                f"No frozen baseline found:\n{ori}\n\n"
                "Run a tool that writes RMS (channel fixer or gain adjustment) first.",
            )
            return
        confirm = QMessageBox.question(
            self,
            "Revert to original",
            "Overwrite features/emotions/rmsSignals.csv with the frozen baseline?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            revert_working_to_original(self._folder)
        except OSError as e:
            QMessageBox.critical(self, "Revert", str(e))
            return
        self._refresh_log_tail()
        QMessageBox.information(self, "Revert", "Working RMS file restored from original.")
