"""Placeholder for modes not implemented yet."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    go_main_menu = Signal()

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        back = QPushButton("Main menu")
        back.setProperty("secondary", True)
        back.clicked.connect(self.go_main_menu.emit)

        row_h = QHBoxLayout()
        row_h.addStretch(1)
        row_h.addWidget(back)

        hint = QLabel("This mode is not available yet.")
        hint.setProperty("role", "hint")

        title_l = QLabel(title)
        title_l.setProperty("role", "title")

        body = QVBoxLayout(self)
        body.addLayout(row_h)
        body.addWidget(title_l)
        body.addWidget(hint)
        body.addStretch(1)
