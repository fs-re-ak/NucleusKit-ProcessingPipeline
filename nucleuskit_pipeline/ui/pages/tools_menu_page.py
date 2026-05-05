"""Submenu for auxiliary tools (Channel Fixer, …)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class ToolsMenuPage(QWidget):
    go_main_menu = Signal()
    open_channel_fixer = Signal()
    open_channel_gain = Signal()
    open_revert_original = Signal()
    open_ppg_fixer = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._back = QPushButton("Main menu")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_main_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._back)

        def big_button(text: str, slot) -> QPushButton:
            b = QPushButton(text)
            b.setMinimumHeight(52)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(slot)
            return b

        col = QVBoxLayout()
        col.setSpacing(14)
        col.addWidget(big_button("Channel Fixer", self.open_channel_fixer.emit))
        col.addWidget(big_button("Channel gain adjustment", self.open_channel_gain.emit))
        col.addWidget(big_button("Revert to original", self.open_revert_original.emit))
        col.addWidget(big_button("PPG Fixer", self.open_ppg_fixer.emit))

        outer = QVBoxLayout(self)
        outer.addLayout(top)
        outer.addStretch(1)
        outer.addLayout(col)
        outer.addStretch(2)
