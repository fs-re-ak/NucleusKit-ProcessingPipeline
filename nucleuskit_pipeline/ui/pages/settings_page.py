"""Appearance and other settings."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.ui.theme import ThemeMode

ORGANIZATION = "REAK"
APPLICATION = "NucleusKitPipeline"
SETTINGS_THEME_KEY = "appearance/theme"


def load_theme_setting() -> ThemeMode:
    s = QSettings(ORGANIZATION, APPLICATION)
    raw = s.value(SETTINGS_THEME_KEY, "system")
    if raw in ("light", "dark", "system"):
        return raw  # type: ignore[return-value]
    return "system"


def save_theme_setting(mode: ThemeMode) -> None:
    s = QSettings(ORGANIZATION, APPLICATION)
    s.setValue(SETTINGS_THEME_KEY, mode)


class SettingsPage(QWidget):
    """Theme selection persisted with QSettings."""

    theme_changed = Signal(str)
    go_main_menu = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Light", "light")
        self._theme_combo.addItem("Dark", "dark")
        self._theme_combo.addItem("System", "system")

        current = load_theme_setting()
        idx = max(0, self._theme_combo.findData(current))
        self._theme_combo.setCurrentIndex(idx)

        appearance = QGroupBox("Appearance")
        form = QFormLayout(appearance)
        form.addRow("Theme", self._theme_combo)

        back = QPushButton("Main menu")
        back.setProperty("secondary", True)
        back.clicked.connect(self.go_main_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(back)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        title = QLabel("Settings")
        title.setProperty("role", "title")
        layout.addWidget(title)
        layout.addWidget(appearance)
        layout.addStretch(1)

        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _index: int) -> None:
        mode = self._theme_combo.currentData()
        if mode:
            save_theme_setting(mode)
            self.theme_changed.emit(mode)
