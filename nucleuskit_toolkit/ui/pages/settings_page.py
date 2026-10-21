"""Appearance and other settings."""

from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QVBoxLayout, QWidget

from nucleuskit_toolkit.ui.theme import ThemeMode
from nucleuskit_toolkit.ui.widgets import Card, SectionHeader, label

ORGANIZATION = "REAK"
APPLICATION = "NucleusKitToolkit"
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("Light", "light")
        self._theme_combo.addItem("Dark", "dark")
        self._theme_combo.addItem("System", "system")
        self._theme_combo.setMinimumWidth(220)

        current = load_theme_setting()
        idx = max(0, self._theme_combo.findData(current))
        self._theme_combo.setCurrentIndex(idx)

        appearance = Card()
        appearance.setMaximumWidth(760)
        appearance.body.addWidget(SectionHeader("Appearance", "Colours of the application.", icon="settings"))
        row = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(label("Theme", "strong"))
        text.addWidget(label("System follows the light or dark mode of the operating system.", "hint"))
        row.addLayout(text, 1)
        row.addWidget(self._theme_combo)
        appearance.body.addLayout(row)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(appearance)
        layout.addStretch(1)

        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _index: int) -> None:
        mode = self._theme_combo.currentData()
        if mode:
            save_theme_setting(mode)
            self.theme_changed.emit(mode)
