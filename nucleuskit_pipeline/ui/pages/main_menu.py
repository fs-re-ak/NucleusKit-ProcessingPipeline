"""Branded main menu: mode selection and settings."""

from __future__ import annotations

from importlib import resources

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class MainMenuPage(QWidget):
    open_realtime = Signal()
    open_offline = Signal()
    open_playback = Signal()
    open_tools = Signal()
    open_mqtt = Signal()
    open_settings = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("background: transparent;")
        pm = self._load_logo()
        if pm and not pm.isNull():
            max_logo = QSize(400, 160)
            scaled = pm.scaled(max_logo, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled)
        else:
            logo_label.setText("Nucleus-Kit")
            logo_label.setProperty("role", "title")

        subtitle = QLabel("Processing pipeline")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setProperty("role", "hint")

        def big_button(text: str, slot) -> QPushButton:
            b = QPushButton(text)
            b.setMinimumHeight(52)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.clicked.connect(slot)
            return b

        self._menu_band = QWidget()
        menu_col = QVBoxLayout(self._menu_band)
        menu_col.setContentsMargins(0, 0, 0, 0)
        menu_col.setSpacing(14)
        menu_col.addWidget(big_button("Real-time viewer", self.open_realtime.emit))
        menu_col.addWidget(big_button("Offline processing", self.open_offline.emit))
        menu_col.addWidget(big_button("Playback mode", self.open_playback.emit))
        menu_col.addWidget(big_button("Tools", self.open_tools.emit))
        menu_col.addWidget(big_button("MQTT Controller", self.open_mqtt.emit))
        menu_col.addWidget(big_button("Settings", self.open_settings.emit))

        outer = QVBoxLayout(self)
        outer.addStretch(1)
        outer.addWidget(logo_label)
        outer.addWidget(subtitle)
        outer.addSpacing(24)
        outer.addWidget(self._menu_band, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(2)

        self._update_menu_band_max_width()

    def _screen_width_px(self) -> int:
        screen = self.screen()
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return 1920
        return screen.availableGeometry().width()

    def _update_menu_band_max_width(self) -> None:
        w = self._screen_width_px()
        self._menu_band.setMaximumWidth(max(160, int(0.33 * w)))

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        self._update_menu_band_max_width()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._update_menu_band_max_width()

    def _load_logo(self) -> QPixmap | None:
        try:
            root = resources.files("nucleuskit_pipeline.ui.resources.branding")
            path = root / "logo.png"
            data = path.read_bytes()
            pm = QPixmap()
            if pm.loadFromData(data):
                return pm
        except (OSError, FileNotFoundError, ModuleNotFoundError):
            pass
        return None
