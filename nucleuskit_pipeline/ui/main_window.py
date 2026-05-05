"""Main window with stacked navigation between menu, modes, and settings."""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from nucleuskit_pipeline.ui.pages.channel_fixer_page import ChannelFixerPage
from nucleuskit_pipeline.ui.pages.channel_gain_page import ChannelGainPage
from nucleuskit_pipeline.ui.pages.main_menu import MainMenuPage
from nucleuskit_pipeline.ui.pages.mqtt_controller_page import MqttControllerPage
from nucleuskit_pipeline.ui.pages.offline_page import OfflinePage
from nucleuskit_pipeline.ui.pages.playback_page import PlaybackPage
from nucleuskit_pipeline.ui.pages.ppg_fixer_page import PpgFixerPage
from nucleuskit_pipeline.ui.pages.realtime_viewer_page import RealtimeViewerPage
from nucleuskit_pipeline.ui.pages.revert_original_page import RevertOriginalPage
from nucleuskit_pipeline.ui.pages.settings_page import SettingsPage, load_theme_setting
from nucleuskit_pipeline.ui.pages.tools_menu_page import ToolsMenuPage
from nucleuskit_pipeline.ui.theme import ThemeMode, apply_theme


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Nucleus-Kit Processing Pipeline")
        self.setMinimumSize(720, 520)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._menu = MainMenuPage()
        self._offline = OfflinePage()
        self._tools = ToolsMenuPage()
        self._channel_fixer = ChannelFixerPage()
        self._channel_gain = ChannelGainPage()
        self._revert_original = RevertOriginalPage()
        self._ppg_fixer = PpgFixerPage()
        self._settings = SettingsPage()
        self._realtime = RealtimeViewerPage()
        self._playback = PlaybackPage()
        self._mqtt = MqttControllerPage()

        self._stack.addWidget(self._menu)
        self._stack.addWidget(self._offline)
        self._stack.addWidget(self._tools)
        self._stack.addWidget(self._channel_fixer)
        self._stack.addWidget(self._channel_gain)
        self._stack.addWidget(self._revert_original)
        self._stack.addWidget(self._ppg_fixer)
        self._stack.addWidget(self._settings)
        self._stack.addWidget(self._realtime)
        self._stack.addWidget(self._playback)
        self._stack.addWidget(self._mqtt)

        self._menu.open_realtime.connect(lambda: self._stack.setCurrentWidget(self._realtime))
        self._menu.open_offline.connect(lambda: self._stack.setCurrentWidget(self._offline))
        self._menu.open_playback.connect(lambda: self._stack.setCurrentWidget(self._playback))
        self._menu.open_tools.connect(lambda: self._stack.setCurrentWidget(self._tools))
        self._menu.open_mqtt.connect(lambda: self._stack.setCurrentWidget(self._mqtt))
        self._menu.open_settings.connect(lambda: self._stack.setCurrentWidget(self._settings))

        self._mqtt.go_main_menu.connect(lambda: self._stack.setCurrentWidget(self._menu))

        self._tools.go_main_menu.connect(lambda: self._stack.setCurrentWidget(self._menu))
        self._tools.open_channel_fixer.connect(lambda: self._stack.setCurrentWidget(self._channel_fixer))
        self._tools.open_channel_gain.connect(lambda: self._stack.setCurrentWidget(self._channel_gain))
        self._tools.open_revert_original.connect(lambda: self._stack.setCurrentWidget(self._revert_original))
        self._tools.open_ppg_fixer.connect(lambda: self._stack.setCurrentWidget(self._ppg_fixer))
        self._channel_fixer.go_tools_menu.connect(lambda: self._stack.setCurrentWidget(self._tools))
        self._channel_gain.go_tools_menu.connect(lambda: self._stack.setCurrentWidget(self._tools))
        self._revert_original.go_tools_menu.connect(lambda: self._stack.setCurrentWidget(self._tools))
        self._ppg_fixer.go_tools_menu.connect(lambda: self._stack.setCurrentWidget(self._tools))

        self._offline.go_main_menu.connect(lambda: self._stack.setCurrentWidget(self._menu))
        self._settings.go_main_menu.connect(lambda: self._stack.setCurrentWidget(self._menu))
        self._realtime.go_main_menu.connect(self._leave_realtime_to_menu)
        self._playback.go_main_menu.connect(lambda: self._stack.setCurrentWidget(self._menu))

        self._settings.theme_changed.connect(self._on_theme_changed)

        app = QApplication.instance()
        if app is not None:
            sh = app.styleHints()
            if hasattr(sh, "colorSchemeChanged"):
                sh.colorSchemeChanged.connect(self._on_system_color_scheme_changed)

    def _leave_realtime_to_menu(self) -> None:
        # RealtimeViewerPage validates before emitting; keep a defensive check here.
        if not self._realtime.can_navigate_to_main_menu():
            return
        self._stack.setCurrentWidget(self._menu)

    def _on_theme_changed(self, mode: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        apply_theme(app, cast(ThemeMode, mode))

    def _on_system_color_scheme_changed(self, _scheme: Qt.ColorScheme | int) -> None:
        if load_theme_setting() != "system":
            return
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, "system")
