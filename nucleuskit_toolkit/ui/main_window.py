"""Main window with stacked navigation between menu, modes, and settings.

Pages are created **on first navigation** (lazy loading) so that heavy
third-party libraries (bleak, joblib, scipy, pandas, …) are not imported
at startup.  Only MainMenuPage is instantiated immediately.
"""

from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

# These two are needed at startup: the menu and the theme helpers.
from nucleuskit_toolkit.ui.pages.main_menu import MainMenuPage
from nucleuskit_toolkit.ui.pages.settings_page import load_theme_setting
from nucleuskit_toolkit.ui.theme import ThemeMode, apply_theme


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Nucleus-Kit Processing Toolkit")
        self.setMinimumSize(720, 520)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # The main menu is shown immediately — it has no heavy dependencies.
        self._menu = MainMenuPage()
        self._stack.addWidget(self._menu)

        # All other pages start as None and are created on first navigation.
        self._offline = None
        self._tools = None
        self._channel_fixer = None
        self._channel_gain = None
        self._revert_original = None
        self._ppg_fixer = None
        self._eeg_regression = None
        self._recording_merge = None
        self._settings = None
        self._realtime = None
        self._playback = None
        self._mqtt = None

        # Connect menu signals to lazy factory methods.
        self._menu.open_offline.connect(self._show_offline)
        self._menu.open_realtime.connect(self._show_realtime)
        self._menu.open_playback.connect(self._show_playback)
        self._menu.open_tools.connect(self._show_tools)
        self._menu.open_mqtt.connect(self._show_mqtt)
        self._menu.open_settings.connect(self._show_settings)

        app = QApplication.instance()
        if app is not None:
            sh = app.styleHints()
            if hasattr(sh, "colorSchemeChanged"):
                sh.colorSchemeChanged.connect(self._on_system_color_scheme_changed)

    # ------------------------------------------------------------------ #
    # Lazy page factories — each creates the page at most once            #
    # ------------------------------------------------------------------ #

    def _show_offline(self) -> None:
        if self._offline is None:
            from nucleuskit_toolkit.ui.pages.offline_page import OfflinePage
            self._offline = OfflinePage()
            self._stack.addWidget(self._offline)
            self._offline.go_main_menu.connect(
                lambda: self._stack.setCurrentWidget(self._menu)
            )
        self._stack.setCurrentWidget(self._offline)

    def _show_tools(self) -> None:
        if self._tools is None:
            from nucleuskit_toolkit.ui.pages.tools_menu_page import ToolsMenuPage
            self._tools = ToolsMenuPage()
            self._stack.addWidget(self._tools)
            self._tools.go_main_menu.connect(
                lambda: self._stack.setCurrentWidget(self._menu)
            )
            self._tools.open_channel_fixer.connect(self._show_channel_fixer)
            self._tools.open_channel_gain.connect(self._show_channel_gain)
            self._tools.open_revert_original.connect(self._show_revert_original)
            self._tools.open_ppg_fixer.connect(self._show_ppg_fixer)
            self._tools.open_eeg_regression.connect(self._show_eeg_regression)
            self._tools.open_recording_merge.connect(self._show_recording_merge)
        self._stack.setCurrentWidget(self._tools)

    def _show_channel_fixer(self) -> None:
        if self._channel_fixer is None:
            from nucleuskit_toolkit.ui.pages.channel_fixer_page import ChannelFixerPage
            self._channel_fixer = ChannelFixerPage()
            self._stack.addWidget(self._channel_fixer)
            # _tools is guaranteed to exist by the time this is reachable.
            self._channel_fixer.go_tools_menu.connect(self._show_tools)
        self._stack.setCurrentWidget(self._channel_fixer)

    def _show_channel_gain(self) -> None:
        if self._channel_gain is None:
            from nucleuskit_toolkit.ui.pages.channel_gain_page import ChannelGainPage
            self._channel_gain = ChannelGainPage()
            self._stack.addWidget(self._channel_gain)
            self._channel_gain.go_tools_menu.connect(self._show_tools)
        self._stack.setCurrentWidget(self._channel_gain)

    def _show_revert_original(self) -> None:
        if self._revert_original is None:
            from nucleuskit_toolkit.ui.pages.revert_original_page import RevertOriginalPage
            self._revert_original = RevertOriginalPage()
            self._stack.addWidget(self._revert_original)
            self._revert_original.go_tools_menu.connect(self._show_tools)
        self._stack.setCurrentWidget(self._revert_original)

    def _show_ppg_fixer(self) -> None:
        if self._ppg_fixer is None:
            from nucleuskit_toolkit.ui.pages.ppg_fixer_page import PpgFixerPage
            self._ppg_fixer = PpgFixerPage()
            self._stack.addWidget(self._ppg_fixer)
            self._ppg_fixer.go_tools_menu.connect(self._show_tools)
        self._stack.setCurrentWidget(self._ppg_fixer)

    def _show_eeg_regression(self) -> None:
        if self._eeg_regression is None:
            from nucleuskit_toolkit.ui.pages.eeg_regression_page import EegRegressionPage
            self._eeg_regression = EegRegressionPage()
            self._stack.addWidget(self._eeg_regression)
            self._eeg_regression.go_tools_menu.connect(self._show_tools)
        self._stack.setCurrentWidget(self._eeg_regression)

    def _show_recording_merge(self) -> None:
        if self._recording_merge is None:
            from nucleuskit_toolkit.ui.pages.recording_merge_page import RecordingMergePage
            self._recording_merge = RecordingMergePage()
            self._stack.addWidget(self._recording_merge)
            self._recording_merge.go_tools_menu.connect(self._show_tools)
        self._stack.setCurrentWidget(self._recording_merge)

    def _show_settings(self) -> None:
        if self._settings is None:
            from nucleuskit_toolkit.ui.pages.settings_page import SettingsPage
            self._settings = SettingsPage()
            self._stack.addWidget(self._settings)
            self._settings.go_main_menu.connect(
                lambda: self._stack.setCurrentWidget(self._menu)
            )
            # theme_changed only needs to be connected once, here on first open.
            self._settings.theme_changed.connect(self._on_theme_changed)
        self._stack.setCurrentWidget(self._settings)

    def _show_realtime(self) -> None:
        if self._realtime is None:
            # This import pulls in bleak, scipy, numpy — deferred until now.
            from nucleuskit_toolkit.ui.pages.realtime_viewer_page import RealtimeViewerPage
            self._realtime = RealtimeViewerPage()
            self._stack.addWidget(self._realtime)
            self._realtime.go_main_menu.connect(self._leave_realtime_to_menu)
        self._stack.setCurrentWidget(self._realtime)

    def _show_playback(self) -> None:
        if self._playback is None:
            from nucleuskit_toolkit.ui.pages.playback_page import PlaybackPage
            self._playback = PlaybackPage()
            self._stack.addWidget(self._playback)
            self._playback.go_main_menu.connect(
                lambda: self._stack.setCurrentWidget(self._menu)
            )
        self._stack.setCurrentWidget(self._playback)

    def _show_mqtt(self) -> None:
        if self._mqtt is None:
            from nucleuskit_toolkit.ui.pages.mqtt_controller_page import MqttControllerPage
            self._mqtt = MqttControllerPage()
            self._stack.addWidget(self._mqtt)
            self._mqtt.go_main_menu.connect(
                lambda: self._stack.setCurrentWidget(self._menu)
            )
        self._stack.setCurrentWidget(self._mqtt)

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def _leave_realtime_to_menu(self) -> None:
        # RealtimeViewerPage validates before emitting; guard defensively.
        if self._realtime is None or not self._realtime.can_navigate_to_main_menu():
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
