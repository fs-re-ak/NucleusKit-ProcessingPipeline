"""Main window: sidebar navigation, a shared page header and lazily created pages.

Pages are created **on first navigation** so that heavy third-party libraries (scipy, pandas, bleak, re-ak-core's
processing modules) are not imported at startup. While a page loads for the first time a loading screen with a
spinner is shown, and its imports run on a background thread so the window stays responsive. After the window
appears, the same libraries are warmed up in the background, so most pages open instantly.

A page can refuse to be left (a device is connected, a run is in progress, unsaved edits): the window calls its
``can_leave()`` before switching, when the page defines one.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, cast

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_toolkit.ui.icons import Spinner, icon
from nucleuskit_toolkit.ui.pages.settings_page import load_theme_setting
from nucleuskit_toolkit.ui.theme import ThemeMode, apply_theme, current_palette
from nucleuskit_toolkit.ui.widgets import IconLabel, label, refresh_icons

REPO_URL = "https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline"
DOCS_URL = f"{REPO_URL}/tree/main/doc"
README_URL = f"{REPO_URL}#"

# Libraries imported by the pages; loaded in the background after startup and before a page is first built.
WARMUP_MODULES = (
    "numpy",
    "pandas",
    "scipy.signal",
    "reak_core.models",
    "reak_core.eeg",
    "reak_core.pipeline",
    "bleak",
    "matplotlib",
)


@dataclass(frozen=True)
class PageSpec:
    key: str
    nav_label: str
    icon: str
    title: str
    subtitle: str
    help: str
    module: str
    factory: str
    nav: str = ""  # sidebar item highlighted for this page (defaults to key)


PAGES: dict[str, PageSpec] = {
    spec.key: spec
    for spec in (
        PageSpec(
            "process", "Process", "play", "Process Sessions",
            "Run the Nucleus-Kit pipeline on a single session or an entire dataset.",
            "run", "nucleuskit_toolkit.ui.pages.offline_page", "OfflinePage",
        ),
        PageSpec(
            "realtime", "Real-time viewer", "radio", "Real-time Viewer",
            "Stream, view and record a Hermes headset or a Shimmer wristband live.",
            "graphical-interface", "nucleuskit_toolkit.ui.pages.realtime_viewer_page", "RealtimeViewerPage",
        ),
        PageSpec(
            "playback", "Playback", "film", "Playback",
            "Review a processed session with its video, metrics and annotations.",
            "graphical-interface", "nucleuskit_toolkit.ui.pages.playback_page", "PlaybackPage",
        ),
        PageSpec(
            "tools", "Tools", "wrench", "Tools",
            "Repair and adjust the signals of processed sessions.",
            "tools-submenu", "nucleuskit_toolkit.ui.pages.tools_menu_page", "ToolsMenuPage",
        ),
        PageSpec(
            "mqtt", "MQTT Controller", "network", "MQTT Controller",
            "Control Vizia Mobile devices over MQTT: status, commands and tags.",
            "graphical-interface", "nucleuskit_toolkit.ui.pages.mqtt_controller_page", "MqttControllerPage",
        ),
        PageSpec(
            "settings", "Settings", "settings", "Settings",
            "Appearance of the application.",
            "graphical-interface", "nucleuskit_toolkit.ui.pages.settings_page", "SettingsPage",
        ),
        PageSpec(
            "about", "About", "info", "About",
            "Version, processing core, emotion models and license.",
            "license", "nucleuskit_toolkit.ui.pages.about_page", "AboutPage",
        ),
        PageSpec(
            "channel_fixer", "", "wrench", "Channel Fixer",
            "Repair a single bad RMS channel of a processed session with the bundled channel-fix model.",
            "tools-submenu", "nucleuskit_toolkit.ui.pages.channel_fixer_page", "ChannelFixerPage", nav="tools",
        ),
        PageSpec(
            "channel_gain", "", "wrench", "Channel Gain Adjustment",
            "Adjust per-channel RMS gain and zero offset against a reference distribution.",
            "tools-submenu", "nucleuskit_toolkit.ui.pages.channel_gain_page", "ChannelGainPage", nav="tools",
        ),
        PageSpec(
            "revert_original", "", "wrench", "Revert to Original",
            "Restore the working RMS features from the frozen baseline in features/emotions/original/.",
            "tools-submenu", "nucleuskit_toolkit.ui.pages.revert_original_page", "RevertOriginalPage", nav="tools",
        ),
        PageSpec(
            "ppg_fixer", "", "wrench", "PPG Fixer",
            "Re-run PPG artifact rejection and heart dynamics on a processed session.",
            "tools-submenu", "nucleuskit_toolkit.ui.pages.ppg_fixer_page", "PpgFixerPage", nav="tools",
        ),
        PageSpec(
            "eeg_regression", "", "wrench", "EEG Regression Denoising",
            "Remove shared muscle noise from the EEG by regression, then recompute cognition.",
            "tools-submenu", "nucleuskit_toolkit.ui.pages.eeg_regression_page", "EegRegressionPage", nav="tools",
        ),
        PageSpec(
            "recording_merge", "", "wrench", "Recording Merger",
            "Join up to 5 recordings into one session, padding the gaps.",
            "tools-submenu", "nucleuskit_toolkit.ui.pages.recording_merge_page", "RecordingMergePage", nav="tools",
        ),
    )
}

MAIN_NAV = ("process", "realtime", "playback", "tools", "mqtt")
BOTTOM_NAV = ("settings", "docs", "about")
TOOL_SIGNALS = {
    "open_channel_fixer": "channel_fixer",
    "open_channel_gain": "channel_gain",
    "open_revert_original": "revert_original",
    "open_ppg_fixer": "ppg_fixer",
    "open_eeg_regression": "eeg_regression",
    "open_recording_merge": "recording_merge",
}


class _ImportWorker(QObject):
    """Imports modules on a background thread (no Qt widgets are created there)."""

    done = Signal()

    def __init__(self, modules: tuple[str, ...]) -> None:
        super().__init__()
        self._modules = modules

    def run(self) -> None:
        for name in self._modules:
            try:
                importlib.import_module(name)
            except Exception:  # noqa: BLE001 — a missing optional library is reported by the page itself
                pass
        self.done.emit()


class PageHeader(QWidget):
    """Page title, one-line subtitle and a discreet Help link on the right."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._help_anchor = ""
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        text = QVBoxLayout()
        text.setSpacing(4)
        self.title = label("", "title")
        self.subtitle = label("", "subtitle", wrap=True)
        text.addWidget(self.title)
        text.addWidget(self.subtitle)
        row.addLayout(text, 1)
        self.help = QPushButton(" Help")
        self.help.setProperty("variant", "link")
        self.help.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help.clicked.connect(self._open_help)
        row.addWidget(self.help, 0, Qt.AlignmentFlag.AlignTop)
        self.refresh_icons()

    def set_page(self, spec: PageSpec) -> None:
        self.title.setText(spec.title)
        self.subtitle.setText(spec.subtitle)
        self._help_anchor = spec.help

    def refresh_icons(self) -> None:
        self.help.setIcon(icon("help", current_palette().muted))

    def _open_help(self) -> None:
        QDesktopServices.openUrl(QUrl(README_URL + self._help_anchor))


class LoadingPage(QWidget):
    """Shown while a page is created for the first time."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.addStretch(2)
        self.spinner = Spinner(36)
        col.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        col.addSpacing(14)
        self.message = label("", "section")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self.message)
        hint = label("The first time a page opens, its libraries are loaded. This can take a few seconds.", "hint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(hint)
        col.addStretch(3)

    def show_for(self, title: str) -> None:
        self.message.setText(f"Loading {title}…")
        self.spinner.set_color(current_palette().accent)
        self.spinner.start()


class Sidebar(QFrame):
    """App identity and navigation; ``navigate`` carries a page key, ``open_docs`` the documentation link."""

    navigate = Signal(str)
    open_docs = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(232)
        col = QVBoxLayout(self)
        col.setContentsMargins(14, 18, 14, 18)
        col.setSpacing(4)

        identity = QHBoxLayout()
        identity.setSpacing(12)
        app_icon = QLabel()
        app_icon.setObjectName("AppIcon")
        app_icon.setFixedSize(44, 44)
        inner = QHBoxLayout(app_icon)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(IconLabel("activity", 24, "accent"), 0, Qt.AlignmentFlag.AlignCenter)
        identity.addWidget(app_icon)
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(label("Nucleus-Kit", "appName"))
        names.addWidget(label("Processing Toolkit", "hint"))
        identity.addLayout(names, 1)
        col.addLayout(identity)
        col.addSpacing(18)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for key in MAIN_NAV:
            col.addWidget(self._make_button(key))
        col.addStretch(1)
        for key in BOTTOM_NAV:
            col.addWidget(self._make_button(key))
        self.refresh_icons()

    def _make_button(self, key: str) -> QPushButton:
        if key == "docs":
            btn = QPushButton("   Documentation")
            btn.clicked.connect(self.open_docs.emit)
        else:
            btn = QPushButton("   " + PAGES[key].nav_label)
            btn.setCheckable(True)
            self._group.addButton(btn)
            btn.clicked.connect(lambda _=False, k=key: self.navigate.emit(k))
        btn.setProperty("variant", "nav")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.buttons[key] = btn
        return btn

    def set_current(self, key: str) -> None:
        btn = self.buttons.get(key)
        if btn is not None:
            btn.setChecked(True)

    def set_locked(self, locked: bool, current: str) -> None:
        for key, btn in self.buttons.items():
            btn.setEnabled(not locked or key in (current, "docs"))

    def refresh_icons(self) -> None:
        p = current_palette()
        icons = {key: PAGES[key].icon for key in MAIN_NAV + ("settings", "about")}
        icons["docs"] = "book"
        for key, btn in self.buttons.items():
            color = p.accent_text if btn.isChecked() else p.text
            btn.setIcon(icon(icons[key], color, 18))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Nucleus-Kit Processing Toolkit")
        self.setMinimumSize(1100, 720)

        self._pages: dict[str, QWidget] = {}
        self._current = ""
        self._pending = ""
        self._warm = False
        self._import_thread: QThread | None = None
        self._import_worker: _ImportWorker | None = None
        self._after_import: list[Callable[[], None]] = []

        self._sidebar = Sidebar()
        self._sidebar.navigate.connect(self.show_page)
        self._sidebar.open_docs.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))

        self._header = PageHeader()
        self._stack = QStackedWidget()
        self._loading = LoadingPage()
        self._stack.addWidget(self._loading)

        content = QWidget()
        content_col = QVBoxLayout(content)
        content_col.setContentsMargins(32, 26, 32, 24)
        content_col.setSpacing(22)
        content_col.addWidget(self._header)
        content_col.addWidget(self._stack, 1)

        central = QWidget()
        row = QHBoxLayout(central)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self._sidebar)
        row.addWidget(content, 1)
        self.setCentralWidget(central)

        app = QApplication.instance()
        if app is not None:
            sh = app.styleHints()
            if hasattr(sh, "colorSchemeChanged"):
                sh.colorSchemeChanged.connect(self._on_system_color_scheme_changed)

        self.show_page("process")

    # ------------------------------------------------------------------ #
    # Navigation                                                           #
    # ------------------------------------------------------------------ #

    def show_page(self, key: str) -> None:
        if key == self._current and key in self._pages:
            self._sidebar.set_current(PAGES[key].nav or key)
            return
        current = self._pages.get(self._current)
        can_leave = getattr(current, "can_leave", None)
        if current is not None and callable(can_leave) and not can_leave():
            self._sidebar.set_current(PAGES[self._current].nav or self._current)
            self._sidebar.refresh_icons()
            return
        spec = PAGES[key]
        self._header.set_page(spec)
        self._sidebar.set_current(spec.nav or key)
        self._sidebar.refresh_icons()
        self._current = key
        if key in self._pages:
            self._stack.setCurrentWidget(self._pages[key])
            return
        self._pending = key
        self._loading.show_for(spec.title)
        self._stack.setCurrentWidget(self._loading)
        self._run_imports(lambda: self._build_page(key))

    def _build_page(self, key: str) -> None:
        if key in self._pages:
            return
        spec = PAGES[key]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            module = importlib.import_module(spec.module)
            page = getattr(module, spec.factory)()
        finally:
            QApplication.restoreOverrideCursor()
        self._pages[key] = page
        self._stack.addWidget(page)
        self._connect_page(key, page)
        if self._pending == key:
            self._loading.spinner.stop()
            self._stack.setCurrentWidget(page)

    def _connect_page(self, key: str, page: QWidget) -> None:
        if key == "tools":
            for signal_name, target in TOOL_SIGNALS.items():
                getattr(page, signal_name).connect(lambda t=target: self.show_page(t))
        if hasattr(page, "go_tools_menu"):
            page.go_tools_menu.connect(lambda: self.show_page("tools"))
        if key == "settings":
            page.theme_changed.connect(self._on_theme_changed)
        if hasattr(page, "processing_changed"):
            page.processing_changed.connect(lambda running, k=key: self._sidebar.set_locked(running, k))

    # ------------------------------------------------------------------ #
    # Background imports                                                   #
    # ------------------------------------------------------------------ #

    def start_warmup(self) -> None:
        """Load the pages' libraries in the background (called once the window is visible)."""
        self._run_imports(None)

    def _run_imports(self, then: Callable[[], None] | None) -> None:
        if then is not None:
            if self._warm:
                then()
                return
            self._after_import.append(then)
        if self._import_thread is not None:
            return
        thread = QThread(self)
        worker = _ImportWorker(WARMUP_MODULES)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_imports_done)
        worker.done.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        self._import_thread = thread
        self._import_worker = worker
        thread.start()

    def _on_imports_done(self) -> None:
        self._warm = True
        callbacks, self._after_import = self._after_import, []
        for callback in callbacks:
            callback()

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def _on_theme_changed(self, mode: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        apply_theme(app, cast(ThemeMode, mode))
        self._refresh_theme()

    def _on_system_color_scheme_changed(self, _scheme: Qt.ColorScheme | int) -> None:
        if load_theme_setting() != "system":
            return
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, "system")
            self._refresh_theme()

    def _refresh_theme(self) -> None:
        self._sidebar.refresh_icons()
        self._header.refresh_icons()
        refresh_icons(self)
        for page in self._pages.values():
            hook = getattr(page, "refresh_theme", None)
            if callable(hook):
                hook()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._import_thread is not None and self._import_thread.isRunning():
            self._import_thread.wait(5000)
        super().closeEvent(event)
