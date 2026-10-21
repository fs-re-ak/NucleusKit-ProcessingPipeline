"""
PySide6 GUI: sidebar navigation between processing, real-time viewer, playback, tools, MQTT, settings and about.
"""

from __future__ import annotations

import sys


def run_app() -> None:
    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        raise SystemExit(
            "PySide6 is required for the graphical interface. "
            "Install with: pip install 'nucleuskit-toolkit[gui]'"
        ) from e

    from nucleuskit_toolkit.ui.main_window import MainWindow
    from nucleuskit_toolkit.ui.pages.settings_page import load_theme_setting
    from nucleuskit_toolkit.ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setOrganizationName("REAK")
    app.setApplicationName("NucleusKitToolkit")

    apply_theme(app, load_theme_setting())

    window = MainWindow()
    window.show()
    # Load the pages' libraries in the background once the window is on screen.
    QTimer.singleShot(300, window.start_warmup)
    sys.exit(app.exec())
