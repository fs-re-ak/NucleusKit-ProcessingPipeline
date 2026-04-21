"""
PySide6 GUI: main menu, offline pipeline, settings (theme), and placeholders for other modes.
"""

from __future__ import annotations

import sys


def run_app() -> None:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:
        raise SystemExit(
            "PySide6 is required for the graphical interface. "
            "Install with: pip install 'nucleuskit-pipeline[gui]'"
        ) from e

    from nucleuskit_pipeline.ui.main_window import MainWindow
    from nucleuskit_pipeline.ui.pages.settings_page import load_theme_setting
    from nucleuskit_pipeline.ui.theme import apply_theme

    app = QApplication(sys.argv)
    app.setOrganizationName("REAK")
    app.setApplicationName("NucleusKitPipeline")

    apply_theme(app, load_theme_setting())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
