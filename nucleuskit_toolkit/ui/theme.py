"""Application theming (RE-AK palette) for PySide6."""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ThemeMode = Literal["light", "dark", "system"]

NAVY = "#121434"
PINK = "#eb4162"
YELLOW = "#fae460"
ORANGE = "#e96b28"
GREEN = "#46b14a"

LIGHT_BG = "#f4f4f8"
LIGHT_PANEL = "#ffffff"
LIGHT_MUTED = "#6b6d85"
DARK_BG = NAVY
DARK_PANEL = "#1a1d3d"
DARK_MUTED = "#a8aac4"


def _base_qss() -> str:
    return f"""
    * {{ font-size: 13px; }}
    QGroupBox {{
        font-weight: 600;
        margin-top: 12px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 6px;
    }}
    QLineEdit, QPlainTextEdit, QComboBox {{
        border-radius: 8px;
        padding: 8px 10px;
        min-height: 20px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 28px;
    }}
    QProgressBar {{
        border-radius: 6px;
        text-align: center;
        min-height: 14px;
    }}
    QProgressBar::chunk {{
        border-radius: 6px;
    }}
    QPushButton {{
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        min-height: 22px;
    }}
    QPushButton[secondary="true"] {{
        background-color: transparent;
        border: 2px solid {PINK};
        color: {PINK};
        font-weight: 600;
    }}
    QPushButton[secondary="true"]:hover {{
        background-color: rgba(235, 65, 98, 0.12);
    }}
    QPushButton[secondary="true"]:pressed {{
        background-color: rgba(235, 65, 98, 0.22);
    }}
    QPushButton[secondary="true"]:disabled {{
        border-color: #888899;
        color: #888899;
        background-color: transparent;
    }}
    """


def dark_stylesheet() -> str:
    return (
        _base_qss()
        + f"""
    QWidget {{
        background-color: {DARK_BG};
        color: #ececf4;
    }}
    QMainWindow {{
        background-color: {DARK_BG};
    }}
    QGroupBox {{
        color: #ececf4;
        border: 1px solid #2e3358;
        border-radius: 10px;
        margin-top: 12px;
        padding: 14px 10px 10px 10px;
        background-color: {DARK_PANEL};
    }}
    QLineEdit, QPlainTextEdit {{
        background-color: #22264a;
        border: 1px solid #343960;
        color: #ececf4;
    }}
    QComboBox {{
        background-color: #22264a;
        border: 1px solid #343960;
        color: #ececf4;
    }}
    QComboBox QAbstractItemView {{
        background-color: #22264a;
        color: #ececf4;
        selection-background-color: {ORANGE};
        selection-color: #ffffff;
    }}
    QPlainTextEdit {{
        background-color: #161832;
    }}
    QProgressBar {{
        background-color: #2a2f55;
        border: 1px solid #343960;
        color: #ececf4;
    }}
    QProgressBar::chunk {{
        background-color: {ORANGE};
    }}
    QPushButton {{
        background-color: {ORANGE};
        color: #ffffff;
        border: none;
    }}
    QPushButton:hover {{
        background-color: #ff7d3a;
    }}
    QPushButton:pressed {{
        background-color: #c55a22;
    }}
    QPushButton:disabled {{
        background-color: #4a4e6e;
        color: {DARK_MUTED};
    }}
    QLabel[role="hint"] {{
        color: {DARK_MUTED};
    }}
    QLabel[role="title"] {{
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
    }}
    """
    )


def light_stylesheet() -> str:
    return (
        _base_qss()
        + f"""
    QWidget {{
        background-color: {LIGHT_BG};
        color: {NAVY};
    }}
    QMainWindow {{
        background-color: {LIGHT_BG};
    }}
    QGroupBox {{
        color: {NAVY};
        border: 1px solid #d8d9e6;
        border-radius: 10px;
        margin-top: 12px;
        padding: 14px 10px 10px 10px;
        background-color: {LIGHT_PANEL};
    }}
    QLineEdit, QPlainTextEdit {{
        background-color: {LIGHT_PANEL};
        border: 1px solid #c5c7d8;
        color: {NAVY};
    }}
    QComboBox {{
        background-color: {LIGHT_PANEL};
        border: 1px solid #c5c7d8;
        color: {NAVY};
    }}
    QComboBox QAbstractItemView {{
        background-color: {LIGHT_PANEL};
        color: {NAVY};
        selection-background-color: {ORANGE};
        selection-color: #ffffff;
    }}
    QPlainTextEdit {{
        background-color: #fafbff;
    }}
    QProgressBar {{
        background-color: #e4e6f0;
        border: 1px solid #c5c7d8;
        color: {NAVY};
    }}
    QProgressBar::chunk {{
        background-color: {ORANGE};
    }}
    QPushButton {{
        background-color: {ORANGE};
        color: #ffffff;
        border: none;
    }}
    QPushButton:hover {{
        background-color: #ff7d3a;
    }}
    QPushButton:pressed {{
        background-color: #c55a22;
    }}
    QPushButton:disabled {{
        background-color: #c5c7d8;
        color: {LIGHT_MUTED};
    }}
    QLabel[role="hint"] {{
        color: {LIGHT_MUTED};
    }}
    QLabel[role="title"] {{
        font-size: 22px;
        font-weight: 700;
        color: {NAVY};
    }}
    """
    )


def resolve_effective_theme(mode: ThemeMode) -> Literal["light", "dark"]:
    """Map 'system' to light or dark using Qt style hints."""
    if mode != "system":
        return mode
    hints = QApplication.styleHints()
    scheme = hints.colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return "dark"
    if scheme == Qt.ColorScheme.Light:
        return "light"
    return "dark"


def apply_theme(app: QApplication, mode: ThemeMode) -> Literal["light", "dark"]:
    """Apply Fusion + stylesheet; returns the resolved light/dark used."""
    app.setStyle("Fusion")
    effective = resolve_effective_theme(mode)
    app.setStyleSheet(dark_stylesheet() if effective == "dark" else light_stylesheet())

    pal = QPalette()
    if effective == "dark":
        pal.setColor(QPalette.Window, QColor(DARK_BG))
        pal.setColor(QPalette.WindowText, QColor("#ececf4"))
        pal.setColor(QPalette.Base, QColor("#161832"))
        pal.setColor(QPalette.Text, QColor("#ececf4"))
        pal.setColor(QPalette.Button, QColor(ORANGE))
        pal.setColor(QPalette.ButtonText, QColor("#ffffff"))
    else:
        pal.setColor(QPalette.Window, QColor(LIGHT_BG))
        pal.setColor(QPalette.WindowText, QColor(NAVY))
        pal.setColor(QPalette.Base, QColor("#ffffff"))
        pal.setColor(QPalette.Text, QColor(NAVY))
        pal.setColor(QPalette.Button, QColor(ORANGE))
        pal.setColor(QPalette.ButtonText, QColor("#ffffff"))
    app.setPalette(pal)
    return effective
