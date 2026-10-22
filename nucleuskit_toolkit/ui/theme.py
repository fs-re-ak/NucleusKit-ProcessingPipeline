"""Application theming for PySide6: dark and light palettes, one style sheet for every page.

Colour roles (dark / light):

- Background: page and sidebar.
- Surface: cards, inputs, pills, secondary buttons.
- Muted: descriptions, helper text, placeholders, "Not selected".
- Accent: selection, step numbers, checked states, active navigation, Enabled pills, primary buttons.
- Run: the Run pipeline action only (``QPushButton[variant="run"]``); nothing else uses it.

Widgets opt into the design with dynamic properties: ``card`` (QFrame), ``variant`` (QPushButton: ``primary``,
``secondary``, ``run``, ``link``, ``nav``, ``choice``), ``role`` (QLabel: ``title``, ``subtitle``, ``cardTitle``,
``hint``, ``section``, ``muted``, ``placeholder``, ``error``, ``ok``, ``badge``) and ``pill`` (QLabel: ``on``, ``off``).
The legacy ``secondary="true"`` property of the older pages maps to the outlined secondary button.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from nucleuskit_toolkit.ui.icons import icon_file

ThemeMode = Literal["light", "dark", "system"]


@dataclass(frozen=True)
class Palette:
    name: Literal["light", "dark"]
    background: str
    surface: str
    surface_hover: str
    border: str
    text: str
    muted: str
    accent: str
    accent_hover: str
    accent_soft: str
    accent_text: str
    run: str
    run_hover: str
    run_pressed: str
    disabled_text: str
    error: str
    ok: str
    log_background: str


DARK = Palette(
    name="dark",
    background="#10131C",
    surface="#1C2230",
    surface_hover="#242C3D",
    border="#2A3245",
    text="#EEF2F8",
    muted="#AAB6CC",
    accent="#4F85EE",
    accent_hover="#6A98F1",
    accent_soft="rgba(79, 133, 238, 0.16)",
    accent_text="#8DB0F5",
    run="#F47735",
    run_hover="#F68A50",
    run_pressed="#DB6529",
    disabled_text="#6E7890",
    error="#F27C7C",
    ok="#5FC48A",
    log_background="#141925",
)

LIGHT = Palette(
    name="light",
    background="#F4F6FA",
    surface="#FFFFFF",
    surface_hover="#EEF2F8",
    border="#DCE2EC",
    text="#10131C",
    muted="#5B6579",
    accent="#4F85EE",
    accent_hover="#3D74E0",
    accent_soft="rgba(79, 133, 238, 0.12)",
    accent_text="#2F64CF",
    run="#F47735",
    run_hover="#F68A50",
    run_pressed="#DB6529",
    disabled_text="#9AA3B5",
    error="#C93C3C",
    ok="#2E8B57",
    log_background="#F8FAFD",
)

_current: Palette = DARK


def current_palette() -> Palette:
    """The palette applied last (dark before the first ``apply_theme``)."""
    return _current


def stylesheet(p: Palette) -> str:
    check = icon_file("check", "#FFFFFF", 3.0)
    chevron = icon_file("chevron-down", p.muted)
    chevron_disabled = icon_file("chevron-down", p.disabled_text)
    chevron_up = icon_file("chevron-up", p.muted)
    return f"""
    * {{ font-size: 13px; }}
    QWidget {{ color: {p.text}; }}
    QMainWindow, QDialog, QMessageBox {{ background-color: {p.background}; }}
    QLabel, QCheckBox, QRadioButton {{ background-color: transparent; }}
    QToolTip {{ background-color: {p.surface}; color: {p.text}; border: 1px solid {p.border}; padding: 6px; }}

    /* Cards and sidebar */
    QFrame[card="true"] {{ background-color: {p.surface}; border: 1px solid {p.border}; border-radius: 12px; }}
    QFrame[divider="true"] {{ background-color: {p.border}; border: none; max-height: 1px; min-height: 1px; }}
    QFrame#Sidebar {{ background-color: {p.background}; border: none; border-right: 1px solid {p.border}; }}
    QLabel#StepNumber {{
        background-color: {p.accent}; color: #FFFFFF; border-radius: 16px; font-size: 15px; font-weight: 700;
    }}
    QLabel#HeaderIcon {{ background-color: {p.accent_soft}; border-radius: 18px; }}
    QLabel#AppIcon {{ background-color: {p.surface}; border: 1px solid {p.border}; border-radius: 10px; }}

    /* Text roles */
    QLabel[role="title"] {{ font-size: 26px; font-weight: 700; color: {p.text}; }}
    QLabel[role="subtitle"] {{ font-size: 14px; color: {p.muted}; }}
    QLabel[role="cardTitle"] {{ font-size: 16px; font-weight: 600; color: {p.text}; }}
    QLabel[role="section"] {{ font-size: 14px; font-weight: 600; color: {p.text}; }}
    QLabel[role="strong"] {{ font-weight: 600; color: {p.text}; }}
    QLabel[role="hint"], QLabel[role="muted"] {{ color: {p.muted}; }}
    QLabel[role="placeholder"] {{ color: {p.muted}; font-style: italic; }}
    QLabel[role="error"] {{ color: {p.error}; }}
    QLabel[role="ok"] {{ color: {p.ok}; }}
    QLabel[role="appName"] {{ font-size: 16px; font-weight: 700; color: {p.text}; }}
    QLabel[pill="on"] {{
        background-color: {p.accent}; color: #FFFFFF; border-radius: 8px; padding: 5px 12px; font-weight: 600;
    }}
    QLabel[pill="off"], QLabel[pill="value"] {{
        background-color: {p.surface_hover}; color: {p.text}; border-radius: 8px; padding: 5px 12px;
    }}
    QLabel:disabled {{ color: {p.disabled_text}; }}

    /* Inputs */
    QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {p.surface}; border: 1px solid {p.border}; border-radius: 8px; padding: 8px 10px;
        color: {p.text}; selection-background-color: {p.accent}; selection-color: #FFFFFF;
    }}
    QLineEdit {{ min-height: 22px; }}
    QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {p.accent};
    }}
    QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{ color: {p.disabled_text}; }}
    QLineEdit[invalid="true"] {{ border: 1px solid {p.error}; }}
    QFrame[card="true"] QLineEdit, QFrame[card="true"] QComboBox {{ background-color: {p.background}; }}
    QComboBox {{ padding-right: 34px; min-height: 20px; }}
    QComboBox::drop-down {{ border: none; width: 32px; }}
    QComboBox::down-arrow {{ image: url("{chevron}"); width: 16px; height: 16px; }}
    QComboBox::down-arrow:disabled {{ image: url("{chevron_disabled}"); }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 26px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; width: 22px; border: none; background: transparent;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{chevron_up}"); width: 12px; height: 12px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{chevron}"); width: 12px; height: 12px; }}
    QComboBox QAbstractItemView {{
        background-color: {p.surface}; color: {p.text}; border: 1px solid {p.border};
        selection-background-color: {p.accent}; selection-color: #FFFFFF; outline: none; padding: 4px;
    }}
    QPlainTextEdit#Log {{
        background-color: {p.log_background}; font-family: "Cascadia Mono", "Consolas", "Menlo", monospace;
        font-size: 12px; padding: 10px 12px;
    }}

    /* Check boxes and radio buttons: visible borders when unchecked, accent fill when checked */
    QCheckBox, QRadioButton {{ spacing: 10px; background-color: transparent; }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px; height: 18px; border: 2px solid {p.muted}; background-color: transparent;
    }}
    QCheckBox::indicator {{ border-radius: 5px; }}
    QRadioButton::indicator {{ border-radius: 11px; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {p.accent}; }}
    QCheckBox::indicator:checked {{ background-color: {p.accent}; border-color: {p.accent}; image: url("{check}"); }}
    QRadioButton::indicator:checked {{
        border: 2px solid {p.accent}; background-color: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {p.accent}, stop:0.45 {p.accent}, stop:0.5 transparent, stop:1 transparent);
    }}
    QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{ border-color: {p.disabled_text}; }}
    QCheckBox:disabled, QRadioButton:disabled {{ color: {p.disabled_text}; }}

    /* Buttons: primary (accent, default), secondary (outlined), run (orange, Run pipeline only), link, nav */
    QPushButton {{
        background-color: {p.accent}; color: #FFFFFF; border: none; border-radius: 8px;
        padding: 9px 18px; font-weight: 600; min-height: 20px;
    }}
    QPushButton:hover {{ background-color: {p.accent_hover}; }}
    QPushButton:disabled {{ background-color: {p.surface_hover}; color: {p.disabled_text}; }}
    QPushButton[secondary="true"], QPushButton[variant="secondary"] {{
        background-color: transparent; color: {p.text}; border: 1px solid {p.border}; font-weight: 500;
    }}
    QPushButton[secondary="true"]:hover, QPushButton[variant="secondary"]:hover {{
        background-color: {p.surface_hover}; border-color: {p.muted};
    }}
    QPushButton[secondary="true"]:disabled, QPushButton[variant="secondary"]:disabled {{
        background-color: transparent; color: {p.disabled_text}; border-color: {p.border};
    }}
    QPushButton[variant="run"] {{
        background-color: {p.run}; color: #FFFFFF; border-radius: 12px; padding: 0; min-height: 56px;
    }}
    QPushButton[variant="run"]:hover {{ background-color: {p.run_hover}; }}
    QPushButton[variant="run"]:pressed {{ background-color: {p.run_pressed}; }}
    QPushButton[variant="run"]:disabled {{ background-color: {p.surface_hover}; color: {p.disabled_text}; }}
    QPushButton[variant="cancel"] {{
        background-color: transparent; color: {p.text}; border: 2px solid {p.muted}; border-radius: 12px;
        padding: 0; min-height: 52px;
    }}
    QPushButton[variant="cancel"]:hover {{ background-color: {p.surface_hover}; }}
    QPushButton[variant="link"] {{
        background-color: transparent; color: {p.muted}; border: none; padding: 4px 6px; font-weight: 500;
    }}
    QPushButton[variant="link"]:hover {{ color: {p.text}; }}
    QPushButton[variant="nav"] {{
        background-color: transparent; color: {p.text}; border: none; border-radius: 10px;
        padding: 10px 14px; text-align: left; font-weight: 500;
    }}
    QPushButton[variant="nav"]:hover {{ background-color: {p.surface}; }}
    QPushButton[variant="nav"]:checked {{
        background-color: {p.accent_soft}; color: {p.accent_text}; font-weight: 600;
    }}
    QPushButton[variant="nav"]:disabled {{ color: {p.disabled_text}; }}

    /* Selectable mode cards */
    QFrame[choice="true"] {{ background-color: transparent; border: 1px solid {p.border}; border-radius: 10px; }}
    QFrame[choice="true"]:hover {{ border-color: {p.muted}; }}
    QFrame[choice="true"][selected="true"] {{ background-color: {p.accent_soft}; border: 2px solid {p.accent}; }}
    QFrame[choice="true"]:disabled {{ border-color: {p.border}; }}

    /* Legacy group boxes (older pages) */
    QGroupBox {{
        font-weight: 600; color: {p.text}; background-color: {p.surface}; border: 1px solid {p.border};
        border-radius: 12px; margin-top: 14px; padding: 16px 12px 12px 12px;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {p.text}; }}
    QGroupBox QLineEdit, QGroupBox QComboBox, QGroupBox QPlainTextEdit {{ background-color: {p.background}; }}

    /* Lists, tables, tabs, progress, scroll bars, splitters */
    QListWidget, QTableWidget, QTableView, QTreeView {{
        background-color: {p.surface}; border: 1px solid {p.border}; border-radius: 8px; color: {p.text};
        alternate-background-color: {p.surface_hover}; gridline-color: {p.border};
    }}
    QListWidget::item {{ padding: 8px 10px; border-radius: 6px; }}
    QListWidget::item:selected, QTableView::item:selected {{ background-color: {p.accent}; color: #FFFFFF; }}
    QListWidget::item:hover:!selected {{ background-color: {p.surface_hover}; }}
    QHeaderView::section {{
        background-color: {p.surface_hover}; color: {p.text}; border: none; padding: 6px 8px; font-weight: 600;
    }}
    QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: 8px; top: -1px; }}
    QTabBar::tab {{
        background-color: transparent; color: {p.muted}; padding: 8px 14px; border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}
    QProgressBar {{
        background-color: {p.surface_hover}; border: none; border-radius: 6px; color: {p.text};
        text-align: center; min-height: 12px;
    }}
    QProgressBar::chunk {{ background-color: {p.accent}; border-radius: 6px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 4px; min-height: 30px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {p.border}; border-radius: 4px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    QSplitter::handle {{ background-color: {p.border}; }}
    QMenu {{ background-color: {p.surface}; border: 1px solid {p.border}; padding: 4px; }}
    QMenu::item:selected {{ background-color: {p.accent}; color: #FFFFFF; }}
    QMessageBox QPushButton {{ min-width: 80px; }}
    """


def resolve_effective_theme(mode: ThemeMode) -> Literal["light", "dark"]:
    """Map 'system' to light or dark using Qt style hints."""
    if mode != "system":
        return mode
    scheme = QApplication.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Light:
        return "light"
    return "dark"


def apply_theme(app: QApplication, mode: ThemeMode) -> Literal["light", "dark"]:
    """Apply Fusion, the palette and the style sheet; returns the resolved light/dark used."""
    global _current
    app.setStyle("Fusion")
    effective = resolve_effective_theme(mode)
    p = DARK if effective == "dark" else LIGHT
    _current = p

    font = QFont(app.font())
    font.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "Arial"])
    app.setFont(font)

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(p.background))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p.text))
    pal.setColor(QPalette.ColorRole.Base, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p.surface_hover))
    pal.setColor(QPalette.ColorRole.Text, QColor(p.text))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(p.muted))
    pal.setColor(QPalette.ColorRole.Button, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p.text))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p.accent))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p.surface))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p.text))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(p.disabled_text))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(p.disabled_text))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(p.disabled_text))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(p))
    return effective


def repolish(widget) -> None:  # type: ignore[no-untyped-def]
    """Re-apply the style sheet after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
