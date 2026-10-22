"""About: application and processing-core versions, emotion models, license and links."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

import nucleuskit_toolkit
from nucleuskit_toolkit.ui.icons import icon
from nucleuskit_toolkit.ui.theme import current_palette
from nucleuskit_toolkit.ui.widgets import Card, Divider, SectionHeader, label

LINKS = (
    ("Nucleus-Kit on GitHub", "https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline"),
    ("re-ak-core on GitHub", "https://github.com/fs-re-ak/re-ak-core"),
    ("Changelog", "https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/CHANGELOG.md"),
    ("License (GPL-3.0-or-later)", "https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/LICENSE"),
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


class AboutPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from reak_core.models import DEFAULT_MODEL, builtin_releases, resolve_key

        card = Card()
        card.setMaximumWidth(960)
        card.body.addWidget(
            SectionHeader(
                "Nucleus-Kit Processing Toolkit", "Copyright (C) 2026 RE-AK Technologies Inc.", icon="activity"
            )
        )
        card.body.addWidget(Divider())

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        default = resolve_key(DEFAULT_MODEL)
        models = ", ".join(
            f"{r.name} {r.version}" + (" (recommended)" if key == default else "")
            for key, r in builtin_releases().items()
        )
        rows = (
            ("Version", nucleuskit_toolkit.__version__),
            ("Processing core", f"re-ak-core {_package_version('re-ak-core')}"),
            ("Emotion models", models),
            ("Qt", f"PySide6 {_package_version('PySide6')}"),
            ("License", "GNU General Public License v3.0 or later"),
        )
        for i, (name, value) in enumerate(rows):
            grid.addWidget(label(name, "hint"), i, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(label(value, "strong", wrap=True), i, 1)
        grid.setColumnStretch(1, 1)
        card.body.addLayout(grid)
        card.body.addWidget(Divider())

        links = QHBoxLayout()
        links.setSpacing(8)
        self._link_buttons: list[QPushButton] = []
        for text, url in LINKS:
            btn = QPushButton("  " + text)
            btn.setProperty("variant", "secondary")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, u=url: QDesktopServices.openUrl(QUrl(u)))
            links.addWidget(btn)
            self._link_buttons.append(btn)
        links.addStretch(1)
        card.body.addLayout(links)
        self.refresh_theme()

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(card)
        col.addStretch(1)

    def refresh_theme(self) -> None:
        for btn in self._link_buttons:
            btn.setIcon(icon("external", current_palette().text, 16))
