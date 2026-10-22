"""Building blocks of the app's visual language: cards, section headers, dividers, pills, choice cards.

Every section header follows one pattern: an icon or a step number in a circle, a title and a one-line description.
Colours come from the theme (``nucleuskit_toolkit.ui.theme``) through dynamic properties; icons are re-rendered when
the theme changes (``refresh_icons``).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_toolkit.ui.icons import pixmap
from nucleuskit_toolkit.ui.theme import current_palette, repolish


def label(text: str = "", role: str | None = None, *, wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setProperty("role", role)
    lbl.setWordWrap(wrap)
    return lbl


def set_role(widget: QLabel, role: str) -> None:
    if widget.property("role") != role:
        widget.setProperty("role", role)
        repolish(widget)


class IconLabel(QLabel):
    """A QLabel showing an icon in a theme colour (``muted``, ``accent``, ``text`` or a hex value)."""

    def __init__(self, name: str, size: int = 18, color: str = "muted", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self._size = size
        self._color = color
        self.setFixedSize(size, size)
        self.setStyleSheet("background: transparent;")
        self.refresh()

    @property
    def icon_name(self) -> str:
        return self._name

    def set_icon(self, name: str, color: str | None = None) -> None:
        self._name = name
        if color:
            self._color = color
        self.refresh()

    def refresh(self) -> None:
        p = current_palette()
        color = getattr(p, self._color, self._color)
        self.setPixmap(pixmap(self._name, color, self._size))


def refresh_icons(root: QWidget) -> None:
    """Re-render every IconLabel under ``root`` (after a theme change)."""
    for child in root.findChildren(IconLabel):
        child.refresh()


class Divider(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("divider", True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedHeight(1)


class Pill(QLabel):
    """Status pill: ``on`` (accent), ``off`` (neutral surface) or ``value`` (neutral, free text)."""

    def __init__(self, text: str = "", state: str = "value", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("pill", state)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_state(self, text: str, state: str) -> None:
        self.setText(text)
        if self.property("pill") != state:
            self.setProperty("pill", state)
            repolish(self)


class SectionHeader(QWidget):
    """Icon (or step number) in a circle, title, one-line description."""

    def __init__(
        self,
        title: str,
        description: str = "",
        *,
        icon: str | None = None,
        step: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)
        if step is not None:
            badge = QLabel(str(step))
            badge.setObjectName("StepNumber")
            badge.setFixedSize(32, 32)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        elif icon is not None:
            holder = QLabel()
            holder.setObjectName("HeaderIcon")
            holder.setFixedSize(36, 36)
            inner = QHBoxLayout(holder)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.addWidget(IconLabel(icon, 18, "accent"), 0, Qt.AlignmentFlag.AlignCenter)
            row.addWidget(holder, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = label(title, "cardTitle")
        text.addWidget(self.title)
        self.description = label(description, "hint", wrap=True)
        self.description.setVisible(bool(description))
        text.addWidget(self.description)
        row.addLayout(text, 1)


class Card(QFrame):
    """Rounded surface with consistent padding; ``body`` is its vertical layout."""

    def __init__(self, parent: QWidget | None = None, *, padding: int = 20, spacing: int = 16) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)


class ChoiceCard(QFrame):
    """Selectable card (radio behaviour handled by the owner): indicator, icon, title, description."""

    clicked = Signal()

    def __init__(self, icon: str, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("choice", True)
        self.setProperty("selected", False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(18, 14, 18, 14)
        row.setSpacing(14)
        self._dot = IconLabel("radio-off", 22, "muted")
        self._icon = IconLabel(icon, 22, "muted")
        row.addWidget(self._dot)
        row.addWidget(self._icon)
        text = QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(label(title, "strong"))
        text.addWidget(label(description, "hint"))
        row.addLayout(text, 1)
        self._update_indicator()

    def set_indicator_visible(self, visible: bool) -> None:
        self._dot.setVisible(visible)

    def is_selected(self) -> bool:
        return bool(self.property("selected"))

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        repolish(self)
        self._update_indicator()

    def _update_indicator(self) -> None:
        on = self.is_selected()
        self._dot.set_icon("radio-on" if on else "radio-off", "accent" if on else "muted")
        self._icon.set_icon(self._icon.icon_name, "accent" if on else "muted")

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
