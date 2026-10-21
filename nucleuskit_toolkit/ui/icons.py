"""Line icons (24 x 24, stroke-based) rendered in any colour, plus a small spinner widget.

The icons are drawn for Nucleus-Kit in the style of outline icon sets: 2 px strokes, round caps and joins.
``icon(name, color)`` returns a QIcon; ``icon_file(name, color)`` writes an SVG to a cache folder for Qt
style sheets (``image: url(...)``), which cannot load in-memory images.
"""

from __future__ import annotations

import os
import tempfile

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QSizePolicy, QWidget

_PATHS: dict[str, str] = {
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "play": '<polygon points="7 4 20 12 7 20 7 4"/>',
    "square": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1'
        'M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>'
        '<circle cx="12" cy="12" r="6.5"/>'
    ),
    "book": (
        '<path d="M2 4.5h6a4 4 0 0 1 4 4V21a3 3 0 0 0-3-3H2z"/>'
        '<path d="M22 4.5h-6a4 4 0 0 0-4 4V21a3 3 0 0 1 3-3h7z"/>'
    ),
    "info": '<circle cx="12" cy="12" r="9.5"/><path d="M12 16.5v-5M12 7.5h.01"/>',
    "help": (
        '<circle cx="12" cy="12" r="9.5"/>'
        '<path d="M9.2 9a3 3 0 0 1 5.8 1c0 2-3 2.5-3 4.5"/><path d="M12 17.5h.01"/>'
    ),
    "folder": '<path d="M3 6.5a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "database": (
        '<ellipse cx="12" cy="5.5" rx="8" ry="3"/>'
        '<path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>'
    ),
    "file-text": (
        '<path d="M14 2.5H6a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8.5z"/>'
        '<path d="M14 2.5v6h6M8 13h8M8 17h8M8 9h2"/>'
    ),
    "trash": (
        '<path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>'
        '<path d="M5.5 6l1 14a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2l1-14M10 11v6M14 11v6"/>'
    ),
    "radio": (
        '<circle cx="12" cy="12" r="2"/><path d="M16.2 7.8a6 6 0 0 1 0 8.4M7.8 16.2a6 6 0 0 1 0-8.4"/>'
        '<path d="M19.1 4.9a10 10 0 0 1 0 14.2M4.9 19.1a10 10 0 0 1 0-14.2"/>'
    ),
    "film": (
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<path d="M7 3v18M17 3v18M3 7.5h4M3 12h18M3 16.5h4M17 7.5h4M17 16.5h4"/>'
    ),
    "wrench": (
        '<path d="M14.7 6.3a4 4 0 0 0 5.3 5.3l-8.8 8.8a2.1 2.1 0 0 1-3-3l8.8-8.8a4 4 0 0 1-5.3-5.3'
        'l2.9 2.9 2.4-.4.4-2.4z"/>'
    ),
    "network": (
        '<rect x="9" y="2.5" width="6" height="5" rx="1"/><rect x="2.5" y="16.5" width="6" height="5" rx="1"/>'
        '<rect x="15.5" y="16.5" width="6" height="5" rx="1"/><path d="M12 7.5v4.5M5.5 16.5V12h13v4.5"/>'
    ),
    "chevron-down": '<polyline points="6 9 12 15 18 9"/>',
    "chevron-left": '<polyline points="15 18 9 12 15 6"/>',
    "chevron-up": '<polyline points="6 15 12 9 18 15"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "dot": '<circle cx="12" cy="12" r="5" fill="currentColor"/>',
    "radio-off": '<circle cx="12" cy="12" r="9"/>',
    "radio-on": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4.5" fill="currentColor"/>',
    "alert": '<circle cx="12" cy="12" r="9.5"/><path d="M12 7.5v5.5M12 16.5h.01"/>',
    "check-circle": '<circle cx="12" cy="12" r="9.5"/><polyline points="16.5 9 10.5 15 7.5 12"/>',
    "sliders": (
        '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3"/>'
        '<path d="M1.5 14h5M9.5 8h5M17.5 16h5"/>'
    ),
    "bluetooth": '<polyline points="7 7 17 17 12 22 12 2 17 7 7 17"/>',
    "external": (
        '<path d="M18 13.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5.5"/>'
        '<path d="M15 3h6v6M10 14L21 3"/>'
    ),
}

ICON_NAMES = tuple(_PATHS)


def svg(name: str, color: str, stroke_width: float = 2.0) -> str:
    """The SVG document of ``name`` drawn in ``color``."""
    body = _PATHS[name].replace("currentColor", color)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
        f"{body}</svg>"
    )


def pixmap(name: str, color: str, size: int = 18, ratio: float = 2.0) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg(name, color).encode("utf-8")))
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(ratio)
    return pm


def icon(name: str, color: str, size: int = 18) -> QIcon:
    return QIcon(pixmap(name, color, size))


def icon_file(name: str, color: str, stroke_width: float = 2.0) -> str:
    """Path of a cached SVG file of ``name`` in ``color``, with forward slashes for style sheets."""
    folder = os.path.join(tempfile.gettempdir(), "nucleuskit_icons")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}_{color.lstrip('#')}_{stroke_width:g}.svg")
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg(name, color, stroke_width))
    return path.replace("\\", "/")


class Spinner(QWidget):
    """An animated arc showing that work is in progress."""

    def __init__(self, size: int = 22, color: str = "#4F85EE", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._angle = 0
        self._color = QColor(color)
        self.setFixedSize(QSize(size, size))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def start(self) -> None:
        self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = max(2.0, self.width() / 9)
        rect = QRectF(width, width, self.width() - 2 * width, self.height() - 2 * width)
        track = QColor(self._color)
        track.setAlpha(50)
        painter.setPen(QPen(track, width))
        painter.drawEllipse(rect)
        painter.setPen(QPen(self._color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, -self._angle * 16, 100 * 16)
        painter.end()
