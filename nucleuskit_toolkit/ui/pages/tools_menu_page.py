"""Auxiliary tools for processed sessions (Channel Fixer, …), shown as a grid of cards."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from nucleuskit_toolkit.ui.widgets import ChoiceCard

TOOLS = (
    ("open_channel_fixer", "Channel Fixer", "Repair a single bad RMS channel with the channel-fix model."),
    ("open_channel_gain", "Channel gain adjustment", "Adjust per-channel RMS gain and zero offset."),
    ("open_revert_original", "Revert to original", "Restore the RMS features from the frozen baseline."),
    ("open_ppg_fixer", "PPG Fixer", "Re-run PPG artifact rejection and heart dynamics."),
    ("open_eeg_regression", "EEG Regression Denoising", "Remove shared muscle noise from the EEG."),
    ("open_recording_merge", "Recording Merger", "Join up to 5 recordings into one session."),
)


class ToolsMenuPage(QWidget):
    open_channel_fixer = Signal()
    open_channel_gain = Signal()
    open_revert_original = Signal()
    open_ppg_fixer = Signal()
    open_eeg_regression = Signal()
    open_recording_merge = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)
        for i, (signal_name, title, description) in enumerate(TOOLS):
            card = ChoiceCard("wrench", title, description)
            card.set_indicator_visible(False)
            card.clicked.connect(getattr(self, signal_name).emit)
            grid.addWidget(card, i // 2, i % 2)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(grid)
        outer.addStretch(1)
