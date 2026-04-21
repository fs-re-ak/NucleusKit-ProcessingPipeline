"""Channel Fixer: repair one RMS channel in a processed session."""

from __future__ import annotations

import queue
import sys
import traceback

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.hermes.processorDevelopment.channel_fixer_release.channel_fixer import (
    CANONICAL_CHANNEL_NAMES,
    fix_session,
)
from nucleuskit_pipeline.ui.offline_job import QueueTextWriter, channel_fixer_preflight


class ChannelFixerWorker(QObject):
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        folder: str,
        channel_idx: int,
        log_queue: queue.SimpleQueue[str],
    ) -> None:
        super().__init__()
        self._folder = folder
        self._channel_idx = channel_idx
        self._log_queue = log_queue

    @Slot()
    def run_fix(self) -> None:
        err: str | None = None
        writer = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            out = fix_session(self._folder, self._channel_idx)
            print(f"Report written: {out}")
        except BaseException:
            err = traceback.format_exc()
            self._log_queue.put(err)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        if err:
            self.finished_err.emit("Run finished with errors (see log).")
        else:
            self.finished_ok.emit()


class ChannelFixerPage(QWidget):
    go_tools_menu = Signal()
    processing_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread: QThread | None = None
        self._worker: ChannelFixerWorker | None = None

        self._back = QPushButton("Back")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_tools_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._back)

        self._session = QLineEdit()
        self._session.setPlaceholderText("Session folder path…")
        browse_s = QPushButton("Browse…")
        browse_s.clicked.connect(self._browse_session)

        sess_row = QHBoxLayout()
        sess_row.addWidget(self._session, 1)
        sess_row.addWidget(browse_s)

        self._channel = QComboBox()
        for name in CANONICAL_CHANNEL_NAMES:
            self._channel.addItem(name)

        ch_row = QHBoxLayout()
        ch_row.addWidget(self._channel, 1)

        session_box = QGroupBox("Session (must contain features/emotions/rmsSignals.csv)")
        sg = QVBoxLayout(session_box)
        sg.addLayout(sess_row)

        channel_box = QGroupBox("Channel to fix (canonical name)")
        cg = QVBoxLayout(channel_box)
        cg.addLayout(ch_row)

        self._run = QPushButton("Run channel fixer")
        self._run.clicked.connect(self._run_clicked)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)

        actions = QHBoxLayout()
        actions.addWidget(self._run)
        actions.addWidget(self._progress, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        log_box = QGroupBox("Log")
        lg = QVBoxLayout(log_box)
        lg.addWidget(self._log)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(session_box)
        layout.addWidget(channel_box)
        layout.addLayout(actions)
        layout.addWidget(log_box, 1)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._drain_log_queue)
        self._poll_timer.start(120)

    def _browse_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select session folder")
        if path:
            self._session.setText(path)

    def _set_processing(self, running: bool) -> None:
        self._back.setEnabled(not running)
        self._run.setEnabled(not running)
        self._session.setEnabled(not running)
        self._channel.setEnabled(not running)
        self._progress.setVisible(running)
        self.processing_changed.emit(running)

    def _run_clicked(self) -> None:
        folder = self._session.text().strip()
        err = channel_fixer_preflight(folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return

        channel_idx = self._channel.currentIndex()
        self._set_processing(True)
        self._insert_log(f"\n--- Starting channel fixer: {folder!r} channel index {channel_idx} ---\n")

        thread = QThread()
        worker = ChannelFixerWorker(folder, channel_idx, self._log_queue)
        worker.moveToThread(thread)
        thread.started.connect(worker.run_fix)
        worker.finished_ok.connect(self._on_finished_ok)
        worker.finished_err.connect(self._on_finished_err)
        worker.finished_ok.connect(thread.quit)
        worker.finished_err.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_thread_ref)

        self._thread = thread
        self._worker = worker
        thread.start()

    def _clear_thread_ref(self) -> None:
        self._thread = None
        self._worker = None

    def _on_finished_ok(self) -> None:
        self._set_processing(False)
        QMessageBox.information(self, "Channel fixer", "Run finished.")

    def _on_finished_err(self, msg: str) -> None:
        self._set_processing(False)
        QMessageBox.critical(self, "Channel fixer", msg)

    def _insert_log(self, text: str) -> None:
        self._log.moveCursor(QTextCursor.End)
        self._log.insertPlainText(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _drain_log_queue(self) -> None:
        try:
            while True:
                chunk = self._log_queue.get_nowait()
                self._insert_log(chunk)
        except queue.Empty:
            pass
