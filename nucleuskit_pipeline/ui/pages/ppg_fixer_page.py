"""PPG Fixer: artifact rejection via squared z-score outlier detection."""

from __future__ import annotations

import queue
import sys
import traceback

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
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

from nucleuskit_pipeline.shimmer.processor.ppg_fixer import fix_ppg_session
from nucleuskit_pipeline.ui.offline_job import QueueTextWriter, ppg_fixer_preflight


class _PpgFixerWorker(QObject):
    finished_ok  = Signal()
    finished_err = Signal(str)

    def __init__(self, folder: str, log_queue: queue.SimpleQueue[str]) -> None:
        super().__init__()
        self._folder    = folder
        self._log_queue = log_queue

    @Slot()
    def run(self) -> None:
        err: str | None = None
        writer          = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            fix_ppg_session(self._folder)
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


class PpgFixerPage(QWidget):
    go_tools_menu      = Signal()
    processing_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread: QThread | None           = None
        self._worker: _PpgFixerWorker | None   = None

        # ── top bar ──────────────────────────────────────────────────
        self._back = QPushButton("Back")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_tools_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._back)

        # ── session folder selector ───────────────────────────────────
        self._session = QLineEdit()
        self._session.setPlaceholderText("Session folder path…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_session)

        sess_row = QHBoxLayout()
        sess_row.addWidget(self._session, 1)
        sess_row.addWidget(browse)

        session_box = QGroupBox(
            "Session  (must contain features/ppg/ppg_resampled.csv)"
        )
        sg = QVBoxLayout(session_box)
        sg.addLayout(sess_row)

        # ── actions ───────────────────────────────────────────────────
        self._run = QPushButton("Run PPG Fixer")
        self._run.clicked.connect(self._run_clicked)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)

        actions = QHBoxLayout()
        actions.addWidget(self._run)
        actions.addWidget(self._progress, 1)

        # ── log ───────────────────────────────────────────────────────
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        log_box = QGroupBox("Log")
        lg = QVBoxLayout(log_box)
        lg.addWidget(self._log)

        # ── assemble ─────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(session_box)
        layout.addLayout(actions)
        layout.addWidget(log_box, 1)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._drain_log_queue)
        self._poll_timer.start(120)

    # ── private helpers ───────────────────────────────────────────────

    def _browse_session(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select session folder")
        if path:
            self._session.setText(path)

    def _set_processing(self, running: bool) -> None:
        self._back.setEnabled(not running)
        self._run.setEnabled(not running)
        self._session.setEnabled(not running)
        self._progress.setVisible(running)
        self.processing_changed.emit(running)

    def _run_clicked(self) -> None:
        folder = self._session.text().strip()
        err    = ppg_fixer_preflight(folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return

        self._set_processing(True)
        self._insert_log(f"\n--- Starting PPG Fixer: {folder!r} ---\n")

        thread = QThread()
        worker = _PpgFixerWorker(folder, self._log_queue)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
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
        QMessageBox.information(
            self,
            "PPG Fixer",
            "Done.\n\n"
            "Artifacts removed, features/ppg/ppg_rejected.png saved, and "
            "results/HeartDynamics.csv recomputed.",
        )

    def _on_finished_err(self, msg: str) -> None:
        self._set_processing(False)
        QMessageBox.critical(self, "PPG Fixer", msg)

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
