"""Offline session processing (Qt port of the former tkinter view)."""

from __future__ import annotations

import os
import queue
import sys
import traceback

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.ui.offline_job import QueueTextWriter, dataset_preflight, session_preflight


class PipelineWorker(QObject):
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        folder: str,
        cfg: str | None,
        log_queue: queue.SimpleQueue[str],
        skip_video_rotation: bool = False,
        apply_nlms: bool = True,
    ) -> None:
        super().__init__()
        self._folder = folder
        self._cfg = cfg
        self._log_queue = log_queue
        self._skip_video_rotation = skip_video_rotation
        self._apply_nlms = apply_nlms

    @Slot()
    def run_pipeline(self) -> None:
        err: str | None = None
        writer = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            from nucleuskit_pipeline.logging_utils import configure_logging
            from nucleuskit_pipeline.pipeline import NucleusKitProcessingPipeline
            from nucleuskit_pipeline.session_job import session_job_from_folder

            configure_logging()
            job = session_job_from_folder(self._folder, pov_config_json=self._cfg)
            pipe = NucleusKitProcessingPipeline(
                None,
                skip_video_rotation=self._skip_video_rotation,
                apply_nlms=self._apply_nlms,
            )
            pipe.processSession(job)
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


class BatchPipelineWorker(QObject):
    """Runs the pipeline sequentially on a list of session folders."""

    finished_ok   = Signal()
    finished_err  = Signal(str)
    progress      = Signal(int)   # emits 1 … N after each session completes
    session_start = Signal(str)   # emits "N/M  <folder_name>" before each session

    def __init__(
        self,
        folders: list[str],
        log_queue: queue.SimpleQueue[str],
        skip_video_rotation: bool = False,
        apply_nlms: bool = True,
    ) -> None:
        super().__init__()
        self._folders = folders
        self._log_queue = log_queue
        self._skip_video_rotation = skip_video_rotation
        self._apply_nlms = apply_nlms

    @Slot()
    def run_pipeline(self) -> None:
        has_error = False
        writer = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            from nucleuskit_pipeline.logging_utils import configure_logging
            from nucleuskit_pipeline.pipeline import NucleusKitProcessingPipeline
            from nucleuskit_pipeline.session_job import session_job_from_folder

            configure_logging()
            total = len(self._folders)
            for idx, folder in enumerate(self._folders, start=1):
                name = os.path.basename(folder.rstrip("/\\"))
                self.session_start.emit(f"{idx}/{total}  {name}")
                try:
                    job = session_job_from_folder(folder)
                    pipe = NucleusKitProcessingPipeline(
                        None,
                        skip_video_rotation=self._skip_video_rotation,
                        apply_nlms=self._apply_nlms,
                    )
                    pipe.processSession(job)
                except BaseException as e:
                    import traceback
                    msg = f"[BatchWorker] ERROR in session {name!r}: {e}\n{traceback.format_exc()}"
                    self._log_queue.put(msg)
                    has_error = True
                self.progress.emit(idx)
        except BaseException:
            import traceback
            err = traceback.format_exc()
            self._log_queue.put(err)
            has_error = True
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        if has_error:
            self.finished_err.emit("Batch run finished with one or more errors (see log).")
        else:
            self.finished_ok.emit()


class OfflinePage(QWidget):
    go_main_menu = Signal()
    processing_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

        self._back = QPushButton("Main menu")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_main_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._back)

        # ── Mode radio buttons ────────────────────────────────────────────────
        self._radio_session = QRadioButton("Single session")
        self._radio_session.setChecked(True)
        self._radio_dataset = QRadioButton("Dataset (all sessions in folder)")
        self._radio_session.toggled.connect(self._on_mode_changed)

        radio_row = QHBoxLayout()
        radio_row.addWidget(self._radio_session)
        radio_row.addWidget(self._radio_dataset)
        radio_row.addStretch(1)

        # ── Shared path input ─────────────────────────────────────────────────
        self._session = QLineEdit()
        self._session.setPlaceholderText("Session folder path…")
        self._session.textChanged.connect(self._on_path_changed)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._browse_clicked)

        path_row = QHBoxLayout()
        path_row.addWidget(self._session, 1)
        path_row.addWidget(self._browse_btn)

        self._dataset_hint = QLabel("")
        self._dataset_hint.setVisible(False)

        input_box = QGroupBox("Input")
        ig = QVBoxLayout(input_box)
        ig.addLayout(radio_row)
        ig.addLayout(path_row)
        ig.addWidget(self._dataset_hint)

        # ── Processing options ────────────────────────────────────────────────
        self._skip_video_rotation = QCheckBox("Skip video rotation")
        self._skip_video_rotation.setToolTip(
            "When checked, the 180° rotation of rawData/video.mp4 is skipped entirely,\n"
            "even if it has not been applied yet."
        )

        self._apply_nlms = QCheckBox("Apply adaptive NLMS decorrelation")
        self._apply_nlms.setChecked(True)
        self._apply_nlms.setToolTip(
            "Apply the causal NLMS adaptive regression filter to raw EEG/EMG before processing.\n"
            "For each channel, the other channels act as predictors to remove common-mode\n"
            "cross-channel contamination. Uncheck to use the unfiltered raw signal."
        )

        self._run = QPushButton("Run pipeline")
        self._run.clicked.connect(self._run_clicked)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)

        actions = QHBoxLayout()
        actions.addWidget(self._run)
        actions.addWidget(self._skip_video_rotation)
        actions.addWidget(self._apply_nlms)
        actions.addStretch(1)
        actions.addWidget(self._progress, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        log_box = QGroupBox("Log")
        lg = QVBoxLayout(log_box)
        lg.addWidget(self._log)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(input_box)
        layout.addLayout(actions)
        layout.addWidget(log_box, 1)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._drain_log_queue)
        self._poll_timer.start(120)

    # ── Mode / path helpers ───────────────────────────────────────────────────

    def _is_dataset_mode(self) -> bool:
        return self._radio_dataset.isChecked()

    def _on_mode_changed(self) -> None:
        """Update placeholder text and refresh dataset hint when mode changes."""
        if self._is_dataset_mode():
            self._session.setPlaceholderText("Dataset root folder path…")
        else:
            self._session.setPlaceholderText("Session folder path…")
        self._on_path_changed(self._session.text())

    def _on_path_changed(self, text: str) -> None:
        """In dataset mode, scan for sessions and update the hint label."""
        if not self._is_dataset_mode():
            self._dataset_hint.setVisible(False)
            return
        folder = text.strip()
        if not folder:
            self._dataset_hint.setText("")
            self._dataset_hint.setVisible(False)
            return
        sessions, err = dataset_preflight(folder)
        if err:
            self._dataset_hint.setText(err)
        else:
            self._dataset_hint.setText(f"Found {len(sessions)} session(s)")
        self._dataset_hint.setVisible(True)

    def _browse_clicked(self) -> None:
        title = "Select dataset root folder" if self._is_dataset_mode() else "Select session folder"
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            self._session.setText(path)

    # ── Run logic ─────────────────────────────────────────────────────────────

    def _set_processing(self, running: bool) -> None:
        self._back.setEnabled(not running)
        self._run.setEnabled(not running)
        self._session.setEnabled(not running)
        self._browse_btn.setEnabled(not running)
        self._radio_session.setEnabled(not running)
        self._radio_dataset.setEnabled(not running)
        self._skip_video_rotation.setEnabled(not running)
        self._apply_nlms.setEnabled(not running)
        self._progress.setVisible(running)
        self.processing_changed.emit(running)

    def _run_clicked(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return

        skip_rot = self._skip_video_rotation.isChecked()
        apply_nlms = self._apply_nlms.isChecked()

        if self._is_dataset_mode():
            self._run_dataset(skip_rot, apply_nlms)
        else:
            self._run_single(skip_rot, apply_nlms)

    def _run_single(self, skip_rot: bool, apply_nlms: bool) -> None:
        folder = self._session.text().strip()
        err = session_preflight(folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return

        self._set_processing(True)
        self._progress.setRange(0, 0)   # indeterminate spinner
        self._insert_log(f"\n--- Starting run: {folder!r} ---\n")
        if skip_rot:
            self._insert_log("  (video rotation skipped by user request)\n")
        if not apply_nlms:
            self._insert_log("  (NLMS adaptive decorrelation disabled by user request)\n")

        thread = QThread()
        worker = PipelineWorker(
            folder, None, self._log_queue,
            skip_video_rotation=skip_rot,
            apply_nlms=apply_nlms,
        )
        self._start_worker(thread, worker)

    def _run_dataset(self, skip_rot: bool, apply_nlms: bool) -> None:
        folder = self._session.text().strip()
        sessions, err = dataset_preflight(folder)
        if err:
            QMessageBox.warning(self, "Dataset", err)
            return

        n = len(sessions)
        if n > 10:
            answer = QMessageBox.question(
                self,
                "Process entire dataset?",
                f"This will run the pipeline on {n} sessions sequentially.\n"
                "This may take a long time. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self._set_processing(True)
        self._progress.setRange(0, n)   # determinate bar
        self._progress.setValue(0)
        self._insert_log(f"\n--- Starting dataset run: {folder!r} ({n} sessions) ---\n")
        if skip_rot:
            self._insert_log("  (video rotation skipped by user request)\n")
        if not apply_nlms:
            self._insert_log("  (NLMS adaptive decorrelation disabled by user request)\n")

        thread = QThread()
        worker = BatchPipelineWorker(
            sessions, self._log_queue,
            skip_video_rotation=skip_rot,
            apply_nlms=apply_nlms,
        )
        worker.session_start.connect(
            lambda label: self._insert_log(f"\n--- Session {label} ---\n")
        )
        worker.progress.connect(self._progress.setValue)
        self._start_worker(thread, worker)

    def _start_worker(self, thread: QThread, worker: QObject) -> None:
        worker.moveToThread(thread)
        thread.started.connect(worker.run_pipeline)
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
        QMessageBox.information(self, "Pipeline", "Run finished.")

    def _on_finished_err(self, msg: str) -> None:
        self._set_processing(False)
        QMessageBox.critical(self, "Pipeline", msg)

    # ── Log helpers ───────────────────────────────────────────────────────────

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
