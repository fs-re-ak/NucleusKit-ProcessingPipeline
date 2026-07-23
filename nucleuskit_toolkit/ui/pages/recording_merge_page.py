"""Recording Merger: join up to 5 recordings with null-padded CSV gaps and black-screen video gaps."""

from __future__ import annotations

import queue
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_toolkit.tools.recording_merger import merge_recordings
from nucleuskit_toolkit.ui.offline_job import QueueTextWriter

_MAX_RECORDINGS = 5


# ── worker ─────────────────────────────────────────────────────────────────────

class _MergeWorker(QObject):
    finished_ok  = Signal()
    finished_err = Signal(str)

    def __init__(
        self,
        folders: list[str],
        offsets: list[float],
        output_dir: str,
        log_queue: queue.SimpleQueue[str],
        skip_video: bool = False,
    ) -> None:
        super().__init__()
        self._folders    = folders
        self._offsets    = offsets
        self._output_dir = output_dir
        self._log_queue  = log_queue
        self._skip_video = skip_video

    @Slot()
    def run(self) -> None:
        err: str | None = None
        writer = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            merge_recordings(
                self._folders, self._offsets, self._output_dir,
                self._log_queue, skip_video=self._skip_video,
            )
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


# ── recording row widget ────────────────────────────────────────────────────────

class _RecordingRow(QWidget):
    """One row: index label | path field | Browse | [Offset label | spin] | [Remove]."""

    path_changed     = Signal()
    remove_requested = Signal(object)   # emits self

    def __init__(self, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index

        self._label = QLabel(f"Recording {index + 1}:")
        self._label.setFixedWidth(100)

        self._path = QLineEdit()
        self._path.setPlaceholderText("Recording folder path…")
        self._path.textChanged.connect(lambda _: self.path_changed.emit())

        self._browse = QPushButton("Browse…")
        self._browse.clicked.connect(self._browse_clicked)

        self._offset_label = QLabel("Offset:")
        self._offset = QDoubleSpinBox()
        self._offset.setRange(0.0, 86_400.0)   # up to 24 h
        self._offset.setSingleStep(1.0)
        self._offset.setSuffix(" s")
        self._offset.setValue(0.0)
        self._offset.setFixedWidth(110)
        self._offset.setToolTip("Gap in seconds from the end of the previous recording to the start of this one.")

        self._remove = QPushButton("Remove")
        self._remove.setProperty("secondary", True)
        self._remove.clicked.connect(lambda: self.remove_requested.emit(self))

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.addWidget(self._label)
        row.addWidget(self._path, 1)
        row.addWidget(self._browse)
        row.addWidget(self._offset_label)
        row.addWidget(self._offset)
        row.addWidget(self._remove)

        self._refresh_visibility()

    # ── public ────────────────────────────────────────────────────────────────

    def folder(self) -> str:
        return self._path.text().strip()

    def offset(self) -> float:
        return self._offset.value()

    def set_index(self, new_index: int) -> None:
        self._index = new_index
        self._label.setText(f"Recording {new_index + 1}:")
        self._refresh_visibility()

    def set_controls_enabled(self, enabled: bool) -> None:
        self._path.setEnabled(enabled)
        self._browse.setEnabled(enabled)
        self._offset.setEnabled(enabled)
        self._remove.setEnabled(enabled)

    # ── private ───────────────────────────────────────────────────────────────

    def _refresh_visibility(self) -> None:
        is_first = self._index == 0
        self._offset_label.setVisible(not is_first)
        self._offset.setVisible(not is_first)
        self._remove.setVisible(not is_first)

    def _browse_clicked(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select recording folder")
        if path:
            self._path.setText(path)


# ── main page ──────────────────────────────────────────────────────────────────

class RecordingMergePage(QWidget):
    go_tools_menu      = Signal()
    processing_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread: QThread | None            = None
        self._worker: _MergeWorker | None       = None
        self._rows: list[_RecordingRow]         = []

        # ── top bar ───────────────────────────────────────────────────────────
        self._back = QPushButton("Back")
        self._back.setProperty("secondary", True)
        self._back.clicked.connect(self.go_tools_menu.emit)

        top = QHBoxLayout()
        top.addStretch(1)
        top.addWidget(self._back)

        # ── recordings group ──────────────────────────────────────────────────
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(4)

        self._add_btn = QPushButton("+ Add recording")
        self._add_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._add_btn.clicked.connect(self._add_row)

        rows_inner = QVBoxLayout()
        rows_inner.addLayout(self._rows_layout)
        rows_inner.addWidget(self._add_btn)

        recordings_box = QGroupBox("Recordings  (2–5, in order)")
        recordings_box.setLayout(rows_inner)

        # ── output folder ─────────────────────────────────────────────────────
        self._output = QLineEdit()
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("Set Recording 1 path to auto-generate…")

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder (auto):"))
        out_row.addWidget(self._output, 1)

        # ── actions ───────────────────────────────────────────────────────────
        self._skip_video = QCheckBox("Skip video merge")
        self._skip_video.setToolTip(
            "When checked, only result CSVs are merged; video files are ignored."
        )

        self._run = QPushButton("Start Merge")
        self._run.clicked.connect(self._run_clicked)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)

        actions = QHBoxLayout()
        actions.addWidget(self._skip_video)
        actions.addStretch(1)
        actions.addWidget(self._run)
        actions.addWidget(self._progress, 1)

        # ── log ───────────────────────────────────────────────────────────────
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)

        log_box = QGroupBox("Log")
        lg = QVBoxLayout(log_box)
        lg.addWidget(self._log)

        # ── assemble ──────────────────────────────────────────────────────────
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(recordings_box)
        layout.addLayout(out_row)
        layout.addLayout(actions)
        layout.addWidget(log_box, 1)

        # seed with two rows so the tool is immediately usable
        self._add_row()
        self._add_row()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._drain_log_queue)
        self._poll_timer.start(120)

    # ── row management ────────────────────────────────────────────────────────

    def _add_row(self) -> None:
        if len(self._rows) >= _MAX_RECORDINGS:
            return
        row = _RecordingRow(len(self._rows), self)
        row.path_changed.connect(self._update_output_folder)
        row.remove_requested.connect(self._remove_row)
        self._rows_layout.addWidget(row)
        self._rows.append(row)
        self._add_btn.setEnabled(len(self._rows) < _MAX_RECORDINGS)

    def _remove_row(self, row: _RecordingRow) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        for i, r in enumerate(self._rows):
            r.set_index(i)
        self._add_btn.setEnabled(len(self._rows) < _MAX_RECORDINGS)
        self._update_output_folder()

    def _update_output_folder(self) -> None:
        if not self._rows:
            self._output.setText("")
            return
        first_path = self._rows[0].folder()
        if not first_path:
            self._output.setText("")
            return
        names = [Path(r.folder()).name for r in self._rows if r.folder()]
        merged_name = "merged_" + "_".join(names) if names else "merged_recording"
        self._output.setText(str(Path(first_path).parent / merged_name))

    # ── validation ────────────────────────────────────────────────────────────

    def _validate(self) -> str | None:
        if len(self._rows) < 2:
            return "Add at least 2 recordings to merge."

        for i, row in enumerate(self._rows):
            f = row.folder()
            if not f:
                return f"Recording {i + 1} has no folder selected."
            p = Path(f)
            if not p.is_dir():
                return f"Recording {i + 1}: folder does not exist:\n{f}"
            if not (p / "results").is_dir() and not (p / "rawData").is_dir():
                return (
                    f"Recording {i + 1}: folder does not look like a recording\n"
                    f"(no 'results/' or 'rawData/' subfolder found):\n{f}"
                )

        output = self._output.text().strip()
        if not output:
            return "Cannot determine output folder. Set Recording 1 path first."
        if Path(output).exists():
            return (
                f"Output folder already exists:\n{output}\n\n"
                "Remove or rename it before merging."
            )

        return None

    # ── run ───────────────────────────────────────────────────────────────────

    def _run_clicked(self) -> None:
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Recording Merger", err)
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A merge is already in progress.")
            return

        folders    = [row.folder() for row in self._rows]
        offsets    = [row.offset() for row in self._rows[1:]]
        output_dir = self._output.text().strip()
        skip_video = self._skip_video.isChecked()

        self._set_processing(True)
        self._insert_log(
            f"\n--- Starting merge of {len(folders)} recordings ---\n"
            + "".join(f"  [{i + 1}] {f}\n" for i, f in enumerate(folders))
            + (f"  (video merge skipped)\n" if skip_video else "")
            + "\n"
        )

        thread = QThread()
        worker = _MergeWorker(folders, offsets, output_dir, self._log_queue, skip_video)
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

    def _set_processing(self, running: bool) -> None:
        self._back.setEnabled(not running)
        self._run.setEnabled(not running)
        self._add_btn.setEnabled(not running)
        self._skip_video.setEnabled(not running)
        self._progress.setVisible(running)
        for row in self._rows:
            row.set_controls_enabled(not running)
        self.processing_changed.emit(running)

    def _clear_thread_ref(self) -> None:
        self._thread = None
        self._worker = None

    def _on_finished_ok(self) -> None:
        self._set_processing(False)
        QMessageBox.information(
            self,
            "Recording Merger",
            f"Done.\n\nMerged recording saved to:\n{self._output.text()}",
        )

    def _on_finished_err(self, msg: str) -> None:
        self._set_processing(False)
        QMessageBox.critical(self, "Recording Merger", msg)

    # ── log helpers ───────────────────────────────────────────────────────────

    def _insert_log(self, text: str) -> None:
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.insertPlainText(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _drain_log_queue(self) -> None:
        try:
            while True:
                self._insert_log(self._log_queue.get_nowait())
        except queue.Empty:
            pass
