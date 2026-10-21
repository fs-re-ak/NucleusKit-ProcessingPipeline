"""Process Sessions: run the pipeline on a single session or on every session of a dataset folder.

Layout: numbered step cards on the left (1. input, 2. processing options), the run summary with the Run pipeline
button on the right, the processing log at the bottom. The selected folder is inspected in the background (read
only) and Run is enabled once it is a valid session or dataset. While running, Run becomes Cancel: the step in
progress finishes, then the remaining steps (and sessions) are skipped.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from reak_core.models import DEFAULT_MODEL, builtin_releases, resolve_key

from nucleuskit_toolkit.ui.icons import Spinner, icon
from nucleuskit_toolkit.ui.offline_job import QueueTextWriter, dataset_preflight, session_preflight
from nucleuskit_toolkit.ui.session_inspect import DatasetInfo, SessionInfo, inspect_input
from nucleuskit_toolkit.ui.theme import current_palette, repolish
from nucleuskit_toolkit.ui.widgets import (
    Card,
    ChoiceCard,
    Divider,
    IconLabel,
    Pill,
    SectionHeader,
    label,
    set_role,
)

MAX_CONTENT_WIDTH = 1280
INSPECT_DELAY_MS = 300

MODE_TEXT = {
    False: ("Session folder path", "Select the folder containing the session data.", "Enter session folder path…",
            "Select session folder"),
    True: ("Dataset folder path", "Select the folder containing the session folders.", "Enter dataset folder path…",
           "Select dataset root folder"),
}


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
        emotion_model: str | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._folder = folder
        self._cfg = cfg
        self._log_queue = log_queue
        self._skip_video_rotation = skip_video_rotation
        self._apply_nlms = apply_nlms
        self._emotion_model = emotion_model
        self._cancel = cancel

    @Slot()
    def run_pipeline(self) -> None:
        err: str | None = None
        writer = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            from nucleuskit_toolkit.logging_utils import configure_logging
            from nucleuskit_toolkit.pipeline import NucleusKitProcessingToolkit
            from nucleuskit_toolkit.session_job import session_job_from_folder

            configure_logging()
            job = session_job_from_folder(self._folder, pov_config_json=self._cfg)
            pipe = NucleusKitProcessingToolkit(
                None,
                skip_video_rotation=self._skip_video_rotation,
                apply_nlms=self._apply_nlms,
                emotion_model=self._emotion_model,
                cancel=self._cancel,
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
    """Runs the pipeline sequentially on a list of session folders; a failing session does not stop the batch."""

    finished_ok = Signal()
    finished_err = Signal(str)
    progress = Signal(int)  # emits 1 … N after each session completes
    session_start = Signal(str)  # emits "N/M  <folder_name>" before each session

    def __init__(
        self,
        folders: list[str],
        log_queue: queue.SimpleQueue[str],
        skip_video_rotation: bool = False,
        apply_nlms: bool = True,
        emotion_model: str | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self._folders = folders
        self._log_queue = log_queue
        self._skip_video_rotation = skip_video_rotation
        self._apply_nlms = apply_nlms
        self._emotion_model = emotion_model
        self._cancel = cancel

    @Slot()
    def run_pipeline(self) -> None:
        has_error = False
        writer = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            from nucleuskit_toolkit.logging_utils import configure_logging
            from nucleuskit_toolkit.pipeline import NucleusKitProcessingToolkit
            from nucleuskit_toolkit.session_job import session_job_from_folder

            configure_logging()
            total = len(self._folders)
            for idx, folder in enumerate(self._folders, start=1):
                if self._cancel is not None and self._cancel.is_set():
                    self._log_queue.put(f"\n--- Cancelled: {total - idx + 1} session(s) not processed ---\n")
                    break
                name = os.path.basename(folder.rstrip("/\\"))
                self.session_start.emit(f"{idx}/{total}  {name}")
                try:
                    job = session_job_from_folder(folder)
                    pipe = NucleusKitProcessingToolkit(
                        None,
                        skip_video_rotation=self._skip_video_rotation,
                        apply_nlms=self._apply_nlms,
                        emotion_model=self._emotion_model,
                        cancel=self._cancel,
                    )
                    pipe.processSession(job)
                except BaseException as e:
                    msg = f"[BatchWorker] ERROR in session {name!r}: {e}\n{traceback.format_exc()}"
                    self._log_queue.put(msg)
                    has_error = True
                self.progress.emit(idx)
        except BaseException:
            self._log_queue.put(traceback.format_exc())
            has_error = True
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

        if has_error:
            self.finished_err.emit("Batch run finished with one or more errors (see log).")
        else:
            self.finished_ok.emit()


class _InspectWorker(QObject):
    """Inspects the selected folder off the GUI thread (reads only)."""

    done = Signal(int, object, object)

    @Slot(int, str, bool)
    def inspect(self, request: int, path: str, dataset: bool) -> None:
        try:
            result, error = inspect_input(path, dataset)
        except Exception as exc:  # noqa: BLE001 — shown inline, never fatal
            result, error = None, f"Cannot read this folder: {exc}"
        self.done.emit(request, result, error)


class OptionRow(QWidget):
    """Check box with a title and a one-line description."""

    def __init__(self, title: str, description: str, checked: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(14)
        self.check = QCheckBox()
        self.check.setChecked(checked)
        self.check.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.check, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title = label(title, "strong")
        self.description = label(description, "hint", wrap=True)
        text.addWidget(self.title)
        text.addWidget(self.description)
        row.addLayout(text, 1)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.check.toggle()
        super().mousePressEvent(event)


class RunButton(QPushButton):
    """Large primary action: icon, title and subtitle; switches between Run and Cancel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        col = QVBoxLayout(self)
        col.setContentsMargins(16, 12, 16, 12)
        col.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addStretch(1)
        self._icon = IconLabel("play", 20, "#FFFFFF")
        self._title = label("Run pipeline")
        self._title.setStyleSheet("background: transparent; font-size: 17px; font-weight: 700;")
        top.addWidget(self._icon)
        top.addWidget(self._title)
        top.addStretch(1)
        self._subtitle = label("Start processing with the selected settings")
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._subtitle.setStyleSheet("background: transparent; font-size: 13px;")
        col.addLayout(top)
        col.addWidget(self._subtitle)
        for child in (self._icon, self._title, self._subtitle):
            child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.set_mode("run")

    def set_mode(self, mode: str) -> None:
        """``run``, ``cancel`` or ``cancelling``."""
        p = current_palette()
        if mode == "run":
            variant, name, title = "run", "play", "Run pipeline"
            subtitle = "Start processing with the selected settings"
        elif mode == "cancel":
            variant, name, title, subtitle = "cancel", "square", "Cancel", "Stop after the current step"
        else:
            variant, name, title, subtitle = "cancel", "square", "Cancelling…", "Finishing the current step"
        self.setProperty("variant", variant)
        repolish(self)
        self._mode = mode
        self._title.setText(title)
        self._subtitle.setText(subtitle)
        if mode != "run":  # in run mode, the page enables the button once the input is valid
            self.setEnabled(mode == "cancel")
        self._icon.set_icon(name)
        self._update_colors(p)

    def changeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().changeEvent(event)
        if hasattr(self, "_mode"):
            self._update_colors(current_palette())

    def refresh_theme(self) -> None:
        self._update_colors(current_palette())

    def _update_colors(self, p) -> None:  # type: ignore[no-untyped-def]
        if not self.isEnabled():
            color = p.disabled_text
        elif self._mode == "run":
            color = "#FFFFFF"
        else:
            color = p.text
        self._icon.set_icon(self._icon.icon_name, color)
        for lbl in (self._title, self._subtitle):
            base = lbl.styleSheet().split(" color:")[0]
            lbl.setStyleSheet(f"{base} color: {color};")


class OfflinePage(QWidget):
    processing_changed = Signal(bool)
    _request_inspect = Signal(int, str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._cancel = threading.Event()
        self._running = False
        self._inspect_request = 0
        self._input: SessionInfo | DatasetInfo | None = None
        self._input_error: str | None = None
        self._inspecting = False

        self._build_inspector()

        left = QVBoxLayout()
        left.setSpacing(18)
        left.addWidget(self._build_input_card())
        left.addWidget(self._build_options_card())
        left.addStretch(1)

        columns = QHBoxLayout()
        columns.setSpacing(18)
        columns.addLayout(left, 3)
        columns.addWidget(self._build_summary_card(), 2, Qt.AlignmentFlag.AlignTop)

        content = QWidget()
        content.setMaximumWidth(MAX_CONTENT_WIDTH)
        body = QVBoxLayout(content)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(18)
        body.addLayout(columns)
        body.addWidget(self._build_log_card(), 1)

        for card in (self._input_card, self._options_card, self._summary_card):
            card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMinimumHeight(0)

        # Scrolls when the window is too short instead of squeezing the cards.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        scroll.setWidget(content)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll, 1)

        self._set_mode(dataset=False)
        self._reset_options()
        self._refresh_summary()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._drain_log_queue)
        self._poll_timer.start(120)
        self.refresh_theme()

    # ── Construction ──────────────────────────────────────────────────────────

    def _build_inspector(self) -> None:
        self._inspect_timer = QTimer(self)
        self._inspect_timer.setSingleShot(True)
        self._inspect_timer.setInterval(INSPECT_DELAY_MS)
        self._inspect_timer.timeout.connect(self._start_inspection)
        self._inspect_thread = QThread(self)
        self._inspector = _InspectWorker()
        self._inspector.moveToThread(self._inspect_thread)
        self._request_inspect.connect(self._inspector.inspect)
        self._inspector.done.connect(self._on_inspected)
        self._inspect_thread.start()
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_inspector)

    def _stop_inspector(self) -> None:
        self._inspect_thread.quit()
        self._inspect_thread.wait(2000)

    def _build_input_card(self) -> Card:
        card = Card()
        card.body.addWidget(SectionHeader(
            "Select input", "Choose whether to process a single session or all sessions in a dataset.", step=1))

        self._mode_session = ChoiceCard("folder", "Single session", "Process one session folder")
        self._mode_dataset = ChoiceCard("database", "Dataset (all sessions in folder)",
                                        "Process all session folders in a dataset")
        self._mode_session.clicked.connect(lambda: self._set_mode(dataset=False))
        self._mode_dataset.clicked.connect(lambda: self._set_mode(dataset=True))
        modes = QHBoxLayout()
        modes.setSpacing(12)
        modes.addWidget(self._mode_session)
        modes.addWidget(self._mode_dataset)
        card.body.addLayout(modes)

        field = QVBoxLayout()
        field.setSpacing(4)
        self._path_label = label("", "strong")
        self._path_help = label("", "hint")
        field.addWidget(self._path_label)
        field.addWidget(self._path_help)
        field.addSpacing(4)

        self._path = QLineEdit()
        self._path.setClearButtonEnabled(True)
        self._path_action = self._path.addAction(icon("folder", current_palette().muted),
                                                 QLineEdit.ActionPosition.LeadingPosition)
        self._path.textChanged.connect(self._on_path_changed)
        self._browse = QPushButton("  Browse…")
        self._browse.setProperty("variant", "secondary")
        self._browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse.clicked.connect(self._browse_clicked)
        path_row = QHBoxLayout()
        path_row.setSpacing(10)
        path_row.addWidget(self._path, 1)
        path_row.addWidget(self._browse)
        field.addLayout(path_row)
        field.addSpacing(6)

        status = QHBoxLayout()
        status.setSpacing(8)
        self._status_spinner = Spinner(16)
        self._status_spinner.hide()
        self._status_icon = IconLabel("check-circle", 16, "ok")
        self._status_icon.hide()
        self._status = label("", "hint", wrap=True)
        status.addWidget(self._status_spinner, 0, Qt.AlignmentFlag.AlignTop)
        status.addWidget(self._status_icon, 0, Qt.AlignmentFlag.AlignTop)
        status.addWidget(self._status, 1)
        field.addLayout(status)
        card.body.addLayout(field)
        self._input_card = card
        return card

    def _build_options_card(self) -> Card:
        card = Card(spacing=12)
        card.body.addWidget(SectionHeader("Processing options", "Configure how the pipeline should run.", step=2))
        card.body.addWidget(Divider())
        self._opt_skip_rotation = OptionRow(
            "Skip video rotation", "Keep the original video orientation (do not rotate).", False)
        self._opt_nlms = OptionRow(
            "Apply adaptive NLMS decorrelation (EEG channels only)",
            "Improve EEG signal decorrelation using adaptive NLMS. Emotions (fEMG) are not affected.", True)
        self._opt_skip_rotation.check.setToolTip(
            "When checked, the 180° rotation of rawData/video.mp4 is skipped entirely,\n"
            "even if it has not been applied yet.")
        self._opt_nlms.check.setToolTip(
            "Apply the causal NLMS adaptive regression filter to the raw EEG before the cognition step.\n"
            "For each channel, the other channels act as predictors to remove common-mode\n"
            "cross-channel contamination. The emotion (fEMG) features never use it.")
        for row in (self._opt_skip_rotation, self._opt_nlms):
            row.check.toggled.connect(self._refresh_summary)
        card.body.addWidget(self._opt_skip_rotation)
        card.body.addWidget(Divider())
        card.body.addWidget(self._opt_nlms)
        card.body.addWidget(Divider())

        self._emotion_model = QComboBox()
        self._emotion_model.setFixedWidth(330)
        self._emotion_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self._emotion_model.setToolTip(
            "Emotion classifier release used for this run. Filter settings\n"
            "(bandpass / notch) follow the selected model's config.json.")
        self._default_model = resolve_key(DEFAULT_MODEL)
        self._model_labels: dict[str, str] = {}
        for key, release in builtin_releases().items():
            text = f"{release.name} {release.version}".strip()
            self._model_labels[key] = text
            self._emotion_model.addItem(text + (" (recommended)" if key == self._default_model else ""), key)
        self._emotion_model.currentIndexChanged.connect(self._refresh_summary)
        model_row = QHBoxLayout()
        model_text = QVBoxLayout()
        model_text.setSpacing(2)
        model_text.addWidget(label("Emotion model", "strong"))
        model_text.addWidget(label("Choose the emotion recognition model to use.", "hint"))
        model_row.addLayout(model_text, 1)
        model_row.addWidget(self._emotion_model, 0, Qt.AlignmentFlag.AlignVCenter)
        card.body.addLayout(model_row)
        self._options_card = card
        return card

    def _build_summary_card(self) -> Card:
        card = Card(spacing=14)
        card.setMinimumWidth(360)
        self._summary_card = card
        card.body.addWidget(SectionHeader("Run summary", "Review your configuration before running.", icon="file-text"))
        card.body.addWidget(Divider())

        def section(icon_name: str, title: str) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setSpacing(10)
            row.addWidget(IconLabel(icon_name, 18, "muted"))
            row.addWidget(label(title, "section"), 1)
            return row

        grid_input = QGridLayout()
        grid_input.setHorizontalSpacing(12)
        grid_input.setVerticalSpacing(10)
        self._sum_mode = Pill("Single session", "value")
        self._sum_path = label("Not selected", "placeholder", wrap=True)
        self._sum_path.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid_input.addWidget(label("Mode"), 0, 0)
        grid_input.addWidget(self._sum_mode, 0, 1, Qt.AlignmentFlag.AlignRight)
        grid_input.addWidget(label("Selected"), 1, 0, Qt.AlignmentFlag.AlignTop)
        grid_input.addWidget(self._sum_path, 1, 1)
        grid_input.setColumnStretch(1, 1)
        card.body.addLayout(section("folder", "Input"))
        card.body.addLayout(grid_input)
        card.body.addWidget(Divider())

        grid_opts = QGridLayout()
        grid_opts.setHorizontalSpacing(12)
        grid_opts.setVerticalSpacing(10)
        self._sum_rotation = Pill("", "off")
        self._sum_nlms = Pill("", "on")
        self._sum_model = Pill("", "value")
        for i, (name, pill) in enumerate(
            (("Skip video rotation", self._sum_rotation), ("NLMS decorrelation (EEG)", self._sum_nlms),
             ("Emotion model", self._sum_model))
        ):
            grid_opts.addWidget(label(name), i, 0)
            grid_opts.addWidget(pill, i, 1, Qt.AlignmentFlag.AlignRight)
        grid_opts.setColumnStretch(0, 1)
        card.body.addLayout(section("sliders", "Processing options"))
        card.body.addLayout(grid_opts)
        card.body.addSpacing(6)

        self._run = RunButton()
        self._run.clicked.connect(self._run_or_cancel)
        card.body.addWidget(self._run)
        self._run_hint = label("", "hint", wrap=True)
        self._run_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card.body.addWidget(self._run_hint)
        return card

    def _build_log_card(self) -> Card:
        card = Card(spacing=12)
        head = QHBoxLayout()
        head.addWidget(SectionHeader("Processing log", "Real-time pipeline output will appear here.",
                                     icon="file-text"), 1)
        self._clear = QPushButton("  Clear")
        self._clear.setProperty("variant", "secondary")
        self._clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear.clicked.connect(lambda: self._log.clear())
        head.addWidget(self._clear, 0, Qt.AlignmentFlag.AlignTop)
        card.body.addLayout(head)
        self._log = QPlainTextEdit()
        self._log.setObjectName("Log")
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("No logs yet. Run the pipeline to see output here.")
        self._log.setMinimumHeight(140)
        card.body.addWidget(self._log, 1)
        return card

    # ── Theme ─────────────────────────────────────────────────────────────────

    def refresh_theme(self) -> None:
        p = current_palette()
        self._path_action.setIcon(icon("folder", p.muted))
        self._browse.setIcon(icon("folder", p.text, 16))
        self._clear.setIcon(icon("trash", p.text, 16))
        self._status_spinner.set_color(p.accent)
        self._run.refresh_theme()

    # ── Input ─────────────────────────────────────────────────────────────────

    def _is_dataset_mode(self) -> bool:
        return self._mode_dataset.is_selected()

    def _set_mode(self, dataset: bool) -> None:
        self._mode_session.set_selected(not dataset)
        self._mode_dataset.set_selected(dataset)
        title, helper, placeholder, _ = MODE_TEXT[dataset]
        self._path_label.setText(title)
        self._path_help.setText(helper)
        self._path.setPlaceholderText(placeholder)
        self._sum_mode.setText("Dataset" if dataset else "Single session")
        self._on_path_changed(self._path.text())

    def _browse_clicked(self) -> None:
        title = MODE_TEXT[self._is_dataset_mode()][3]
        path = QFileDialog.getExistingDirectory(self, title, self._path.text().strip())
        if path:
            self._path.setText(os.path.normpath(path))

    def _on_path_changed(self, _text: str = "") -> None:
        self._inspect_request += 1
        self._input = None
        self._input_error = None
        if self._path.text().strip():
            self._inspecting = True
            self._inspect_timer.start()
        else:
            self._inspecting = False
            self._inspect_timer.stop()
        self._refresh_status()
        self._refresh_summary()

    def _start_inspection(self) -> None:
        self._request_inspect.emit(self._inspect_request, self._path.text(), self._is_dataset_mode())

    def _on_inspected(self, request: int, result: object, error: object) -> None:
        if request != self._inspect_request:
            return  # the path or mode changed meanwhile
        self._inspecting = False
        self._input = result if isinstance(result, (SessionInfo, DatasetInfo)) else None
        self._input_error = error if isinstance(error, str) else None
        self._refresh_status()
        self._refresh_summary()

    def _refresh_status(self) -> None:
        self._status_spinner.stop()
        self._status_icon.hide()
        self._path.setProperty("invalid", False)
        if self._inspecting:
            self._status_spinner.start()
            self._status.setText("Checking folder…")
            set_role(self._status, "hint")
        elif self._input_error:
            self._status_icon.set_icon("alert", "error")
            self._status_icon.show()
            self._status.setText(self._input_error)
            set_role(self._status, "error")
            self._path.setProperty("invalid", True)
        elif self._input is not None:
            self._status_icon.set_icon("check-circle", "ok")
            self._status_icon.show()
            self._status.setText(self._input.summary())
            set_role(self._status, "strong")
        else:
            self._status.setText("")
        repolish(self._path)

    # ── Options and summary ───────────────────────────────────────────────────

    def _reset_options(self) -> None:
        self._opt_skip_rotation.check.setChecked(False)
        self._opt_nlms.check.setChecked(True)
        idx = self._emotion_model.findData(self._default_model)
        if idx >= 0:
            self._emotion_model.setCurrentIndex(idx)

    def _refresh_summary(self) -> None:
        if self._input is not None:
            self._sum_path.setText(self._input.summary())
            self._sum_path.setToolTip(self._input.path)
            set_role(self._sum_path, "strong")
        else:
            self._sum_path.setText("Not selected")
            self._sum_path.setToolTip("")
            set_role(self._sum_path, "placeholder")
        skip = self._opt_skip_rotation.check.isChecked()
        nlms = self._opt_nlms.check.isChecked()
        self._sum_rotation.set_state("Enabled" if skip else "Disabled", "on" if skip else "off")
        self._sum_nlms.set_state("Enabled" if nlms else "Disabled", "on" if nlms else "off")
        key = self._emotion_model.currentData()
        self._sum_model.setText(self._model_labels.get(key, "") if key else "")
        if not self._running:
            ready = self._input is not None
            self._run.setEnabled(ready)
            if ready:
                self._run_hint.setText("")
            elif self._inspecting:
                self._run_hint.setText("Checking the selected folder…")
            else:
                self._run_hint.setText("Select a valid session or dataset folder to enable Run.")

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if not self._running:
            self._reset_options()

    # ── Run and cancel ────────────────────────────────────────────────────────

    def _run_mode(self) -> str:
        if not self._running:
            return "run"
        return "cancelling" if self._cancel.is_set() else "cancel"

    def can_leave(self) -> bool:
        if self._running:
            QMessageBox.information(self, "Processing", "Processing is running. Cancel it before leaving this page.")
            return False
        return True

    def _set_processing(self, running: bool) -> None:
        self._running = running
        for widget in (self._input_card, self._options_card, self._clear):
            widget.setEnabled(not running)
        self._run.set_mode(self._run_mode())
        if not running:
            self._refresh_summary()
        else:
            self._run_hint.setText("")
        self.processing_changed.emit(running)

    def _run_or_cancel(self) -> None:
        if self._running:
            self._cancel.set()
            self._run.set_mode("cancelling")
            self._insert_log("\n--- Cancel requested: finishing the current step ---\n")
            return
        if self._input is None:
            return
        self._cancel.clear()
        skip_rot = self._opt_skip_rotation.check.isChecked()
        apply_nlms = self._opt_nlms.check.isChecked()
        emotion_model = self._emotion_model.currentData()
        if isinstance(self._input, DatasetInfo):
            self._run_dataset(skip_rot, apply_nlms, emotion_model)
        else:
            self._run_single(skip_rot, apply_nlms, emotion_model)

    def _log_options(self, skip_rot: bool, apply_nlms: bool, emotion_model: str | None) -> None:
        if skip_rot:
            self._insert_log("  (video rotation skipped by user request)\n")
        if not apply_nlms:
            self._insert_log("  (NLMS adaptive decorrelation disabled by user request)\n")
        if emotion_model:
            self._insert_log(f"  (emotion model: {self._model_labels.get(emotion_model, emotion_model)})\n")

    def _run_single(self, skip_rot: bool, apply_nlms: bool, emotion_model: str | None) -> None:
        folder = self._path.text().strip()
        err = session_preflight(folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return
        self._set_processing(True)
        self._insert_log(f"\n--- Starting run: {folder!r} ---\n")
        self._log_options(skip_rot, apply_nlms, emotion_model)
        worker = PipelineWorker(
            folder, None, self._log_queue,
            skip_video_rotation=skip_rot,
            apply_nlms=apply_nlms,
            emotion_model=emotion_model,
            cancel=self._cancel,
        )
        self._start_worker(QThread(), worker)

    def _run_dataset(self, skip_rot: bool, apply_nlms: bool, emotion_model: str | None) -> None:
        folder = self._path.text().strip()
        sessions, err = dataset_preflight(folder)
        if err:
            QMessageBox.warning(self, "Dataset", err)
            return
        n = len(sessions)
        if n > 10:
            answer = QMessageBox.question(
                self,
                "Process entire dataset?",
                f"This will run the pipeline on {n} sessions sequentially.\nThis may take a long time. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._set_processing(True)
        self._insert_log(f"\n--- Starting dataset run: {folder!r} ({n} sessions) ---\n")
        self._log_options(skip_rot, apply_nlms, emotion_model)
        worker = BatchPipelineWorker(
            sessions, self._log_queue,
            skip_video_rotation=skip_rot,
            apply_nlms=apply_nlms,
            emotion_model=emotion_model,
            cancel=self._cancel,
        )
        worker.session_start.connect(lambda text: self._insert_log(f"\n--- Session {text} ---\n"))
        self._start_worker(QThread(), worker)

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
        cancelled = self._cancel.is_set()
        self._drain_log_queue()
        self._set_processing(False)
        self._on_path_changed()  # outputs and layout may have changed: inspect again
        if cancelled:
            QMessageBox.information(self, "Pipeline", "Run cancelled.")
        else:
            QMessageBox.information(self, "Pipeline", "Run finished.")

    def _on_finished_err(self, msg: str) -> None:
        self._drain_log_queue()
        self._set_processing(False)
        self._on_path_changed()
        QMessageBox.critical(self, "Pipeline", msg)

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _insert_log(self, text: str) -> None:
        self._log.moveCursor(QTextCursor.MoveOperation.End)
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
