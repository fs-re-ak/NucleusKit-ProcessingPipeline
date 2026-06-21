"""EEG Regression Denoising tool page.

Applies EMG reference-channel regression to remove shared muscle noise from EEG
before recomputing all cognition outputs for a session.  The standard pipeline is
not modified; this tool clears the existing cognition cache and regenerates it.

Reference channels used as regressors (CHEEK_R, CHEEK_L, BROW_L, NOSE) are
recorded by the Hermes hardware but are not used in the normal EEG derivation.
"""

from __future__ import annotations

import os
import queue
import shutil
import sys
import traceback

import numpy as np
import pandas as pd

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

from nucleuskit_pipeline.hermes.processor.cognition_processor import (
    GAP_THRESHOLD_S,
    bandpass_filter,
    compute_cognitive_indexes,
    compute_eeg_power_bands,
    regress_noise_channels,
    reject_temporal_artefacts,
    save_eeg_artefact_plot,
    save_filtered_eeg_csv,
    _simple_resample,
)
from nucleuskit_pipeline.hermes.processor.data_interface import HermesDataInterface
from nucleuskit_pipeline.logging_utils import printError, printInfo, printWarning
from nucleuskit_pipeline.ui.offline_job import QueueTextWriter, eeg_regression_preflight


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _EegRegressionWorker(QObject):
    finished_ok  = Signal()
    finished_err = Signal(str)

    def __init__(self, folder: str, log_queue: queue.SimpleQueue[str]) -> None:
        super().__init__()
        self._folder    = folder
        self._log_queue = log_queue

    @Slot()
    def run(self) -> None:
        err: str | None = None
        writer = QueueTextWriter(self._log_queue)
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = writer
            sys.stderr = writer
            self._process()
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

    def _process(self) -> None:
        recpath  = self._folder
        sf       = HermesDataInterface.SAMPLING_RATE
        features_dir     = os.path.join(recpath, "features", "cognition")
        cognitive_path   = os.path.join(recpath, "results", "Cognition.csv")
        powerbands_path  = os.path.join(features_dir, "powerBands.csv")
        artefact_path    = os.path.join(features_dir, "artefactStats.csv")
        temporal_bp_path = os.path.join(features_dir, "temporalBandPowers.csv")
        filtered_eeg_path  = os.path.join(features_dir, "filteredEEG.csv")
        artefact_plot_path = os.path.join(features_dir, "eegArtefactPlot.png")

        # -- Clear existing cognition cache so outputs are fully regenerated --
        printInfo("[eegRegressionTool] Clearing existing cognition outputs")
        if os.path.isfile(cognitive_path):
            os.remove(cognitive_path)
        if os.path.isdir(features_dir):
            shutil.rmtree(features_dir)

        # -- Load EEG ------------------------------------------------------------
        printInfo("[eegRegressionTool] Loading EEG data")
        original_timestamps, eeg_data = HermesDataInterface(recpath).getEEG()

        if eeg_data is None:
            printError("[eegRegressionTool] HermesDataInterface.getEEG returned None — aborting")
            raise RuntimeError("No EEG data available.")

        printInfo(f"[eegRegressionTool] EEG loaded: {len(eeg_data)} samples, "
                  f"columns: {list(eeg_data.columns)}")

        # -- Timestamp gap detection + hardware invalid mask ---------------------
        if original_timestamps is None:
            printWarning(
                "[eegRegressionTool] Hardware timestamps unavailable — "
                "falling back to synthetic timestamps. Gap detection disabled."
            )
            original_timestamps = np.arange(len(eeg_data), dtype=float) / sf
            gap_sample_mask = np.zeros(len(eeg_data), dtype=bool)
        else:
            dt = np.diff(original_timestamps)
            gap_indices = np.where(dt >= GAP_THRESHOLD_S)[0]
            gap_sample_mask = np.zeros(len(original_timestamps), dtype=bool)
            if len(gap_indices):
                printWarning(
                    f"[eegRegressionTool] {len(gap_indices)} hardware timestamp gap(s) "
                    f">= {GAP_THRESHOLD_S:.0f} s detected — affected windows will be NaN."
                )
                gap_sample_mask[gap_indices + 1] = True

        hardware_invalid = eeg_data.isna().any(axis=1).to_numpy() | gap_sample_mask

        # -- Bandpass filter EEG -------------------------------------------------
        printInfo("[eegRegressionTool] Bandpass filtering EEG")
        filtered_eeg = bandpass_filter(eeg_data, sf)

        # -- Load + filter reference channels ------------------------------------
        printInfo("[eegRegressionTool] Loading EMG reference channels")
        _, ref_data = HermesDataInterface(recpath).getReferenceChannels()

        if ref_data is not None:
            printInfo("[eegRegressionTool] Bandpass filtering reference channels")
            filtered_refs = bandpass_filter(ref_data, sf)

            # First pass: denoise all EEG channels with the EMG reference set
            # (CHEEK_R, CHEEK_L, BROW_L, NOSE).
            printInfo("[eegRegressionTool] Applying EMG regression denoising (all channels)")
            filtered_eeg = regress_noise_channels(filtered_eeg, filtered_refs)

            # Second pass: denoise T9/T10 with an extended reference set that
            # additionally includes AF7 and AF8.  These frontal channels capture
            # EOG (eye blinks, lateral movements) and frontal movement artefacts
            # that contaminate the temporal electrodes via the shared EAR_R
            # midpoint reference.  Only T9/T10 are targeted to avoid regressing
            # frontal brain activity out of the frontal channels themselves.
            if {'AF7', 'AF8', 'T9', 'T10'}.issubset(filtered_eeg.columns):
                printInfo(
                    "[eegRegressionTool] Applying extended regression on T9/T10 "
                    "using AF7+AF8 as additional EOG/movement references"
                )
                extended_refs = pd.concat(
                    [filtered_refs, filtered_eeg[['AF7', 'AF8']]],
                    axis=1,
                )
                cleaned_temporal = regress_noise_channels(
                    filtered_eeg[['T9', 'T10']],
                    extended_refs,
                )
                filtered_eeg = filtered_eeg.copy()
                filtered_eeg[['T9', 'T10']] = cleaned_temporal[['T9', 'T10']]
            else:
                printWarning(
                    "[eegRegressionTool] AF7/AF8 or T9/T10 missing from filtered EEG "
                    "— skipping extended T9/T10 regression"
                )
        else:
            printWarning(
                "[eegRegressionTool] Reference channels unavailable — "
                "proceeding without regression denoising"
            )

        # -- Artefact rejection --------------------------------------------------
        printInfo("[eegRegressionTool] Running temporal artefact rejection")
        artefact_mask, artefact_stats, epoch_metrics = reject_temporal_artefacts(
            filtered_eeg, hardware_invalid, sf
        )
        combined_invalid = hardware_invalid | artefact_mask

        # -- Save filtered EEG + artefact plot -----------------------------------
        os.makedirs(features_dir, exist_ok=True)
        save_filtered_eeg_csv(filtered_eeg, original_timestamps, filtered_eeg_path)
        save_eeg_artefact_plot(filtered_eeg, original_timestamps, combined_invalid, artefact_plot_path)

        if artefact_stats is not None:
            artefact_stats.to_csv(artefact_path, index=False)
            printInfo(f"[eegRegressionTool] Artefact stats saved to {artefact_path}")

        epoch_metrics_path = os.path.join(features_dir, "epochMetrics.csv")
        if epoch_metrics is not None:
            epoch_metrics.to_csv(epoch_metrics_path, index=False)
            printInfo(f"[eegRegressionTool] Epoch metrics saved to {epoch_metrics_path}")

        # -- Power bands ---------------------------------------------------------
        printInfo("[eegRegressionTool] Computing power bands")
        powerbands = compute_eeg_power_bands(
            filtered_eeg[['T9', 'T10']],
            timestamps=original_timestamps,
            hardware_invalid=combined_invalid,
        )
        if powerbands is None:
            raise RuntimeError("compute_eeg_power_bands returned None — recording may be too short.")

        powerbands.to_csv(powerbands_path, index=False)
        printInfo(f"[eegRegressionTool] Power bands saved to {powerbands_path}")

        # -- Cognitive indexes ---------------------------------------------------
        printInfo("[eegRegressionTool] Computing cognitive indexes")
        result, temporal_bands = compute_cognitive_indexes(powerbands)
        if result is None:
            raise RuntimeError("compute_cognitive_indexes returned None.")

        if temporal_bands is not None:
            temporal_bands.to_csv(temporal_bp_path, index=False)
            printInfo(f"[eegRegressionTool] Temporal band powers saved to {temporal_bp_path}")

        result = _simple_resample(result, target_interval=0.5)

        os.makedirs(os.path.join(recpath, "results"), exist_ok=True)
        result.to_csv(cognitive_path, index=False)
        printInfo(f"[eegRegressionTool] Cognition results saved to {cognitive_path}")

        printInfo("[eegRegressionTool] Done.")


# ---------------------------------------------------------------------------
# UI page
# ---------------------------------------------------------------------------

class EegRegressionPage(QWidget):
    go_tools_menu      = Signal()
    processing_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._log_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread: QThread | None            = None
        self._worker: _EegRegressionWorker | None = None

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
            "Session  (must contain rawData/ with a raw EEG file)"
        )
        sg = QVBoxLayout(session_box)
        sg.addLayout(sess_row)

        # ── actions ───────────────────────────────────────────────────
        self._run = QPushButton("Run EEG Regression Denoising")
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
        err    = eeg_regression_preflight(folder)
        if err:
            QMessageBox.warning(self, "Session", err)
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "Busy", "A run is already in progress.")
            return

        self._set_processing(True)
        self._insert_log(f"\n--- Starting EEG Regression Denoising: {folder!r} ---\n")

        thread = QThread()
        worker = _EegRegressionWorker(folder, self._log_queue)
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
            "EEG Regression Denoising",
            "Done.\n\n"
            "Cognition outputs regenerated with EMG regression applied.\n"
            "results/Cognition.csv and features/cognition/ have been updated.",
        )

    def _on_finished_err(self, msg: str) -> None:
        self._set_processing(False)
        QMessageBox.critical(self, "EEG Regression Denoising", msg)

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
