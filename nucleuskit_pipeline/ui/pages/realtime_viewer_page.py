"""Real-time Hermes viewer: device selection, BLE scan, connect, plot, recording."""

from __future__ import annotations

import asyncio
import csv
import os
import platform
import sys
import threading
from datetime import datetime

import numpy as np
from PySide6.QtCore import QObject, QSettings, QStandardPaths, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.hermes.realtime.proxy import HermesBleProxy
from nucleuskit_pipeline.shimmer import ShimmerSerialProxy
from nucleuskit_pipeline.ui.realtime_eeg_plot import RealtimeEegPlot
from nucleuskit_pipeline.ui.realtime_hermes_motion_plot import RealtimeHermesMotionPlot
from nucleuskit_pipeline.ui.realtime_shimmer_plot import RealtimeShimmerPlot

PAGE_DEVICE = 0
PAGE_HERMES_SCAN = 1
PAGE_SHIMMER_COM = 2
PAGE_STREAM = 3


class StreamBridge(QObject):
    """Marshals EEG rows from BLE worker threads to the GUI thread."""

    eeg = Signal(object)


class ShimmerStreamBridge(QObject):
    """Marshals Shimmer decoded rows to the GUI thread (IMU + EDA + PPG for plotting)."""

    sample = Signal(float, float, float, float, float)


class MotionStreamBridge(QObject):
    """Marshals Hermes motion tuples from the BLE worker thread to the GUI thread."""

    motion = Signal(float, float, float, float, float, float, float, float, float)


def _bleak_scan_hermes_devices(timeout_s: float = 8.0) -> list[tuple[str, str]]:
    """
    Run Bleak discovery in the current thread.
    On Windows, asyncio must use SelectorEventLoop in worker threads (not the default
    Proactor), or BleakScanner.discover can hang indefinitely with no error output.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    from bleak import BleakScanner

    from nucleuskit_pipeline.hermes.realtime.proxy import HERMES_NAME_SUBSTRING

    async def discover() -> list[tuple[str, str]]:
        devices = await BleakScanner.discover(timeout=timeout_s)
        return [
            (d.name or "", d.address)
            for d in devices
            if d.name and HERMES_NAME_SUBSTRING in d.name
        ]

    return asyncio.run(discover())


class HermesScanThread(QThread):
    """Runs BLE scan on a background thread; keeps a strong C++/Python ref (no QObject GC bug)."""

    scan_finished = Signal(list)
    scan_failed = Signal(str)

    def run(self) -> None:
        try:
            result = _bleak_scan_hermes_devices(8.0)
            self.scan_finished.emit(result)
        except Exception as e:
            self.scan_failed.emit(str(e))


class WaitConnectThread(QThread):
    """Blocks on HermesBleProxy.wait_until_connected() without blocking the GUI thread."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, proxy: HermesBleProxy) -> None:
        super().__init__()
        self._proxy = proxy

    def run(self) -> None:
        try:
            self._proxy.wait_until_connected()
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


class WaitShimmerConnectThread(QThread):
    """Blocks on ShimmerSerialProxy.wait_until_connected() without blocking the GUI thread."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, proxy: ShimmerSerialProxy) -> None:
        super().__init__()
        self._proxy = proxy

    def run(self) -> None:
        try:
            self._proxy.wait_until_connected()
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


EEG_COLUMNS = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]
MOTION_COLUMNS = ["x(g)", "y(g)", "z(g)", "x(deg/s)", "y(deg/s)", "z(deg/s)", "x(G)", "y(G)", "z(G)"]


class RealtimeViewerPage(QWidget):
    go_main_menu = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stack = QStackedWidget()
        self._bridge = StreamBridge(self)
        self._shimmer_bridge = ShimmerStreamBridge(self)
        self._motion_bridge = MotionStreamBridge(self)
        self._proxy: HermesBleProxy | None = None
        self._shimmer_proxy: ShimmerSerialProxy | None = None
        self._connecting = False
        self._recording = False
        self._record_lock = threading.Lock()
        self._eeg_file = None
        self._motion_file = None
        self._eeg_writer = None
        self._motion_writer = None
        self._shimmer_file = None
        self._shimmer_writer = None

        self._scan_thread: HermesScanThread | None = None
        self._wait_thread: WaitConnectThread | None = None
        self._wait_shimmer_thread: WaitShimmerConnectThread | None = None

        self._build_device_step()
        self._build_scan_step()
        self._build_shimmer_com_step()
        self._build_streaming_step()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        top = QHBoxLayout()
        top_nav = QPushButton("Main menu")
        top_nav.setProperty("secondary", True)
        top_nav.clicked.connect(self._request_main_menu)
        top.addWidget(top_nav)
        top.addStretch(1)
        root.addLayout(top)

        root.addWidget(self._stack, stretch=1)

        self._stack.setCurrentIndex(PAGE_DEVICE)

    def can_navigate_to_main_menu(self) -> bool:
        return self._proxy is None and self._shimmer_proxy is None and not self._connecting

    def _is_streaming_connected(self) -> bool:
        if self._proxy is not None and self._proxy.is_connected:
            return True
        if self._shimmer_proxy is not None and self._shimmer_proxy.is_connected:
            return True
        return False

    def _address_column_title(self) -> str:
        if platform.system() == "Darwin":
            return "Address (UUID on macOS)"
        return "MAC address"

    def _build_device_step(self) -> None:
        w = QWidget()
        title = QLabel("Select device type")
        title.setProperty("role", "title")

        self._radio_hermes = QRadioButton("Hermes Headset")
        self._radio_shimmer = QRadioButton("Shimmer wristband")
        self._radio_hermes.setChecked(True)

        hint = QLabel(
            "Shimmer: pair the wristband in Windows Bluetooth settings (Classic), then connect by MAC address."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)

        self._device_next = QPushButton("Next")
        self._device_next.clicked.connect(self._on_device_next)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._device_next)

        col = QVBoxLayout(w)
        col.addWidget(title)
        col.addSpacing(8)
        col.addWidget(self._radio_hermes)
        col.addWidget(self._radio_shimmer)
        col.addWidget(hint)
        col.addStretch(1)
        col.addLayout(row)

        self._stack.addWidget(w)

    def _on_device_next(self) -> None:
        if self._radio_hermes.isChecked():
            self._stack.setCurrentIndex(PAGE_HERMES_SCAN)
        else:
            self._stack.setCurrentIndex(PAGE_SHIMMER_COM)

    def _build_scan_step(self) -> None:
        w = QWidget()
        title = QLabel("Discover Hermes headset")
        title.setProperty("role", "title")

        self._addr_hint = QLabel(f"Discovered devices show name and {self._address_column_title()}.")
        self._addr_hint.setWordWrap(True)

        self._device_list = QListWidget()
        self._device_list.itemDoubleClicked.connect(lambda _i: self._on_connect_clicked())

        self._scan_btn = QPushButton("Scan")
        self._scan_btn.clicked.connect(self._start_scan)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        self._connect_btn.setEnabled(False)
        self._device_list.itemSelectionChanged.connect(self._update_connect_enabled)

        row_btns = QHBoxLayout()
        row_btns.addWidget(self._scan_btn)
        row_btns.addWidget(self._connect_btn)
        row_btns.addStretch(1)

        back = QPushButton("Back")
        back.setProperty("secondary", True)
        back.clicked.connect(self._scan_back)

        row_nav = QHBoxLayout()
        row_nav.addWidget(back)
        row_nav.addStretch(1)

        col = QVBoxLayout(w)
        col.addWidget(title)
        col.addWidget(self._addr_hint)
        col.addWidget(self._device_list, stretch=1)
        col.addLayout(row_btns)
        col.addLayout(row_nav)

        self._stack.addWidget(w)

    def _build_shimmer_com_step(self) -> None:
        w = QWidget()
        title = QLabel("Shimmer wristband — connect")
        title.setProperty("role", "title")

        # ── Paired device scan ───────────────────────────────────────────
        scan_label = QLabel("Paired Bluetooth devices")
        scan_label.setProperty("role", "section")

        scan_hint = QLabel(
            "Tap Scan to list devices already paired in Windows Bluetooth settings. "
            "Click a device to fill in its MAC address automatically."
        )
        scan_hint.setWordWrap(True)

        self._shimmer_scan_btn = QPushButton("Scan paired devices")
        self._shimmer_scan_btn.clicked.connect(self._on_shimmer_scan)

        self._shimmer_device_list = QListWidget()
        self._shimmer_device_list.setMaximumHeight(140)
        self._shimmer_device_list.itemClicked.connect(self._on_shimmer_device_selected)

        # ── Separator ───────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)

        # ── Manual MAC entry ─────────────────────────────────────────────
        manual_label = QLabel("MAC address")
        manual_label.setProperty("role", "section")

        row_mac = QHBoxLayout()
        row_mac.addWidget(QLabel("MAC address:"))
        self._shimmer_mac_edit = QLineEdit()
        self._shimmer_mac_edit.setPlaceholderText("e.g. 00:06:66:B7:00:00")
        self._shimmer_mac_edit.setMaxLength(17)
        self._shimmer_mac_edit.setToolTip(
            "Bluetooth MAC address in XX:XX:XX:XX:XX:XX format (colons or hyphens).\n"
            "Find it in Windows Bluetooth settings or on the Shimmer device label."
        )
        self._shimmer_mac_edit.textChanged.connect(self._on_shimmer_mac_changed)
        row_mac.addWidget(self._shimmer_mac_edit, stretch=1)

        # ── Action buttons ───────────────────────────────────────────────
        self._shimmer_connect_btn = QPushButton("Connect")
        self._shimmer_connect_btn.setEnabled(False)
        self._shimmer_connect_btn.clicked.connect(self._on_shimmer_connect_clicked)

        back = QPushButton("Back")
        back.setProperty("secondary", True)
        back.clicked.connect(self._shimmer_com_back)

        row_btns = QHBoxLayout()
        row_btns.addWidget(back)
        row_btns.addStretch(1)
        row_btns.addWidget(self._shimmer_connect_btn)

        col = QVBoxLayout(w)
        col.addWidget(title)
        col.addSpacing(8)
        col.addWidget(scan_label)
        col.addWidget(scan_hint)
        col.addWidget(self._shimmer_scan_btn)
        col.addWidget(self._shimmer_device_list)
        col.addSpacing(8)
        col.addWidget(sep)
        col.addSpacing(8)
        col.addWidget(manual_label)
        col.addLayout(row_mac)
        col.addStretch(1)
        col.addLayout(row_btns)

        self._stack.addWidget(w)
        self._load_shimmer_mac()

    @staticmethod
    def _scan_paired_bt_devices() -> list[tuple[str, str]]:
        """Return (name, MAC) pairs for all Bluetooth Classic devices paired in Windows.

        Reads from the registry key where Windows stores paired BT device info.
        Returns an empty list on non-Windows platforms or if the key cannot be opened.
        """
        if sys.platform != "win32":
            return []
        try:
            import winreg

            key_path = r"SYSTEM\CurrentControlSet\Services\BTHPORT\Parameters\Devices"
            root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
            devices: list[tuple[str, str]] = []
            idx = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(root, idx)
                    idx += 1
                    if len(sub_name) != 12:
                        continue
                    mac = ":".join(sub_name[j : j + 2] for j in range(0, 12, 2)).upper()
                    try:
                        sub = winreg.OpenKey(root, sub_name)
                        name_raw, _ = winreg.QueryValueEx(sub, "Name")
                        winreg.CloseKey(sub)
                        if isinstance(name_raw, bytes):
                            name = name_raw.rstrip(b"\x00").decode("utf-8", errors="replace")
                        else:
                            name = str(name_raw)
                    except OSError:
                        name = mac
                    devices.append((name, mac))
                except OSError:
                    break
            winreg.CloseKey(root)
            return devices
        except Exception:
            return []

    def _on_shimmer_scan(self) -> None:
        self._shimmer_scan_btn.setEnabled(False)
        self._shimmer_device_list.clear()
        all_devices = self._scan_paired_bt_devices()
        devices = [(n, m) for n, m in all_devices if "shimmer" in n.lower()]
        if not devices:
            if sys.platform != "win32":
                msg = "Paired device scan is only available on Windows. Enter the MAC address manually."
            elif not all_devices:
                msg = "No paired Bluetooth devices found. Pair the Shimmer in Windows Bluetooth settings first."
            else:
                msg = (
                    f"No Shimmer devices found among {len(all_devices)} paired device(s). "
                    "Ensure the Shimmer is paired and its name contains 'Shimmer'."
                )
            item = QListWidgetItem(msg)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._shimmer_device_list.addItem(item)
        else:
            for name, mac in devices:
                item = QListWidgetItem(f"{name}  |  {mac}")
                item.setData(Qt.ItemDataRole.UserRole, mac)
                self._shimmer_device_list.addItem(item)
        self._shimmer_scan_btn.setEnabled(True)

    def _on_shimmer_device_selected(self, item: QListWidgetItem) -> None:
        mac = item.data(Qt.ItemDataRole.UserRole)
        if mac:
            self._shimmer_mac_edit.setText(mac)

    def _shimmer_com_back(self) -> None:
        if self._connecting:
            QMessageBox.information(
                self,
                "Please wait",
                "Connection in progress. Wait until it completes or fails before going back.",
            )
            return
        if self._is_streaming_connected():
            QMessageBox.information(
                self,
                "Disconnect required",
                "Disconnect from the device before going back.",
            )
            return
        self._stack.setCurrentIndex(PAGE_DEVICE)

    def _shimmer_mac_is_valid(self) -> bool:
        return ShimmerSerialProxy.is_mac_address(self._shimmer_mac_edit.text())

    def _load_shimmer_mac(self) -> None:
        s = QSettings("NucleusKit", "RealtimeViewer")
        mac = s.value("shimmer/mac", "", type=str)
        if mac:
            self._shimmer_mac_edit.setText(mac)

    def _save_shimmer_mac(self, mac: str) -> None:
        s = QSettings("NucleusKit", "RealtimeViewer")
        s.setValue("shimmer/mac", mac)

    def _on_shimmer_mac_changed(self, text: str) -> None:
        valid = ShimmerSerialProxy.is_mac_address(text) if text.strip() else True
        style = "" if valid else "QLineEdit { border: 1px solid #c0392b; }"
        self._shimmer_mac_edit.setStyleSheet(style)
        self._shimmer_connect_btn.setEnabled(self._shimmer_mac_is_valid())

    def _on_shimmer_connect_clicked(self) -> None:
        mac_text = self._shimmer_mac_edit.text().strip()

        if not mac_text:
            QMessageBox.information(
                self,
                "MAC address required",
                "Enter the Shimmer MAC address or scan for paired devices above.\n\n"
                "Find it in Windows Bluetooth settings or on the Shimmer device label.",
            )
            return

        if not ShimmerSerialProxy.is_mac_address(mac_text):
            QMessageBox.warning(
                self,
                "Invalid MAC address",
                f"'{mac_text}' is not a valid Bluetooth MAC address.\n\n"
                "Expected format: XX:XX:XX:XX:XX:XX (e.g. 00:06:66:B7:00:00).",
            )
            return

        target = mac_text.upper().replace("-", ":")
        self._connecting = True
        self._shimmer_connect_btn.setEnabled(False)
        self._shimmer_scan_btn.setEnabled(False)

        def sample_cb(row: list[float]) -> None:
            self._shimmer_bridge.sample.emit(row[1], row[2], row[3], row[4], row[5])
            with self._record_lock:
                w = self._shimmer_writer
                if w is not None:
                    w.writerow(row)

        self._shimmer_proxy = ShimmerSerialProxy(target, sample_cb)

        self._wait_shimmer_thread = WaitShimmerConnectThread(self._shimmer_proxy)
        self._wait_shimmer_thread.done.connect(lambda: self._on_shimmer_connect_done(target))
        self._wait_shimmer_thread.failed.connect(self._on_shimmer_connect_failed)
        self._wait_shimmer_thread.finished.connect(self._cleanup_wait_shimmer_thread)
        self._wait_shimmer_thread.start()

    def _cleanup_wait_shimmer_thread(self) -> None:
        self._wait_shimmer_thread = None
        self._shimmer_scan_btn.setEnabled(True)
        self._shimmer_connect_btn.setEnabled(self._shimmer_mac_is_valid())

    def _on_shimmer_connect_done(self, connected_mac: str | None = None) -> None:
        self._connecting = False
        if connected_mac:
            self._save_shimmer_mac(connected_mac)
        self._plot_stack.setCurrentIndex(1)
        self._stack.setCurrentIndex(PAGE_STREAM)
        self._rec_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(True)
        self._ear_r_ref_checkbox.setVisible(False)
        self._eeg_plot.set_ear_r_re_reference(False)
        self._shimmer_plot.clear_buffers()
        self._stream_hint.setText("Streaming Shimmer wristband. Use the controls below.")

    @staticmethod
    def _format_shimmer_serial_error(msg: str) -> str:
        lower = msg.lower()
        if any(
            s in lower
            for s in (
                "timed out",
                "winerror",
                "connection refused",
                "no route to host",
                "target machine actively refused",
                "wsaehostunreach",
                "a socket operation was attempted",
                "ack",
                "inquiry",
            )
        ):
            return (
                f"{msg}\n\n"
                "Ensure the Shimmer is powered on, awake, and paired in Windows Bluetooth settings. "
                "The device must be in range and not already connected to another host or app "
                "(e.g. Shimmer Consensys). Try power-cycling the Shimmer and connecting again."
            )
        return msg

    def _on_shimmer_connect_failed(self, msg: str) -> None:
        self._connecting = False
        if self._shimmer_proxy is not None:
            self._shimmer_proxy.disconnect()
            self._shimmer_proxy = None
        QMessageBox.warning(self, "Connection failed", self._format_shimmer_serial_error(msg))

    def _update_connect_enabled(self) -> None:
        self._connect_btn.setEnabled(len(self._device_list.selectedItems()) > 0)

    def _scan_back(self) -> None:
        if self._connecting:
            QMessageBox.information(
                self,
                "Please wait",
                "Connection in progress. Wait until it completes or fails before going back.",
            )
            return
        if self._is_streaming_connected():
            QMessageBox.information(
                self,
                "Disconnect required",
                "Disconnect from the device before going back.",
            )
            return
        self._stack.setCurrentIndex(PAGE_DEVICE)

    def _start_scan(self) -> None:
        if self._scan_thread is not None and self._scan_thread.isRunning():
            return

        self._scan_btn.setEnabled(False)
        self._device_list.clear()

        self._scan_thread = HermesScanThread()
        self._scan_thread.scan_finished.connect(self._on_scan_finished)
        self._scan_thread.scan_failed.connect(self._on_scan_failed)
        self._scan_thread.finished.connect(self._cleanup_scan_thread)
        self._scan_thread.start()

    def _cleanup_scan_thread(self) -> None:
        self._scan_thread = None
        self._scan_btn.setEnabled(True)

    def _on_scan_finished(self, items: list[tuple[str, str]]) -> None:
        for name, address in items:
            it = QListWidgetItem(f"{name}  |  {address}")
            it.setData(Qt.ItemDataRole.UserRole, address)
            self._device_list.addItem(it)
        if not items:
            QMessageBox.information(self, "Scan", "No Hermes devices found. Try scanning again.")

    def _on_scan_failed(self, msg: str) -> None:
        QMessageBox.warning(self, "Scan failed", msg)

    def _on_connect_clicked(self) -> None:
        items = self._device_list.selectedItems()
        if not items:
            QMessageBox.information(self, "No device", "Select a device from the list.")
            return

        address = items[0].data(Qt.ItemDataRole.UserRole)
        if not address:
            return

        self._connecting = True
        self._connect_btn.setEnabled(False)
        self._scan_btn.setEnabled(False)

        def eeg_cb(samples: list) -> None:
            arr = np.asarray(samples, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 8:
                self._eeg_plot.enqueue_samples(arr)
            with self._record_lock:
                w = self._eeg_writer
                if w is not None:
                    w.writerows(samples)

        def motion_cb(sample: tuple) -> None:
            self._motion_bridge.motion.emit(*sample)
            with self._record_lock:
                w = self._motion_writer
                if w is not None:
                    w.writerow([f"{x:.5f}" for x in sample])

        self._proxy = HermesBleProxy(address, eeg_callback=eeg_cb, motion_callback=motion_cb)

        self._wait_thread = WaitConnectThread(self._proxy)
        self._wait_thread.done.connect(self._on_connect_done)
        self._wait_thread.failed.connect(self._on_connect_failed)
        self._wait_thread.finished.connect(self._cleanup_wait_thread)
        self._wait_thread.start()

    def _cleanup_wait_thread(self) -> None:
        self._wait_thread = None
        self._connect_btn.setEnabled(len(self._device_list.selectedItems()) > 0)
        self._scan_btn.setEnabled(True)

    def _on_connect_done(self) -> None:
        self._connecting = False
        self._plot_stack.setCurrentIndex(0)
        self._stack.setCurrentIndex(PAGE_STREAM)
        self._rec_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(True)
        self._ear_r_ref_checkbox.setVisible(True)
        self._ear_r_ref_checkbox.setEnabled(True)
        self._eeg_plot.set_ear_r_re_reference(self._ear_r_ref_checkbox.isChecked())
        self._eeg_plot.clear_buffers()
        self._hermes_motion_plot.clear_buffers()
        self._stream_hint.setText("Streaming EEG and 9-axis motion. Use the controls below.")

    def _on_connect_failed(self, msg: str) -> None:
        self._connecting = False
        if self._proxy is not None:
            self._proxy.disconnect()
            self._proxy = None
        QMessageBox.warning(self, "Connection failed", msg)

    def _build_streaming_step(self) -> None:
        w = QWidget()
        self._plot_stack = QStackedWidget()

        hermes_stream = QWidget()
        hermes_split = QSplitter(Qt.Orientation.Vertical)
        self._eeg_plot = RealtimeEegPlot(hermes_stream)
        self._hermes_motion_plot = RealtimeHermesMotionPlot(hermes_stream)
        hermes_split.addWidget(self._eeg_plot)
        hermes_split.addWidget(self._hermes_motion_plot)
        hermes_split.setStretchFactor(0, 2)
        hermes_split.setStretchFactor(1, 1)
        hermes_outer = QVBoxLayout(hermes_stream)
        hermes_outer.setContentsMargins(0, 0, 0, 0)
        hermes_outer.addWidget(hermes_split)

        self._shimmer_plot = RealtimeShimmerPlot(w)
        self._plot_stack.addWidget(hermes_stream)
        self._plot_stack.addWidget(self._shimmer_plot)

        # EEG samples are enqueued directly from the BLE worker thread into the
        # plot's thread-safe buffer; the plot's QTimer flushes at ~30 FPS.
        # This removes the high-rate Qt signal that was flooding the event queue.
        self._motion_bridge.motion.connect(self._hermes_motion_plot.add_motion_sample)
        self._shimmer_bridge.sample.connect(self._shimmer_plot.add_sample)

        self._stream_hint = QLabel("Streaming. Use the controls below.")
        self._stream_hint.setProperty("role", "hint")
        self._stream_hint.setWordWrap(True)

        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.setEnabled(False)
        self._disconnect_btn.clicked.connect(self._on_disconnect)

        self._rec_btn = QPushButton("Start recording")
        self._rec_btn.setEnabled(False)
        self._rec_btn.clicked.connect(self._toggle_recording)

        self._ear_r_ref_checkbox = QCheckBox("Re-reference all channels to EAR_R ÷ 2 (before filter)")
        self._ear_r_ref_checkbox.setEnabled(False)
        self._ear_r_ref_checkbox.setToolTip(
            "Subtract EAR_R/2 from every channel, then apply the viewer bandpass. "
            "Same scheme as offline loadEXG(re_reference=True)."
        )
        self._ear_r_ref_checkbox.toggled.connect(self._eeg_plot.set_ear_r_re_reference)

        row = QHBoxLayout()
        row.addWidget(self._disconnect_btn)
        row.addWidget(self._rec_btn)
        row.addWidget(self._ear_r_ref_checkbox)
        row.addStretch(1)

        col = QVBoxLayout(w)
        col.addWidget(self._stream_hint)
        col.addWidget(self._plot_stack, stretch=1)
        col.addLayout(row)

        self._stack.addWidget(w)

    def _request_main_menu(self) -> None:
        if self._connecting:
            QMessageBox.information(
                self,
                "Please wait",
                "Connection in progress. Wait until it completes or fails before returning to the main menu.",
            )
            return
        if self._is_streaming_connected():
            QMessageBox.information(
                self,
                "Disconnect required",
                "Disconnect from the device before returning to the main menu.",
            )
            return
        self.go_main_menu.emit()

    def _toggle_recording(self) -> None:
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _default_record_dir(self) -> str:
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        folder = os.path.join(base, "NucleusKitRealtime")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _start_recording(self) -> None:
        folder = self._default_record_dir()
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self._plot_stack.currentIndex() == 0:
            eeg_path = os.path.join(folder, f"hermes_eeg_{ts}.csv")
            motion_path = os.path.join(folder, f"hermes_motion_{ts}.csv")
            self._eeg_file = open(eeg_path, "w", newline="")
            self._motion_file = open(motion_path, "w", newline="")
            self._eeg_writer = csv.writer(self._eeg_file)
            self._motion_writer = csv.writer(self._motion_file)
            self._eeg_writer.writerow(EEG_COLUMNS)
            self._motion_writer.writerow(MOTION_COLUMNS)
            self._stream_hint.setText(f"Recording to:\n{eeg_path}\n{motion_path}")
        else:
            shimmer_path = os.path.join(folder, f"rawShimmer_{ts}.csv")
            self._shimmer_file = open(shimmer_path, "w", newline="")
            self._shimmer_writer = csv.writer(self._shimmer_file)
            self._stream_hint.setText(
                f"Recording to:\n{shimmer_path}\n"
                "(columns: epoch_ms, IMU_x, IMU_y, IMU_z, EDA, PPG — no header, offline-compatible)"
            )

        self._recording = True
        self._rec_btn.setText("Stop recording")

    def _stop_recording(self) -> None:
        with self._record_lock:
            self._eeg_writer = None
            self._motion_writer = None
            self._shimmer_writer = None
            if self._eeg_file is not None:
                self._eeg_file.close()
                self._eeg_file = None
            if self._motion_file is not None:
                self._motion_file.close()
                self._motion_file = None
            if self._shimmer_file is not None:
                self._shimmer_file.close()
                self._shimmer_file = None
        self._recording = False
        self._rec_btn.setText("Start recording")
        if self._plot_stack.currentIndex() == 0:
            self._stream_hint.setText("Streaming EEG and 9-axis motion. Use the controls below.")
        else:
            self._stream_hint.setText("Streaming Shimmer wristband. Use the controls below.")

    def _on_disconnect(self) -> None:
        if self._recording:
            self._stop_recording()

        if self._proxy is not None:
            self._proxy.disconnect()
            self._proxy = None

        if self._shimmer_proxy is not None:
            self._shimmer_proxy.disconnect()
            self._shimmer_proxy = None

        self._rec_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(False)
        self._ear_r_ref_checkbox.setVisible(True)
        self._ear_r_ref_checkbox.setEnabled(False)
        self._ear_r_ref_checkbox.setChecked(False)
        self._eeg_plot.set_ear_r_re_reference(False)
        self._rec_btn.setText("Start recording")
        self._plot_stack.setCurrentIndex(0)
        self._stream_hint.setText("Streaming. Use the controls below.")
        self._eeg_plot.clear_buffers()
        self._hermes_motion_plot.clear_buffers()
        self._shimmer_plot.clear_buffers()
        self._stack.setCurrentIndex(PAGE_DEVICE)
