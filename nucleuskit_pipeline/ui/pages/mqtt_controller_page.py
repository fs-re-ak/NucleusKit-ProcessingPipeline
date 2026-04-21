"""MQTT Controller page — connect to a Vizia Mobile broker and send recording commands."""

from __future__ import annotations

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nucleuskit_pipeline.mqtt.mqtt_controller import (
    DEFAULT_BROKER_PORT,
    DEFAULT_COMMAND_TOPIC_SUFFIX,
    DEFAULT_STATUS_TOPIC,
    TARGET_ALL,
    DeviceStatus,
    MqttController,
)

_SETTINGS_ORG = "NucleusKit"
_SETTINGS_APP = "MqttController"
_KEY_HOST = "broker/host"
_KEY_PORT = "broker/port"
_KEY_STATUS_TOPIC = "broker/status_topic"
_KEY_CMD_SUFFIX = "broker/command_suffix"


class _MqttBridge(QObject):
    """
    Bridges paho callbacks (arbitrary thread) to Qt signals (GUI thread).
    Must live on the GUI thread so signal delivery is automatic via the event loop.
    """

    log_line = Signal(str)
    device_updated = Signal(object)   # DeviceStatus
    connected = Signal()
    disconnected = Signal()


class MqttControllerPage(QWidget):
    """Full-page MQTT controller: broker configuration, device discovery, commands."""

    go_main_menu = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._bridge = _MqttBridge(self)
        self._bridge.log_line.connect(self._append_log)
        self._bridge.device_updated.connect(self._on_device_updated)
        self._bridge.connected.connect(self._on_connected)
        self._bridge.disconnected.connect(self._on_disconnected)

        self._controller: MqttController | None = None

        self._build_ui()
        self._load_settings()

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Top navigation bar ───────────────────────────────────────────
        top_bar = QHBoxLayout()
        back_btn = QPushButton("Main menu")
        back_btn.setProperty("secondary", True)
        back_btn.clicked.connect(self._on_go_main_menu)
        top_bar.addWidget(back_btn)
        top_bar.addStretch(1)
        root.addLayout(top_bar)

        title = QLabel("MQTT Controller")
        title.setProperty("role", "title")
        root.addWidget(title)

        # ── Main splitter: left panel / right log ────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        left_col = QVBoxLayout(left)
        left_col.setContentsMargins(0, 0, 8, 0)

        left_col.addWidget(self._build_broker_group())
        left_col.addSpacing(8)
        left_col.addWidget(self._build_devices_group())
        left_col.addSpacing(8)
        left_col.addWidget(self._build_commands_group())
        left_col.addStretch(1)

        right = QWidget()
        right_col = QVBoxLayout(right)
        right_col.setContentsMargins(8, 0, 0, 0)
        right_col.addWidget(self._build_log_group())

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter, stretch=1)

    def _build_broker_group(self) -> QGroupBox:
        group = QGroupBox("Broker connection")
        form = QFormLayout(group)

        self._host_edit = QLineEdit()
        self._host_edit.setPlaceholderText("e.g. 192.168.1.100")
        self._host_edit.setToolTip("IP address or hostname of the MQTT broker.")
        form.addRow("Host:", self._host_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(DEFAULT_BROKER_PORT)
        form.addRow("Port:", self._port_spin)

        self._status_topic_edit = QLineEdit()
        self._status_topic_edit.setPlaceholderText(DEFAULT_STATUS_TOPIC)
        self._status_topic_edit.setToolTip(
            "Topic on which Vizia devices publish status packets.\n"
            "Default: vizia/status"
        )
        form.addRow("Status topic:", self._status_topic_edit)

        self._cmd_suffix_edit = QLineEdit()
        self._cmd_suffix_edit.setPlaceholderText(DEFAULT_COMMAND_TOPIC_SUFFIX)
        self._cmd_suffix_edit.setToolTip(
            "Suffix appended after {recorderId}/ when publishing commands.\n"
            "Default: control/"
        )
        form.addRow("Command suffix:", self._cmd_suffix_edit)

        btn_row = QHBoxLayout()
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._disconnect_btn.setEnabled(False)
        self._disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        btn_row.addWidget(self._connect_btn)
        btn_row.addWidget(self._disconnect_btn)

        self._status_label = QLabel("Not connected")
        self._status_label.setProperty("role", "hint")

        form.addRow(btn_row)
        form.addRow("Status:", self._status_label)

        return group

    def _build_devices_group(self) -> QGroupBox:
        group = QGroupBox("Discovered devices")
        col = QVBoxLayout(group)

        hint = QLabel(
            "Devices are discovered automatically from incoming status packets.\n"
            "Select a target below before sending commands."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "hint")
        col.addWidget(hint)

        self._device_combo = QComboBox()
        self._device_combo.addItem("All devices", TARGET_ALL)
        self._device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        col.addWidget(self._device_combo)

        self._device_status_label = QLabel("")
        self._device_status_label.setProperty("role", "hint")
        self._device_status_label.setWordWrap(True)
        col.addWidget(self._device_status_label)

        return group

    def _build_commands_group(self) -> QGroupBox:
        group = QGroupBox("Commands")
        col = QVBoxLayout(group)

        rec_row = QHBoxLayout()
        self._start_btn = QPushButton("Start recording")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start_recording)
        self._stop_btn = QPushButton("Stop recording")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_recording)
        rec_row.addWidget(self._start_btn)
        rec_row.addWidget(self._stop_btn)
        col.addLayout(rec_row)

        col.addSpacing(4)

        tag_form = QFormLayout()
        self._tag_code_edit = QLineEdit()
        self._tag_code_edit.setPlaceholderText("e.g. STIMULUS_ON")
        tag_form.addRow("Tag code:", self._tag_code_edit)

        self._tag_values_edit = QLineEdit()
        self._tag_values_edit.setPlaceholderText("optional detail string")
        tag_form.addRow("Tag values:", self._tag_values_edit)
        col.addLayout(tag_form)

        self._tag_btn = QPushButton("Send tag")
        self._tag_btn.setEnabled(False)
        self._tag_btn.clicked.connect(self._on_send_tag)
        col.addWidget(self._tag_btn)

        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Connection log")
        col = QVBoxLayout(group)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        font = QFont("Monospace")
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        font.setPointSize(9)
        self._log_view.setFont(font)
        col.addWidget(self._log_view)

        clear_btn = QPushButton("Clear log")
        clear_btn.setProperty("secondary", True)
        clear_btn.clicked.connect(self._log_view.clear)
        col.addWidget(clear_btn)

        return group

    # ── Settings persistence ─────────────────────────────────────────────

    def _load_settings(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._host_edit.setText(s.value(_KEY_HOST, "", type=str))
        self._port_spin.setValue(int(s.value(_KEY_PORT, DEFAULT_BROKER_PORT)))
        self._status_topic_edit.setText(s.value(_KEY_STATUS_TOPIC, "", type=str))
        self._cmd_suffix_edit.setText(s.value(_KEY_CMD_SUFFIX, "", type=str))

    def _save_settings(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        s.setValue(_KEY_HOST, self._host_edit.text().strip())
        s.setValue(_KEY_PORT, self._port_spin.value())
        s.setValue(_KEY_STATUS_TOPIC, self._status_topic_edit.text().strip())
        s.setValue(_KEY_CMD_SUFFIX, self._cmd_suffix_edit.text().strip())

    # ── Connection actions ───────────────────────────────────────────────

    def _on_connect_clicked(self) -> None:
        host = self._host_edit.text().strip()
        if not host:
            QMessageBox.warning(self, "Host required", "Enter the broker IP address or hostname.")
            return

        port = self._port_spin.value()
        status_topic = self._status_topic_edit.text().strip() or DEFAULT_STATUS_TOPIC
        cmd_suffix = self._cmd_suffix_edit.text().strip() or DEFAULT_COMMAND_TOPIC_SUFFIX

        self._save_settings()

        if self._controller is not None:
            self._controller.disconnect()
            self._controller = None

        self._device_combo.clear()
        self._device_combo.addItem("All devices", TARGET_ALL)
        self._device_status_label.setText("")

        self._controller = MqttController(
            broker_host=host,
            broker_port=port,
            status_topic=status_topic,
            command_topic_suffix=cmd_suffix,
            on_log=self._bridge.log_line.emit,
            on_device_update=self._bridge.device_updated.emit,
            on_connected=self._bridge.connected.emit,
            on_disconnected=self._bridge.disconnected.emit,
        )

        self._connect_btn.setEnabled(False)
        self._status_label.setText("Connecting …")
        self._controller.connect()

    def _on_disconnect_clicked(self) -> None:
        if self._controller is not None:
            self._controller.disconnect()
            self._controller = None
        self._set_disconnected_state()

    # ── Bridge callbacks (GUI thread) ────────────────────────────────────

    def _on_connected(self) -> None:
        self._disconnect_btn.setEnabled(True)
        self._connect_btn.setEnabled(False)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._tag_btn.setEnabled(True)
        self._status_label.setText("Connected")

    def _on_disconnected(self) -> None:
        self._set_disconnected_state()

    def _set_disconnected_state(self) -> None:
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._tag_btn.setEnabled(False)
        self._status_label.setText("Not connected")

    def _on_device_updated(self, status: DeviceStatus) -> None:
        device_id = status.device_id

        found = False
        for i in range(self._device_combo.count()):
            if self._device_combo.itemData(i) == device_id:
                self._device_combo.setItemText(
                    i,
                    f"{device_id}  (recording={'yes' if status.recording else 'no'}, battery={status.battery}%)",
                )
                found = True
                break

        if not found:
            self._device_combo.addItem(
                f"{device_id}  (recording={'yes' if status.recording else 'no'}, battery={status.battery}%)",
                device_id,
            )

        self._refresh_device_status_label()

    def _refresh_device_status_label(self) -> None:
        if self._controller is None:
            self._device_status_label.setText("")
            return
        devices = self._controller.discovered_devices
        if not devices:
            self._device_status_label.setText("No devices seen yet.")
        else:
            lines = [
                f"• {d.device_id}: {'● recording' if d.recording else '○ idle'}, battery {d.battery}%"
                for d in devices
            ]
            self._device_status_label.setText("\n".join(lines))

    # ── Command actions ──────────────────────────────────────────────────

    def _selected_device_id(self) -> str:
        data = self._device_combo.currentData()
        return data if data is not None else TARGET_ALL

    def _on_start_recording(self) -> None:
        if self._controller is None:
            return
        self._controller.send_start_recording(self._selected_device_id())

    def _on_stop_recording(self) -> None:
        if self._controller is None:
            return
        self._controller.send_stop_recording(self._selected_device_id())

    def _on_send_tag(self) -> None:
        if self._controller is None:
            return
        code = self._tag_code_edit.text().strip()
        if not code:
            QMessageBox.warning(self, "Tag code required", "Enter a tag code before sending.")
            return
        values = self._tag_values_edit.text().strip()
        self._controller.send_tag(code, values, self._selected_device_id())

    # ── Log ──────────────────────────────────────────────────────────────

    def _append_log(self, msg: str) -> None:
        self._log_view.appendPlainText(msg)
        self._log_view.verticalScrollBar().setValue(
            self._log_view.verticalScrollBar().maximum()
        )

    # ── Navigation ───────────────────────────────────────────────────────

    def _on_go_main_menu(self) -> None:
        if self._controller is not None and self._controller.is_connected:
            answer = QMessageBox.question(
                self,
                "Disconnect?",
                "You are connected to a broker. Disconnect and return to the main menu?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._on_disconnect_clicked()
        self.go_main_menu.emit()
