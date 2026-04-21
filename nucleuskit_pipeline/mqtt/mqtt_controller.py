"""
MQTT controller for Vizia Mobile Android devices.

Connects to a user-configured MQTT broker and drives recording sessions via
the Vizia Mobile MQTT interface (see docs/mqtt_spec.md for the full contract).

Thread-safety: paho-mqtt runs its own network loop in a background thread.
All public methods are safe to call from any thread. GUI consumers should
connect to the Qt signals defined in MqttBridge instead of calling these
methods directly from UI callbacks.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

DEFAULT_BROKER_PORT = 1883
DEFAULT_STATUS_TOPIC = "vizia/status"
DEFAULT_COMMAND_TOPIC_SUFFIX = "control/"

TARGET_ALL = "__ALL__"


@dataclass
class DeviceStatus:
    device_id: str
    recording: bool = False
    battery: int = -1
    last_seen: float = field(default_factory=time.time)


class MqttController:
    """
    Manages a single MQTT connection to a Vizia Mobile broker.

    Parameters
    ----------
    broker_host : str
        IP address or hostname of the MQTT broker.
    broker_port : int
        TCP port of the broker (default 1883).
    status_topic : str
        Topic on which devices publish status packets (default "vizia/status").
    command_topic_suffix : str
        Suffix appended after ``{recorderId}/`` when publishing commands
        (default "control/").
    on_log : Callable[[str], None] | None
        Called from the paho network thread with human-readable log lines.
    on_device_update : Callable[[DeviceStatus], None] | None
        Called whenever a device status packet is received or updated.
    on_connected : Callable[[], None] | None
        Called once the broker connection is established.
    on_disconnected : Callable[[], None] | None
        Called when the connection drops.
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int = DEFAULT_BROKER_PORT,
        status_topic: str = DEFAULT_STATUS_TOPIC,
        command_topic_suffix: str = DEFAULT_COMMAND_TOPIC_SUFFIX,
        on_log: Callable[[str], None] | None = None,
        on_device_update: Callable[[DeviceStatus], None] | None = None,
        on_connected: Callable[[], None] | None = None,
        on_disconnected: Callable[[], None] | None = None,
    ) -> None:
        self._broker_host = broker_host.strip()
        self._broker_port = broker_port
        self._status_topic = status_topic
        self._command_topic_suffix = command_topic_suffix
        self._on_log = on_log
        self._on_device_update = on_device_update
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected

        self._devices: dict[str, DeviceStatus] = {}
        self._devices_lock = threading.Lock()
        self._connected = False
        self._client = None

    # ── Public properties ────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def discovered_devices(self) -> list[DeviceStatus]:
        with self._devices_lock:
            return list(self._devices.values())

    # ── Connection management ────────────────────────────────────────────

    def connect(self) -> None:
        """Connect to the broker and start the background network loop."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "paho-mqtt is required for the MQTT controller. "
                "Install with: pip install 'nucleuskit-pipeline[gui]'"
            ) from exc

        self._log(f"Connecting to broker {self._broker_host}:{self._broker_port} …")

        client = mqtt.Client(
            client_id=f"nucleuskit-{int(time.time())}",
            protocol=mqtt.MQTTv311,
            clean_session=True,
        )
        client.on_connect = self._on_mqtt_connect
        client.on_disconnect = self._on_mqtt_disconnect
        client.on_message = self._on_mqtt_message
        client.on_log = self._on_mqtt_log

        client.connect_async(self._broker_host, self._broker_port, keepalive=60)
        client.loop_start()
        self._client = client

    def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""
        if self._client is None:
            return
        self._log("Disconnecting from broker …")
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as exc:
            self._log(f"[WARN] Error during disconnect: {exc}")
        self._client = None
        self._connected = False

    # ── Command publishing ───────────────────────────────────────────────

    def send_start_recording(self, device_id: str = TARGET_ALL) -> None:
        """Send START_RECORDING to one device or all discovered devices."""
        payload = json.dumps({"Type": "COMMAND", "Content": {"Code": "START_RECORDING"}})
        self._publish_to(device_id, payload, label="START_RECORDING")

    def send_stop_recording(self, device_id: str = TARGET_ALL) -> None:
        """Send STOP_RECORDING to one device or all discovered devices."""
        payload = json.dumps({"Type": "COMMAND", "Content": {"Code": "STOP_RECORDING"}})
        self._publish_to(device_id, payload, label="STOP_RECORDING")

    def send_tag(self, code: str, values: str = "", device_id: str = TARGET_ALL) -> None:
        """Inject an event marker into the active recording session."""
        payload = json.dumps({"Type": "TAG", "Content": {"Code": code, "Values": values}})
        self._publish_to(device_id, payload, label=f"TAG:{code}")

    # ── Internal helpers ─────────────────────────────────────────────────

    def _publish_to(self, device_id: str, payload: str, label: str) -> None:
        if not self._connected or self._client is None:
            self._log("[WARN] Not connected — command not sent.")
            return

        if device_id == TARGET_ALL:
            with self._devices_lock:
                targets = list(self._devices.keys())
            if not targets:
                self._log("[WARN] No devices discovered yet — command not sent.")
                return
            for did in targets:
                topic = f"{did}/{self._command_topic_suffix}"
                self._client.publish(topic, payload, qos=0)
                self._log(f"→ [{label}] → {topic}")
        else:
            topic = f"{device_id}/{self._command_topic_suffix}"
            self._client.publish(topic, payload, qos=0)
            self._log(f"→ [{label}] → {topic}")

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            self._on_log(msg)

    # ── paho callbacks (run on the paho network thread) ──────────────────

    def _on_mqtt_connect(self, client, userdata, flags, rc) -> None:  # type: ignore[no-untyped-def]
        import paho.mqtt.client as mqtt

        if rc == mqtt.CONNACK_ACCEPTED:
            self._connected = True
            self._log(f"✓ Connected to {self._broker_host}:{self._broker_port}")
            self._log(f"  Subscribing to status topic: {self._status_topic}")
            client.subscribe(self._status_topic, qos=0)
            self._log("  Ready — waiting for device status packets …")
            if self._on_connected:
                self._on_connected()
        else:
            reason = {
                1: "unacceptable protocol version",
                2: "identifier rejected",
                3: "server unavailable",
                4: "bad username or password",
                5: "not authorised",
            }.get(rc, f"unknown reason code {rc}")
            self._log(f"✗ Connection refused: {reason}")

    def _on_mqtt_disconnect(self, client, userdata, rc) -> None:  # type: ignore[no-untyped-def]
        was_connected = self._connected
        self._connected = False
        if rc == 0:
            self._log("Disconnected cleanly.")
        else:
            self._log(f"[WARN] Unexpected disconnect (rc={rc}) — paho will attempt reconnect …")
        if was_connected and self._on_disconnected:
            self._on_disconnected()

    def _on_mqtt_message(self, client, userdata, msg) -> None:  # type: ignore[no-untyped-def]
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
            device_id = str(data.get("id", "device"))
            recording = bool(data.get("recording", False))
            battery = int(data.get("battery", -1))

            with self._devices_lock:
                existing = self._devices.get(device_id)
                is_new = existing is None
                status = DeviceStatus(
                    device_id=device_id,
                    recording=recording,
                    battery=battery,
                    last_seen=time.time(),
                )
                self._devices[device_id] = status

            if is_new:
                self._log(
                    f"  ● New device discovered: '{device_id}'  "
                    f"(recording={recording}, battery={battery}%)"
                )

            if self._on_device_update:
                self._on_device_update(status)

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self._log(f"[WARN] Malformed status packet on {msg.topic}: {exc}")

    def _on_mqtt_log(self, client, userdata, level, buf) -> None:  # type: ignore[no-untyped-def]
        import paho.mqtt.client as mqtt

        if level in (mqtt.MQTT_LOG_DEBUG,):
            return
        prefix = {
            mqtt.MQTT_LOG_INFO: "[paho:INFO]",
            mqtt.MQTT_LOG_NOTICE: "[paho:NOTICE]",
            mqtt.MQTT_LOG_WARNING: "[paho:WARN]",
            mqtt.MQTT_LOG_ERR: "[paho:ERR]",
        }.get(level, "[paho]")
        self._log(f"{prefix} {buf}")
