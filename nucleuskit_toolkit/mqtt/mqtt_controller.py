"""
MQTT controller for Vizia Mobile Android devices.

Connects to a user-configured MQTT broker and drives recording sessions via
the Vizia Mobile MQTT interface (see doc/mqtt_interface.md; source of truth:
viziaMobile2/MQTT_INTERFACE.md, contract v1).

Devices are discovered on ``{experimentId}/+/status`` when an experiment id is given, and on the legacy
shared ``vizia/status`` topic (phones without an experiment, local brokers only). Commands go to each
device's own control topic: ``{experimentId}/{recorderId}/control`` or, legacy, ``{recorderId}/control/``.
The broker address may be a URL (``mqtt://``, ``mqtts://``, ``ws://``, ``wss://``) for TLS / WebSocket.

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
from urllib.parse import urlparse

DEFAULT_BROKER_PORT = 1883
DEFAULT_STATUS_TOPIC = "vizia/status"
DEFAULT_COMMAND_TOPIC_SUFFIX = "control/"

TARGET_ALL = "__ALL__"

_DEFAULT_PORTS = {"mqtt": 1883, "mqtts": 8883, "ws": 80, "wss": 443}


@dataclass(frozen=True)
class BrokerEndpoint:
    scheme: str
    host: str
    port: int
    path: str | None = None

    @property
    def tls(self) -> bool:
        return self.scheme in ("mqtts", "wss")

    @property
    def websocket(self) -> bool:
        return self.scheme in ("ws", "wss")


def parse_broker(address: str, fallback_port: int = DEFAULT_BROKER_PORT) -> BrokerEndpoint:
    """Parse a broker URL, or a bare host / IP (plain TCP on ``fallback_port``)."""
    text = address.strip()
    if "://" not in text:
        host, _, port = text.partition(":")
        if not host:
            raise ValueError("broker host required")
        return BrokerEndpoint("mqtt", host, int(port) if port else fallback_port)
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not parsed.hostname:
        raise ValueError(f"unsupported broker address: {address!r}")
    path = None
    if scheme in ("ws", "wss"):
        path = parsed.path if parsed.path not in ("", "/") else "/mqtt"
    return BrokerEndpoint(scheme, parsed.hostname, parsed.port or _DEFAULT_PORTS[scheme], path)


@dataclass
class DeviceStatus:
    device_id: str
    recording: bool = False
    battery: int = -1
    last_seen: float = field(default_factory=time.time)
    experiment_id: str | None = None
    """Experiment of the device's topics; None for legacy topics."""
    control_topic: str = ""
    session_id: str | None = None
    headset_connected: bool | None = None


class MqttController:
    """
    Manages a single MQTT connection to a Vizia Mobile broker.

    Parameters
    ----------
    broker_host : str
        Broker URL (``mqtt://``, ``mqtts://``, ``ws://``, ``wss://``) or bare IP / hostname.
    broker_port : int
        TCP port for a bare IP / hostname (default 1883).
    status_topic : str
        Legacy shared status topic (default "vizia/status"); empty to listen only to the experiment.
    command_topic_suffix : str
        Legacy suffix appended after ``{recorderId}/`` when publishing commands
        (default "control/").
    experiment_id : str | None
        Experiment whose devices to discover on ``{experimentId}/+/status``.
    username, password : str | None
        Broker login (per-device or per-controller account).
    ca_file : str | None
        CA certificate for TLS (the RE-AK private CA); None = system CAs.
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
        experiment_id: str | None = None,
        username: str | None = None,
        password: str | None = None,
        ca_file: str | None = None,
        on_log: Callable[[str], None] | None = None,
        on_device_update: Callable[[DeviceStatus], None] | None = None,
        on_connected: Callable[[], None] | None = None,
        on_disconnected: Callable[[], None] | None = None,
    ) -> None:
        self._endpoint = parse_broker(broker_host, broker_port)
        self._broker_host = self._endpoint.host
        self._broker_port = self._endpoint.port
        self._status_topic = status_topic.strip()
        self._command_topic_suffix = command_topic_suffix
        self._experiment_id = (experiment_id or "").strip() or None
        self._username = username or None
        self._password = password or None
        self._ca_file = ca_file or None
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
                "Install with: pip install 'nucleuskit-toolkit[gui]'"
            ) from exc

        ep = self._endpoint
        self._log(f"Connecting to broker {ep.scheme}://{ep.host}:{ep.port} …")

        client = mqtt.Client(
            client_id=f"nucleuskit-{int(time.time())}",
            protocol=mqtt.MQTTv311,
            clean_session=True,
            transport="websockets" if ep.websocket else "tcp",
        )
        if ep.websocket:
            client.ws_set_options(path=ep.path or "/mqtt")
        if ep.tls:
            client.tls_set(ca_certs=self._ca_file)
        if self._username:
            client.username_pw_set(self._username, self._password)
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
                topic = self.control_topic(did)
                self._client.publish(topic, payload, qos=0)
                self._log(f"→ [{label}] → {topic}")
        else:
            topic = self.control_topic(device_id)
            self._client.publish(topic, payload, qos=0)
            self._log(f"→ [{label}] → {topic}")

    def status_subscriptions(self) -> list[str]:
        """Topics on which device status packets are received."""
        topics = []
        if self._experiment_id:
            topics.append(f"{self._experiment_id}/+/status")
        if self._status_topic:
            topics.append(self._status_topic)
        return topics

    def control_topic(self, device_id: str) -> str:
        """Control topic of a discovered device (experiment-scoped or legacy)."""
        with self._devices_lock:
            known = self._devices.get(device_id)
        if known is not None and known.control_topic:
            return known.control_topic
        if self._experiment_id:
            return f"{self._experiment_id}/{device_id}/control"
        return f"{device_id}/{self._command_topic_suffix}"

    def _parse_status(self, topic: str, payload: bytes) -> DeviceStatus:
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("status payload is not a JSON object")
        parts = topic.split("/")
        if self._experiment_id and len(parts) == 3 and parts[0] == self._experiment_id and parts[2] == "status":
            device_id = parts[1]
            if data.get("id") not in (None, device_id):
                raise ValueError(f"id {data.get('id')!r} does not match topic recorder {device_id!r}")
            experiment = self._experiment_id
            control = f"{experiment}/{device_id}/control"
        else:
            device_id = str(data.get("id", "device"))
            experiment = None
            control = f"{device_id}/{self._command_topic_suffix}"
        return DeviceStatus(
            device_id=device_id,
            recording=bool(data.get("recording", False)),
            battery=int(data.get("battery", -1)),
            last_seen=time.time(),
            experiment_id=experiment,
            control_topic=control,
            session_id=data.get("sessionId"),
            headset_connected=data.get("headsetConnected"),
        )

    def _log(self, msg: str) -> None:
        if self._on_log is not None:
            self._on_log(msg)

    # ── paho callbacks (run on the paho network thread) ──────────────────

    def _on_mqtt_connect(self, client, userdata, flags, rc) -> None:  # type: ignore[no-untyped-def]
        import paho.mqtt.client as mqtt

        if rc == mqtt.CONNACK_ACCEPTED:
            self._connected = True
            self._log(f"✓ Connected to {self._broker_host}:{self._broker_port}")
            for topic in self.status_subscriptions():
                self._log(f"  Subscribing to status topic: {topic}")
                client.subscribe(topic, qos=0)
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
            status = self._parse_status(msg.topic, msg.payload)
            device_id, recording, battery = status.device_id, status.recording, status.battery

            with self._devices_lock:
                existing = self._devices.get(device_id)
                is_new = existing is None
                self._devices[device_id] = status

            if is_new:
                self._log(
                    f"  ● New device discovered: '{device_id}'  "
                    f"(recording={recording}, battery={battery}%)"
                )

            if self._on_device_update:
                self._on_device_update(status)

        except (json.JSONDecodeError, KeyError, ValueError, TypeError, UnicodeDecodeError) as exc:
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
