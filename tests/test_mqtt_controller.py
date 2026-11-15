"""MQTT controller: broker addresses, experiment-scoped discovery and per-device control topics (contract v1)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from nucleuskit_toolkit.mqtt.mqtt_controller import MqttController, parse_broker


def test_parse_broker_urls_and_bare_hosts():
    assert parse_broker("192.168.1.10") == parse_broker("mqtt://192.168.1.10:1883")
    assert parse_broker("192.168.1.10", fallback_port=1884).port == 1884
    tls = parse_broker("mqtts://broker.example")
    assert tls.port == 8883 and tls.tls and not tls.websocket
    wss = parse_broker("wss://broker.example/live")
    assert wss.port == 443 and wss.path == "/live" and wss.websocket
    assert parse_broker("ws://h:9001").path == "/mqtt"
    with pytest.raises(ValueError):
        parse_broker("http://host")
    with pytest.raises(ValueError):
        parse_broker("")


def _message(controller: MqttController, topic: str, payload: dict) -> None:
    controller._on_mqtt_message(None, None, SimpleNamespace(topic=topic, payload=json.dumps(payload).encode()))


def test_experiment_discovery_and_control_topics():
    c = MqttController("mqtts://broker.example", experiment_id="exp-1")
    assert c.status_subscriptions() == ["exp-1/+/status", "vizia/status"]
    _message(c, "exp-1/rec-01/status", {"v": 1, "id": "rec-01", "recording": True, "battery": 80, "sessionId": "s1"})
    _message(c, "vizia/status", {"id": "old-phone", "recording": False, "battery": 50})
    devices = {d.device_id: d for d in c.discovered_devices}
    assert devices["rec-01"].experiment_id == "exp-1" and devices["rec-01"].session_id == "s1"
    assert c.control_topic("rec-01") == "exp-1/rec-01/control"
    assert c.control_topic("old-phone") == "old-phone/control/"
    assert c.control_topic("never-seen") == "exp-1/never-seen/control"


def test_status_id_must_match_topic():
    c = MqttController("127.0.0.1", experiment_id="exp-1")
    _message(c, "exp-1/rec-01/status", {"id": "someone-else", "recording": True})
    assert c.discovered_devices == []


def test_legacy_only_controller_is_unchanged():
    c = MqttController("127.0.0.1")
    assert c.status_subscriptions() == ["vizia/status"]
    _message(c, "vizia/status", {"id": "rec-02", "recording": True, "battery": 70})
    assert c.control_topic("rec-02") == "rec-02/control/"


def test_publish_goes_to_each_device_topic():
    c = MqttController("127.0.0.1", experiment_id="exp-1")
    _message(c, "exp-1/a/status", {"id": "a", "recording": False})
    _message(c, "vizia/status", {"id": "b", "recording": False})
    sent = []
    c._client = SimpleNamespace(publish=lambda topic, payload, qos: sent.append(topic))
    c._connected = True
    c.send_start_recording()
    assert sorted(sent) == ["b/control/", "exp-1/a/control"]
