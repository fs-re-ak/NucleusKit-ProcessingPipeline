"""Live Hermes BLE scan: devices are reported as soon as they are seen, once each, and the scan can be stopped."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
bleak = pytest.importorskip("bleak")

from nucleuskit_toolkit.hermes.realtime.proxy import HERMES_NAME_SUBSTRING  # noqa: E402
from nucleuskit_toolkit.ui.pages import realtime_viewer_page as page  # noqa: E402


class FakeScanner:
    """Advertises a Hermes headset twice, another device, then a second headset."""

    def __init__(self, detection_callback) -> None:  # type: ignore[no-untyped-def]
        self._cb = detection_callback
        self.stopped = False

    async def start(self) -> None:
        hermes = f"{HERMES_NAME_SUBSTRING}-01"
        self._cb(SimpleNamespace(name=hermes, address="AA:01"), SimpleNamespace(local_name=hermes))
        self._cb(SimpleNamespace(name=hermes, address="AA:01"), SimpleNamespace(local_name=hermes))
        self._cb(SimpleNamespace(name="Phone", address="BB:02"), SimpleNamespace(local_name="Phone"))
        self._cb(SimpleNamespace(name=None, address="AA:03"), SimpleNamespace(local_name=f"{HERMES_NAME_SUBSTRING}-03"))

    async def stop(self) -> None:
        self.stopped = True


def test_scan_reports_each_hermes_device_once(monkeypatch) -> None:
    monkeypatch.setattr(bleak, "BleakScanner", FakeScanner)
    found: list[tuple[str, str]] = []
    count = page._bleak_scan_hermes_devices(0.3, lambda n, a: found.append((n, a)), threading.Event())
    assert count == 2
    assert [a for _, a in found] == ["AA:01", "AA:03"]


def test_scan_stops_early(monkeypatch) -> None:
    monkeypatch.setattr(bleak, "BleakScanner", FakeScanner)
    stop = threading.Event()
    stop.set()
    t0 = time.monotonic()
    page._bleak_scan_hermes_devices(30.0, lambda n, a: None, stop)
    assert time.monotonic() - t0 < 2.0
