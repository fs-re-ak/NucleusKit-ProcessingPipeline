"""Hermes EEG stream robustness: packet counter handling and the EEG worker, without a headset."""

from __future__ import annotations

import logging
import queue
import threading

import pytest

pytest.importorskip("bleak")

from nucleuskit_toolkit.hermes.realtime import proxy  # noqa: E402
from nucleuskit_toolkit.hermes.realtime.proxy import EegPacketAssembler, HermesBleProxy  # noqa: E402

PAYLOAD = bytes(24 * 10)  # 10 samples of 8 channels
REAL_LOG_PATH = proxy.ble_log_path


@pytest.fixture(autouse=True)
def _log_to_tmp(tmp_path, monkeypatch):
    """Write the BLE log under tmp_path instead of the user's app-data folder."""
    for handler in list(proxy._log.handlers):
        proxy._log.removeHandler(handler)
    monkeypatch.setattr(proxy, "ble_log_path", lambda: str(tmp_path / "logs" / "hermes_ble.log"))
    yield
    for handler in list(proxy._log.handlers):
        handler.close()
        proxy._log.removeHandler(handler)


def _assembler():
    out: list[tuple] = []
    return EegPacketAssembler(out.append), out


def _packet(counter: int) -> bytes:
    return bytes([counter]) + PAYLOAD


def test_in_order_packets_pass_through_including_wrap() -> None:
    a, out = _assembler()
    for c in [125, 126, 127, 0, 1]:
        assert a.push(_packet(c))
    assert [(r, n) for r, _, n in out] == [(True, 125), (True, 126), (True, 127), (True, 0), (True, 1)]


def test_gap_is_filled_once_enough_newer_packets_arrive() -> None:
    a, out = _assembler()
    a.push(_packet(10))
    a.push(_packet(13))  # 11 and 12 missing
    assert [n for _, _, n in out] == [10]  # held back, waiting for late packets
    for c in range(14, 14 + EegPacketAssembler.MAX_WAIT):
        a.push(_packet(c))
    numbers = [(r, n) for r, _, n in out]
    assert numbers[:4] == [(True, 10), (False, 11), (False, 12), (True, 13)]
    assert a.dropped == 2


@pytest.mark.parametrize("bad", [128, 200, 255])
def test_out_of_range_counter_is_ignored_without_hanging(bad: int) -> None:
    a, out = _assembler()
    a.push(_packet(5))
    done = threading.Event()

    def push_bad() -> None:
        a.push(_packet(bad))
        done.set()

    threading.Thread(target=push_bad, daemon=True).start()
    assert done.wait(2.0), "an invalid counter must not loop forever"
    assert a.ignored == 1 and a.last_packet == 5
    a.push(_packet(6))
    assert [n for _, _, n in out] == [5, 6]


def test_repeated_counter_is_ignored_instead_of_inventing_127_missing_packets() -> None:
    a, out = _assembler()
    a.push(_packet(40))
    assert not a.push(_packet(40))
    a.push(_packet(41))
    assert [(r, n) for r, _, n in out] == [(True, 40), (True, 41)]
    assert a.dropped == 0


def test_large_jump_resynchronises() -> None:
    a, out = _assembler()
    a.push(_packet(10))
    a.push(_packet(9))  # 127 "missing" packets if trusted: an out-of-order or stale counter
    assert a.resyncs == 1 and a.dropped == 0
    a.push(_packet(10))
    assert [(r, n) for r, _, n in out] == [(True, 10), (True, 9), (True, 10)]


def test_reset_forgets_previous_stream() -> None:
    a, out = _assembler()
    a.push(_packet(100))
    a.push(_packet(103))
    a.reset()
    a.push(_packet(3))  # first packet after a reconnection: no gap is computed against 103
    assert a.last_packet == 3 and a.dropped == 0


def test_eeg_worker_survives_a_failing_callback() -> None:
    q: queue.Queue = queue.Queue()
    received: list[int] = []
    calls = {"n": 0}

    def callback(samples: list) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("I/O operation on closed file.")
        received.append(len(samples))

    q.put((True, PAYLOAD, 1))
    q.put((True, PAYLOAD, 2))
    q.put((False, None, 3))
    q.put((True, b"\x00" * 5, 4))  # no complete sample: skipped
    q.put((True, PAYLOAD, 5))
    q.put(None)
    worker = threading.Thread(target=HermesBleProxy.worker_process, args=(q, callback), daemon=True)
    worker.start()
    worker.join(5.0)
    assert not worker.is_alive()
    assert received == [10, 10, 10]  # packets 2, 3 (NaN) and 5 after the error on packet 1


def test_messages_are_written_to_the_ble_log(tmp_path) -> None:
    a, _ = _assembler()
    a.push(_packet(1))
    a.push(_packet(250))
    for handler in proxy._log.handlers:
        handler.flush()
    text = (tmp_path / "logs" / "hermes_ble.log").read_text(encoding="utf-8")
    assert "invalid counter 250" in text and "WARNING" in text


def test_log_path_is_in_the_user_folder(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    path = REAL_LOG_PATH()
    assert path.endswith("hermes_ble.log") and "NucleusKitToolkit" in path


def test_decode_keeps_complete_samples(caplog) -> None:
    caplog.set_level(logging.WARNING)
    samples = HermesBleProxy.decode_eeg_payload(PAYLOAD + b"\x01\x02", 7)
    assert len(samples) == 10 and all(len(s) == 8 for s in samples)
