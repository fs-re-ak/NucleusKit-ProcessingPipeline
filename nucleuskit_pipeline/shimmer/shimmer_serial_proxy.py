"""
Shimmer3 LogAndStream client over Bluetooth Classic.

Supports two connection modes:
  - COM port  (e.g. "COM7")     — uses pyserial, classic approach.
  - MAC address (e.g. "00:06:66:B7:00:00") — opens a direct Bluetooth RFCOMM
    socket, bypassing COM-port selection entirely.  This is more reliable on
    Windows where two "Standard Serial over Bluetooth link" ports are created
    per device (incoming / outgoing) and only one can be opened by applications.

Configures streaming at ~51.2 Hz and invokes a callback with each decoded row:
  [epoch_ms, imu_x, imu_y, imu_z, eda_kohm, ppg_mv]  (host epoch milliseconds).
"""

from __future__ import annotations

import math
import re
import struct
import sys
import threading
from collections.abc import Callable
from time import sleep, time
from typing import Any

from nucleuskit_pipeline.hermes.HermesConstants import ShimmerConstants

ShimmerSampleCallback = Callable[[list[float]], None]

ACK_BYTE = b"\xff"
FRAME_SIZE = 14  # 1 type + 3 ts + 5x uint16 payload (accel xyz, PPG, GSR packed)

_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}$")


class _BtSocketAdapter:
    """Wraps a Bluetooth RFCOMM socket with a serial-like read/write interface.

    ``is_open`` starts True and is set to False on ``close()``.  ``read(n)``
    accumulates bytes until *n* bytes arrive or the per-call socket timeout
    elapses, matching the behaviour of ``serial.Serial.read(n, timeout=…)``.
    """

    def __init__(self, sock: Any) -> None:
        self._sock = sock
        self.is_open = True

    def reset_input_buffer(self) -> None:
        """Discard any data already in the receive buffer."""
        self._sock.settimeout(0.05)
        try:
            while True:
                data = self._sock.recv(4096)
                if not data:
                    break
        except OSError:
            pass
        self._sock.settimeout(0.2)

    def read(self, n: int) -> bytes:
        """Return up to *n* bytes; returns fewer (or b'') on timeout / error."""
        result = b""
        while len(result) < n:
            try:
                chunk = self._sock.recv(n - len(result))
                if not chunk:
                    break
                result += chunk
            except OSError:
                break
        return result

    def write(self, data: bytes) -> None:
        self._sock.sendall(data)

    def close(self) -> None:
        self.is_open = False
        try:
            self._sock.close()
        except OSError:
            pass


class ShimmerSerialProxy:
    """Background thread: connect to Shimmer3, send setup, stream packets until disconnect.

    ``port_or_mac`` may be:

    * A serial port name  — ``"COM7"``, ``"/dev/rfcomm0"`` etc.
    * A Bluetooth MAC address — ``"00:06:66:B7:00:00"`` (colons or hyphens).
      A direct RFCOMM socket is opened; no virtual COM port is needed.

    Pass the result of :meth:`is_mac_address` to determine which mode will be
    used before constructing the proxy.
    """

    def __init__(self, port_or_mac: str, sample_callback: ShimmerSampleCallback) -> None:
        self._target = port_or_mac
        self._sample_callback = sample_callback
        self.is_connected = False
        self._conn: Any = None  # serial.Serial or _BtSocketAdapter
        self._stop = threading.Event()
        self._connection_error: str | None = None
        self._worker = threading.Thread(target=self._run, name="ShimmerSerial", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_mac_address(s: str) -> bool:
        """Return True if *s* looks like a Bluetooth MAC address."""
        return bool(_MAC_RE.match(s.strip()))

    def wait_until_connected(self, timeout: float = 90.0) -> None:
        deadline = time() + timeout
        while time() < deadline:
            if self.is_connected:
                return
            if self._connection_error is not None:
                raise RuntimeError(self._connection_error)
            sleep(0.05)
        raise TimeoutError("Shimmer connection timed out")

    def disconnect(self) -> None:
        self._stop.set()
        conn = self._conn
        if conn is not None and conn.is_open:
            try:
                conn.write(struct.pack("B", 0x20))
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        self._worker.join(timeout=5.0)
        self.is_connected = False
        self._conn = None

    # ------------------------------------------------------------------
    # Internal: connection
    # ------------------------------------------------------------------

    @staticmethod
    def _open_bt_socket(mac: str) -> _BtSocketAdapter:
        """Open a direct Bluetooth RFCOMM socket to *mac* (channel 1)."""
        import socket as _socket

        mac = mac.upper().replace("-", ":")

        if sys.platform == "win32":
            # AF_BTH = 32, BTPROTO_RFCOMM = 3  (Winsock2 constants)
            AF_BTH = getattr(_socket, "AF_BTH", 32)
            BTPROTO_RFCOMM = getattr(_socket, "BTPROTO_RFCOMM", 3)
            sock = _socket.socket(AF_BTH, _socket.SOCK_STREAM, BTPROTO_RFCOMM)
        else:
            sock = _socket.socket(
                _socket.AF_BLUETOOTH, _socket.SOCK_STREAM, _socket.BTPROTO_RFCOMM
            )

        sock.settimeout(12.0)
        sock.connect((mac, 1))  # RFCOMM channel 1
        sock.settimeout(0.2)
        return _BtSocketAdapter(sock)

    # ------------------------------------------------------------------
    # Internal: Shimmer protocol
    # ------------------------------------------------------------------

    def _wait_for_ack(self, conn: Any, timeout_s: float = 8.0) -> None:
        deadline = time() + timeout_s
        while time() < deadline:
            if self._stop.is_set():
                raise InterruptedError("Shimmer disconnect requested")
            b = conn.read(1)
            if b == ACK_BYTE:
                return
        raise TimeoutError("Timed out waiting for Shimmer ACK (0xFF)")

    def _configure_streaming(self, conn: Any) -> None:
        # SET_SENSORS: accel + GSR + PPG (same bitmask as legacy stack)
        conn.write(struct.pack("BBBB", 0x08, 0x84, 0x01, 0x00))
        self._wait_for_ack(conn)

        conn.write(struct.pack("BB", 0x5E, 0x01))
        self._wait_for_ack(conn)

        conn.write(struct.pack("B", 0x20))
        self._wait_for_ack(conn)

        clock_wait = int(math.ceil((2 << 14) / ShimmerConstants.SAMPLING_RATE))
        conn.write(struct.pack("<BH", 0x05, clock_wait))
        self._wait_for_ack(conn)

        conn.write(struct.pack("B", 0x01))
        self._wait_for_ack(conn)

        response_size = 9
        inquiry = conn.read(response_size)
        if len(inquiry) < response_size:
            raise TimeoutError("Shimmer inquiry response too short")

        num_channels = inquiry[7]
        channels = conn.read(num_channels)
        if len(channels) < num_channels:
            raise TimeoutError("Shimmer channel map too short")

        conn.write(struct.pack("B", 0x07))
        self._wait_for_ack(conn)

    def _decode_frame(self, data: bytes) -> list[float] | None:
        if len(data) != FRAME_SIZE:
            return None
        # Native uint16 order matches legacy Shimmer streaming clients on Windows/Linux.
        x, y, z, ppg_raw, gsr_raw = struct.unpack("HHHHH", data[4:FRAME_SIZE])

        data_range = (gsr_raw >> 14) & 0x03
        rf = (40.2, 287.0, 1000.0, 3300.0)[data_range]
        gsr_to_volts = (gsr_raw & 0x3FFF) * (3.0 / 4095.0)
        den = (gsr_to_volts / 0.5) - 1.0
        if abs(den) < 1e-9:
            gsr_kohm = float("nan")
        else:
            gsr_kohm = rf / den

        ppg_mv = ppg_raw * (3000.0 / 4095.0)
        epoch_ms = round(time() * 1000.0)
        return [float(epoch_ms), float(x), float(y), float(z), float(gsr_kohm), float(ppg_mv)]

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            target = self._target.strip()
            if self.is_mac_address(target):
                conn: Any = self._open_bt_socket(target)
            else:
                import serial
                conn = serial.Serial(target, baudrate=115200, timeout=0.2)

            self._conn = conn
            conn.reset_input_buffer()
            self._configure_streaming(conn)
            self.is_connected = True

            buf = b""
            while not self._stop.is_set():
                chunk = conn.read(FRAME_SIZE)
                if not chunk:
                    continue
                buf += chunk
                while len(buf) >= FRAME_SIZE and not self._stop.is_set():
                    frame = buf[:FRAME_SIZE]
                    buf = buf[FRAME_SIZE:]
                    row = self._decode_frame(frame)
                    if row is not None:
                        self._sample_callback(row)

        except Exception as e:
            self._connection_error = str(e)
        finally:
            self.is_connected = False
            conn = self._conn
            if conn is not None:
                if conn.is_open:
                    try:
                        conn.write(struct.pack("B", 0x20))
                        self._wait_for_ack(conn, timeout_s=2.0)
                    except Exception:
                        pass
                try:
                    conn.close()
                except OSError:
                    pass
                self._conn = None
