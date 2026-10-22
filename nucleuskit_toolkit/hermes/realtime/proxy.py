"""
Hermes V1 BLE client: EEG + motion notifications.

Runs asyncio on a dedicated thread; decodes EEG on worker threads via queues.

Stability fixes applied in 2026-10-NewEmotionsModel:
  - Bug 1: asyncio.Event objects are created inside _run_async_main so they are
    bound to the correct event loop (not the main-thread loop which may not exist
    on Windows).  trigger_shutdown uses loop.call_soon_threadsafe to cross the
    thread boundary safely.
  - Bug 2: The EEG notification callback now dispatches via
    loop.call_soon_threadsafe instead of asyncio.create_task, which is not
    thread-safe when called from a non-loop OS thread (common on Windows BLE).
  - Bug 3: BleakClient receives a disconnected_callback so unexpected drops are
    detected and recorded immediately.
  - Bug 4: main_task contains an automatic reconnection loop with exponential
    back-off (1 s → 2 s → … capped at 30 s, up to MAX_RECONNECTS attempts).

Stream fixes applied in 2026-10-fix-eeg-stream:
  - The EEG worker logs an error and keeps going: one failing packet or callback
    (for example a recording file closed while a packet is written) no longer
    stops the EEG stream for the rest of the connection.
  - Packet counters are validated (7-bit, 0-127): a counter out of range or
    repeated is ignored, and a jump of more than half the counter range
    resynchronises instead of inventing up to 127 missing packets. An invalid
    counter used to loop forever on the event-loop thread, freezing the
    connection.
  - All packet bookkeeping is reset on (re)connection.
  - BLE messages are also written to a rotating log file (ble_log_path()), so
    a drop leaves a trace when the app runs without a console.
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import logging
import os
import queue
import struct
import sys
import threading
from logging.handlers import RotatingFileHandler
from time import sleep, time
from typing import Callable

from bleak import BleakClient

ACC_SENS = 0.061 / 1000   # 0.061 mg/LSB → g
GYRO_SENS = 8.75 / 1000   # 8.75 mdps/LSB → dps
MAG_SENS = 0.14 / 1000    # 0.14 mgauss/LSB → gauss

EEG_DATA_UUID   = "9fa480e1-4967-11e5-a151-0002a5d5c51b"
EEG_CONFIG_UUID = "9fa480e2-4967-11e5-a151-0002a5d5c51b"
EVENT_UUID      = "9fa48301-4967-11e5-a151-0002a5d5c51b"
MOTION_UUID     = "9fa48201-4967-11e5-a151-0002a5d5c51b"

HERMES_NAME_SUBSTRING = "Hermes V1"

EegCallback    = Callable[[list], None] | None
MotionCallback = Callable[[tuple], None] | None
StatusCallback = Callable[[str], None] | None

BLE_LOG_NAME = "hermes_ble.log"
_BLE_LOG_MAX_BYTES = 2_000_000
_BLE_LOG_BACKUPS = 3

_log = logging.getLogger("nucleuskit_toolkit.ble")
_log.propagate = False  # printed to the console by _report; the file handler only writes the file
_log_lock = threading.Lock()


def ble_log_path() -> str:
    """Rotating BLE log: ``%LOCALAPPDATA%/REAK/NucleusKitToolkit/logs`` on Windows, the user log folder elsewhere."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Logs")
    else:
        base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(base, "REAK", "NucleusKitToolkit", "logs", BLE_LOG_NAME)


def _ensure_file_log() -> None:
    with _log_lock:
        if any(isinstance(h, RotatingFileHandler) for h in _log.handlers):
            return
        try:
            path = ble_log_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handler = RotatingFileHandler(
                path, maxBytes=_BLE_LOG_MAX_BYTES, backupCount=_BLE_LOG_BACKUPS, encoding="utf-8"
            )
        except OSError:
            return  # logging must never break streaming
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s"))
        _log.addHandler(handler)
        _log.setLevel(logging.INFO)


def _report(message: str, level: int = logging.INFO, exc_info: bool = False) -> None:
    """Print a BLE message to the console (and the GUI log) and write it to the BLE log file."""
    print(message)
    _ensure_file_log()
    _log.log(level, message, exc_info=exc_info)


class EegPacketAssembler:
    """Orders EEG notifications by their 7-bit packet counter and fills the gaps.

    Each notification starts with a counter (0-127, wrapping). A missing packet is
    emitted as ``(False, None, counter)`` once ``MAX_WAIT`` newer packets arrived,
    so the decoder can insert NaN samples; received packets are emitted as
    ``(True, payload, counter)``, in order.
    """

    MODULO = 128
    MAX_WAIT = 10
    MAX_GAP = MODULO // 2  # a larger jump means the counter cannot be trusted: resynchronise

    def __init__(self, emit: Callable[[tuple], None]) -> None:
        self._emit = emit
        self.reset()

    def reset(self) -> None:
        self.last_packet: int | None = None
        self.packets: collections.deque[int] = collections.deque()
        self.packet_received: collections.deque[bool] = collections.deque()
        self.samples_per_packets: collections.deque[bytes | None] = collections.deque()
        self.ignored = 0
        self.dropped = 0
        self.resyncs = 0

    @classmethod
    def missing_between(cls, last_packet: int | None, current_packet: int) -> list[int]:
        """Counters strictly between ``last_packet`` and ``current_packet`` (wrapping at 128)."""
        if last_packet is None:
            return []
        gap = (current_packet - last_packet) % cls.MODULO
        return [(last_packet + k) % cls.MODULO for k in range(1, gap)]

    def push(self, data: bytes | bytearray) -> bool:
        """Add one notification; returns False when it is ignored."""
        if len(data) < 1:
            self.ignored += 1
            _report("[HermesBleProxy] Ignoring an empty EEG notification.", logging.WARNING)
            return False
        current_packet = data[0]
        payload = bytes(data[1:])
        if not 0 <= current_packet < self.MODULO:
            self.ignored += 1
            _report(f"[HermesBleProxy] Ignoring EEG packet with invalid counter {current_packet}.", logging.WARNING)
            return False
        if current_packet == self.last_packet:
            self.ignored += 1
            _report(f"[HermesBleProxy] Ignoring repeated EEG packet {current_packet}.", logging.WARNING)
            return False

        missing = self.missing_between(self.last_packet, current_packet)
        if len(missing) >= self.MAX_GAP:
            self.resyncs += 1
            _report(
                f"[HermesBleProxy] EEG packet counter jumped from {self.last_packet} to {current_packet}: "
                "resynchronising.",
                logging.WARNING,
            )
            self._flush_all()
            missing = []
        elif missing:
            self.dropped += len(missing)
            span = str(missing[0]) if len(missing) == 1 else f"{missing[0]}-{missing[-1]}"
            _report(f"[HermesBleProxy] Dropped {len(missing)} EEG packet(s): {span}.", logging.WARNING)

        for packet in missing:
            self.packets.append(packet)
            self.packet_received.append(False)
            self.samples_per_packets.append(None)
        self.packets.append(current_packet)
        self.packet_received.append(True)
        self.samples_per_packets.append(payload)
        self.last_packet = current_packet
        self._transfer()
        return True

    def _transfer(self) -> None:
        while self.packet_received:
            delay = (self.last_packet - self.packets[0]) % self.MODULO
            if not self.packet_received[0] and delay < self.MAX_WAIT:
                break
            self._emit_head()

    def _flush_all(self) -> None:
        while self.packet_received:
            self._emit_head()

    def _emit_head(self) -> None:
        item = (self.packet_received.popleft(), self.samples_per_packets.popleft(), self.packets.popleft())
        self._emit(item)


class HermesBleProxy:
    MAX_RECONNECTS        = 10
    _RECONNECT_BASE_DELAY = 1.0   # seconds
    _RECONNECT_MAX_DELAY  = 30.0  # seconds

    def __init__(
        self,
        mac_address: str,
        eeg_callback: EegCallback = None,
        motion_callback: MotionCallback = None,
        status_callback: StatusCallback = None,
    ) -> None:
        self.is_connected = False
        self.client: BleakClient | None = None
        self.mac_address = mac_address

        # Bounded queues — drop oldest item rather than growing without bound.
        self.eeg_queue: queue.Queue    = queue.Queue(maxsize=256)
        self.motion_queue: queue.Queue = queue.Queue(maxsize=128)
        self._assembler = EegPacketAssembler(self._enqueue_eeg)

        # Bug 1 fix: asyncio.Event MUST be created inside the worker event loop.
        # Initialise to None here; _run_async_main populates them before any
        # coroutine runs.
        self.shutdown_event: asyncio.Event | None     = None
        self._disconnected_event: asyncio.Event | None = None
        self.loop: asyncio.AbstractEventLoop | None   = None

        self._connection_error: str | None = None
        self._status_callback = status_callback
        _report(f"[HermesBleProxy] Connecting to {mac_address} (BLE log: {ble_log_path()})")

        self._async_thread = threading.Thread(target=self._run_async_main, daemon=True, name="HermesBle")

        self.eeg_worker = threading.Thread(
            target=HermesBleProxy.worker_process,
            args=(self.eeg_queue, eeg_callback),
            daemon=True,
            name="HermesEeg",
        )
        self.eeg_worker.start()

        self.motion_worker = threading.Thread(
            target=HermesBleProxy.motion_process,
            args=(self.motion_queue, motion_callback),
            daemon=True,
            name="HermesMotion",
        )
        self.motion_worker.start()

        self._async_thread.start()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def wait_until_connected(self, timeout: float = 90.0) -> None:
        """Block until connected, or raise on failure/timeout."""
        deadline = time() + timeout
        while time() < deadline:
            if self.is_connected:
                return
            if self._connection_error is not None:
                raise RuntimeError(self._connection_error)
            if not self._async_thread.is_alive():
                err = self._connection_error or "BLE thread ended before connecting."
                raise RuntimeError(err)
            sleep(0.05)
        raise TimeoutError("Connection timed out.")

    def disconnect(self) -> None:
        try:
            self.trigger_shutdown()
            self._async_thread.join(timeout=30.0)
        finally:
            self.eeg_queue.put(None)
            self.motion_queue.put(None)
            self.eeg_worker.join(timeout=15.0)
            self.motion_worker.join(timeout=15.0)

    def trigger_shutdown(self) -> None:
        """Signal the async thread to shut down (thread-safe, Bug 1 fix)."""
        if self.loop is not None and self.shutdown_event is not None:
            self.loop.call_soon_threadsafe(self.shutdown_event.set)

    # ------------------------------------------------------------------ #
    # Async thread bootstrap                                               #
    # ------------------------------------------------------------------ #

    def _run_async_main(self) -> None:
        # Windows: Proactor loop in a non-main thread breaks many asyncio BLE
        # stacks; use SelectorEventLoop.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Bug 1 fix: create asyncio.Events inside the running loop.
        self.shutdown_event     = asyncio.Event()
        self._disconnected_event = asyncio.Event()

        try:
            self.loop.run_until_complete(self.main_task(self.mac_address))
        except Exception as e:
            self._connection_error = self._connection_error or str(e)
            _report(f"[HermesBleProxy] BLE thread stopped: {e}", logging.ERROR, exc_info=True)

    # ------------------------------------------------------------------ #
    # BLE callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_ble_disconnected(self, client: BleakClient) -> None:
        """Bug 3 fix: called by bleak when the connection drops unexpectedly.

        bleak schedules this callback inside the event loop, so setting an
        asyncio.Event directly is safe here.
        """
        self.is_connected = False
        a = self._assembler
        _report(
            "[HermesBleProxy] Connection dropped by device/OS "
            f"(EEG packets dropped: {a.dropped}, ignored: {a.ignored}, resyncs: {a.resyncs}).",
            logging.WARNING,
        )
        if self._disconnected_event is not None:
            self._disconnected_event.set()

    def _set_status(self, msg: str) -> None:
        """Forward a status string to the UI callback (thread-safe)."""
        if self._status_callback is not None:
            try:
                self._status_callback(msg)
            except Exception:
                pass

    def _enqueue_eeg(self, item: tuple) -> None:
        try:
            self.eeg_queue.put_nowait(item)
        except queue.Full:
            try:
                self.eeg_queue.get_nowait()
            except queue.Empty:
                pass
            self.eeg_queue.put_nowait(item)

    def _handle_eeg_data(self, _sender: object, data: bytearray) -> None:
        """Bug 2 fix: synchronous EEG notification handler.

        Called on the event-loop thread via loop.call_soon_threadsafe (see
        _subscribe_characteristics), so asyncio.create_task is not needed and
        no cross-thread hazard exists.
        """
        try:
            self._assembler.push(data)
        except Exception as e:
            _report(f"[HermesBleProxy] _handle_eeg_data: {e}", logging.ERROR, exc_info=True)

    async def motion_handler(self, _sender: object, data: bytearray) -> None:
        try:
            now = datetime.datetime.now()
            ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw = (
                struct.unpack_from("<hhhhhhhhh", data)
            )
            timestamp_epoch = now.timestamp()
            item = (
                timestamp_epoch,
                ax_raw, ay_raw, az_raw,
                gx_raw, gy_raw, gz_raw,
                cx_raw, cy_raw, cz_raw,
            )
            try:
                self.motion_queue.put_nowait(item)
            except queue.Full:
                try:
                    self.motion_queue.get_nowait()
                except queue.Empty:
                    pass
                self.motion_queue.put_nowait(item)
        except Exception as e:
            _report(f"[HermesBleProxy] motion_handler: {e}", logging.ERROR)

    async def config_handler(self, _sender: object, data: bytearray) -> None:
        try:
            _report(f"[HermesBleProxy] EEG config: {bytes(data).hex()}")
        except Exception as e:
            _report(f"[HermesBleProxy] config_handler: {e}", logging.ERROR)

    async def notification_handler(self, sender: object, data: bytearray) -> None:
        _report(f"Notification from {sender}: {data}")

    # ------------------------------------------------------------------ #
    # GATT subscribe / unsubscribe helpers                                 #
    # ------------------------------------------------------------------ #

    async def _subscribe_characteristics(self) -> None:
        """Subscribe to all four GATT characteristics."""
        assert self.loop is not None

        await self.client.start_notify(EVENT_UUID, self.notification_handler)
        _report("[HermesBleProxy] Subscribed to button notifications.")

        await self.client.start_notify(MOTION_UUID, self.motion_handler)
        _report("[HermesBleProxy] Subscribed to motion notifications.")

        await self.client.start_notify(EEG_CONFIG_UUID, self.config_handler)
        _report("[HermesBleProxy] Subscribed to EEG config notifications.")

        # Bug 2 fix: deliver the notification to _handle_eeg_data on the
        # event-loop thread via call_soon_threadsafe, avoiding the thread-unsafe
        # asyncio.create_task call that was here before.
        loop = self.loop
        await self.client.start_notify(
            EEG_DATA_UUID,
            lambda sender, data: loop.call_soon_threadsafe(
                self._handle_eeg_data, sender, data
            ),
        )
        _report("[HermesBleProxy] Subscribed to EEG data notifications.")

    async def _unsubscribe_characteristics(self) -> None:
        """Unsubscribe from all characteristics (best-effort; ignores errors)."""
        for uuid in (EEG_DATA_UUID, EVENT_UUID, MOTION_UUID, EEG_CONFIG_UUID):
            try:
                await self.client.stop_notify(uuid)
            except Exception:
                pass

    async def _stream_once(self) -> None:
        """Subscribe, then wait until shutdown is requested or connection drops."""
        self._disconnected_event.clear()
        await self._subscribe_characteristics()

        shutdown_fut    = asyncio.ensure_future(self.shutdown_event.wait())
        disconnect_fut  = asyncio.ensure_future(self._disconnected_event.wait())

        try:
            _done, pending = await asyncio.wait(
                [shutdown_fut, disconnect_fut],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        finally:
            # Unsubscribe best-effort (client may already be disconnected).
            if self.client is not None and self.client.is_connected:
                await self._unsubscribe_characteristics()

    # ------------------------------------------------------------------ #
    # Main async task                                                      #
    # ------------------------------------------------------------------ #

    async def main_task(
        self, device_address: str, device_name: str = HERMES_NAME_SUBSTRING
    ) -> None:
        """Connect, stream data, and auto-reconnect on unexpected drops (Bug 4 fix)."""

        # ── Initial connection (fails fast; no retry on first attempt) ──────
        self._assembler.reset()
        self.client = BleakClient(
            device_address,
            disconnected_callback=self._on_ble_disconnected,  # Bug 3 fix
        )
        try:
            await self.client.connect()
        except Exception as e:
            self._connection_error = str(e)
            self.is_connected = False
            _report(f"[HermesBleProxy] Initial connection failed: {e}", logging.ERROR)
            return

        _report(f"[HermesBleProxy] Connected to {device_name} [{device_address}]")
        self._connection_error = None
        self.is_connected = True
        session_start = time()

        # ── Stream + auto-reconnect loop (Bug 4 fix) ────────────────────────
        reconnect_num = 0

        while True:
            # Stream until shutdown or unexpected disconnect.
            if self.is_connected:
                try:
                    await self._stream_once()
                except Exception as e:
                    _report(f"[HermesBleProxy] Streaming error: {e}", logging.ERROR, exc_info=True)
                finally:
                    self.is_connected = False

            # Shutdown takes priority over reconnect.
            if self.shutdown_event.is_set():
                break

            # ── Unexpected disconnect: plan a reconnect attempt ─────────────
            reconnect_num += 1
            if reconnect_num > self.MAX_RECONNECTS:
                _report(
                    f"[HermesBleProxy] Max reconnects ({self.MAX_RECONNECTS}) reached. Stopping.",
                    logging.ERROR,
                )
                self._set_status(
                    "Could not reconnect. Please press Disconnect and try again."
                )
                break

            delay = min(
                self._RECONNECT_BASE_DELAY * (2 ** (reconnect_num - 1)),
                self._RECONNECT_MAX_DELAY,
            )
            status_msg = (
                f"Connection lost. Reconnecting in {delay:.0f} s… "
                f"(attempt {reconnect_num}/{self.MAX_RECONNECTS})"
            )
            _report(f"[HermesBleProxy] {status_msg}", logging.WARNING)
            self._set_status(status_msg)

            # Sleep for the back-off period, but wake immediately on shutdown.
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=delay)
                break  # Shutdown requested during sleep.
            except asyncio.TimeoutError:
                pass  # Normal: delay elapsed.

            if self.shutdown_event.is_set():
                break

            # ── Reconnect attempt ───────────────────────────────────────────
            # Packets in flight before the drop cannot be ordered against the new stream: emit them, then start over.
            self._assembler._flush_all()
            self._assembler.reset()
            self.client = BleakClient(
                device_address,
                disconnected_callback=self._on_ble_disconnected,
            )
            try:
                await self.client.connect()
                self.is_connected = True
                self._connection_error = None
                reconnected_msg = f"Reconnected to {device_name}. Streaming resumed."
                _report(f"[HermesBleProxy] {reconnected_msg}")
                self._set_status(reconnected_msg)
            except Exception as e:
                _report(f"[HermesBleProxy] Reconnect attempt {reconnect_num} failed: {e}", logging.WARNING)
                self.is_connected = False
                # Loop continues: will sleep-then-retry with incremented counter.

        # ── Cleanup ──────────────────────────────────────────────────────────
        elapsed = time() - session_start
        _report(
            f"[HermesBleProxy] Session ended after {elapsed:.1f} s. "
            f"Reconnections: {reconnect_num}."
        )
        if self.client is not None and self.client.is_connected:
            try:
                await self.client.disconnect()
                _report("[HermesBleProxy] Disconnected cleanly.")
            except Exception:
                pass
        self.is_connected = False

    # ------------------------------------------------------------------ #
    # Worker-thread processors                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def decode_eeg_payload(data: bytes, packet_number: int) -> list[list[float]]:
        """Samples of one packet: 24 bytes per sample, 8 channels of 3-byte big-endian signed values."""
        samples: list[list[float]] = []
        n_complete = len(data) // 24
        if len(data) % 24 != 0:
            _report(
                f"Warning: packet {packet_number} has {len(data)} bytes "
                f"(not a multiple of 24). Dropping {len(data) % 24} trailing byte(s).",
                logging.WARNING,
            )
        for i in range(0, n_complete * 24, 24):
            sample = []
            for j in range(0, 24, 3):
                channel_data = data[i + j : i + j + 3]
                sample.append(int.from_bytes(channel_data, byteorder="big", signed=True))
            samples.append(HermesBleProxy.convert_ads1299_to_microvolts(sample))
        return samples

    @staticmethod
    def worker_process(eeg_queue: queue.Queue, callback: EegCallback) -> None:
        errors = 0
        while True:
            task = eeg_queue.get()
            if task is None:
                break
            try:
                packet_received, data, packet_number = task

                if packet_received and data is not None:
                    samples = HermesBleProxy.decode_eeg_payload(data, packet_number)
                    if not samples:
                        _report(f"Warning: packet {packet_number} has no complete samples, skipping.", logging.WARNING)
                        continue
                else:
                    samples = [[float("nan")] * 8 for _ in range(10)]

                if callback is not None:
                    callback(samples)

            except Exception as e:
                # Keep the stream alive: one bad packet or failing callback must not stop the EEG for good.
                errors += 1
                if errors <= 20 or errors % 100 == 0:
                    _report(f"Error in EEG worker (#{errors}): {e}", logging.ERROR, exc_info=errors <= 3)

    @staticmethod
    def motion_process(motion_queue: queue.Queue, callback: MotionCallback) -> None:
        while True:
            sample = motion_queue.get()
            if sample is None:
                break
            try:
                (
                    timestamp,
                    ax_raw, ay_raw, az_raw,
                    gx_raw, gy_raw, gz_raw,
                    cx_raw, cy_raw, cz_raw,
                ) = sample

                ax_raw *= ACC_SENS
                ay_raw *= ACC_SENS
                az_raw *= ACC_SENS

                gx_raw *= GYRO_SENS
                gy_raw *= GYRO_SENS
                gz_raw *= GYRO_SENS

                cx_raw *= MAG_SENS
                cy_raw *= MAG_SENS
                cz_raw *= MAG_SENS

                out = (ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw)
                if callback is not None:
                    callback(out)
            except Exception as e:
                _report(f"[HermesBleProxy] motion_process: {e}", logging.ERROR)

    @staticmethod
    def convert_ads1299_to_microvolts(
        raw_values: list[int], gain: int = 12, vref: float = 4.5
    ) -> list[float]:
        lsb_uV = (2 * vref * 1e6) / (gain * (2**24))
        return [val * lsb_uV for val in raw_values]
