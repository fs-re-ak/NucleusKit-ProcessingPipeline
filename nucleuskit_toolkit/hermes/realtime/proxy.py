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
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import queue
import struct
import sys
import threading
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
        self.last_packet: int | None = None
        self.packets: collections.deque[int] = collections.deque()
        self.samples_per_packets: collections.deque[bytes | None] = collections.deque()
        self.packet_received: collections.deque[bool] = collections.deque()
        self.mac_address = mac_address

        # Bounded queues — drop oldest item rather than growing without bound.
        self.eeg_queue: queue.Queue    = queue.Queue(maxsize=256)
        self.motion_queue: queue.Queue = queue.Queue(maxsize=128)

        # Bug 1 fix: asyncio.Event MUST be created inside the worker event loop.
        # Initialise to None here; _run_async_main populates them before any
        # coroutine runs.
        self.shutdown_event: asyncio.Event | None     = None
        self._disconnected_event: asyncio.Event | None = None
        self.loop: asyncio.AbstractEventLoop | None   = None

        self._connection_error: str | None = None
        self._status_callback = status_callback

        self._async_thread = threading.Thread(target=self._run_async_main, daemon=True)

        self.eeg_worker = threading.Thread(
            target=HermesBleProxy.worker_process,
            args=(self.eeg_queue, eeg_callback),
            daemon=True,
        )
        self.eeg_worker.start()

        self.motion_worker = threading.Thread(
            target=HermesBleProxy.motion_process,
            args=(self.motion_queue, motion_callback),
            daemon=True,
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

        self.loop.run_until_complete(self.main_task(self.mac_address))

    # ------------------------------------------------------------------ #
    # BLE callbacks                                                        #
    # ------------------------------------------------------------------ #

    def _on_ble_disconnected(self, client: BleakClient) -> None:
        """Bug 3 fix: called by bleak when the connection drops unexpectedly.

        bleak schedules this callback inside the event loop, so setting an
        asyncio.Event directly is safe here.
        """
        self.is_connected = False
        print("[HermesBleProxy] Connection dropped by device/OS.")
        if self._disconnected_event is not None:
            self._disconnected_event.set()

    def _set_status(self, msg: str) -> None:
        """Forward a status string to the UI callback (thread-safe)."""
        if self._status_callback is not None:
            try:
                self._status_callback(msg)
            except Exception:
                pass

    def _handle_eeg_data(self, _sender: object, data: bytearray) -> None:
        """Bug 2 fix: synchronous EEG notification handler.

        Called on the event-loop thread via loop.call_soon_threadsafe (see
        _subscribe_characteristics), so asyncio.create_task is not needed and
        no cross-thread hazard exists.
        """
        try:
            current_packet = data[0]
            payload = data[1:]

            for packet in self.detect_missing_packets(self.last_packet, current_packet):
                self.packets.append(packet)
                self.packet_received.append(False)
                self.samples_per_packets.append(None)

            self.samples_per_packets.append(payload)
            self.packet_received.append(True)
            self.packets.append(current_packet)
            self.last_packet = current_packet

            self.xfer_packets()
        except Exception as e:
            print(f"[HermesBleProxy] _handle_eeg_data: {e}")

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
            print(f"[HermesBleProxy] motion_handler: {e}")

    async def config_handler(self, _sender: object, data: bytearray) -> None:
        try:
            print(data)
        except Exception as e:
            print(f"[HermesBleProxy] config_handler: {e}")

    async def notification_handler(self, sender: object, data: bytearray) -> None:
        print(f"Notification from {sender}: {data}")

    # ------------------------------------------------------------------ #
    # GATT subscribe / unsubscribe helpers                                 #
    # ------------------------------------------------------------------ #

    async def _subscribe_characteristics(self) -> None:
        """Subscribe to all four GATT characteristics."""
        assert self.loop is not None

        await self.client.start_notify(EVENT_UUID, self.notification_handler)
        print("[HermesBleProxy] Subscribed to button notifications.")

        await self.client.start_notify(MOTION_UUID, self.motion_handler)
        print("[HermesBleProxy] Subscribed to motion notifications.")

        await self.client.start_notify(EEG_CONFIG_UUID, self.config_handler)
        print("[HermesBleProxy] Subscribed to EEG config notifications.")

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
        print("[HermesBleProxy] Subscribed to EEG data notifications.")

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
        self.client = BleakClient(
            device_address,
            disconnected_callback=self._on_ble_disconnected,  # Bug 3 fix
        )
        try:
            await self.client.connect()
        except Exception as e:
            self._connection_error = str(e)
            self.is_connected = False
            print(f"[HermesBleProxy] Initial connection failed: {e}")
            return

        print(f"[HermesBleProxy] Connected to {device_name} [{device_address}]")
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
                    print(f"[HermesBleProxy] Streaming error: {e}")
                finally:
                    self.is_connected = False

            # Shutdown takes priority over reconnect.
            if self.shutdown_event.is_set():
                break

            # ── Unexpected disconnect: plan a reconnect attempt ─────────────
            reconnect_num += 1
            if reconnect_num > self.MAX_RECONNECTS:
                print(
                    f"[HermesBleProxy] Max reconnects ({self.MAX_RECONNECTS}) reached. Stopping."
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
                f"Connection lost. Reconnecting in {delay:.0f} s\u2026 "
                f"(attempt {reconnect_num}/{self.MAX_RECONNECTS})"
            )
            print(f"[HermesBleProxy] {status_msg}")
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
            self.last_packet = None
            self.client = BleakClient(
                device_address,
                disconnected_callback=self._on_ble_disconnected,
            )
            try:
                await self.client.connect()
                self.is_connected = True
                self._connection_error = None
                reconnected_msg = f"Reconnected to {device_name}. Streaming resumed."
                print(f"[HermesBleProxy] {reconnected_msg}")
                self._set_status(reconnected_msg)
            except Exception as e:
                print(f"[HermesBleProxy] Reconnect attempt {reconnect_num} failed: {e}")
                self.is_connected = False
                # Loop continues: will sleep-then-retry with incremented counter.

        # ── Cleanup ──────────────────────────────────────────────────────────
        elapsed = time() - session_start
        print(
            f"[HermesBleProxy] Session ended after {elapsed:.1f} s. "
            f"Reconnections: {reconnect_num}."
        )
        if self.client is not None and self.client.is_connected:
            try:
                await self.client.disconnect()
                print("[HermesBleProxy] Disconnected cleanly.")
            except Exception:
                pass
        self.is_connected = False

    # ------------------------------------------------------------------ #
    # Packet processing                                                    #
    # ------------------------------------------------------------------ #

    def detect_missing_packets(
        self, last_packet: int | None, current_packet: int
    ) -> list[int]:
        missing_packets: list[int] = []
        if last_packet is not None:
            if last_packet == 127:
                missing_packet = 0
                while missing_packet != current_packet:
                    print(f"Dropped packet {missing_packet}")
                    missing_packets.append(missing_packet)
                    missing_packet = (missing_packet + 1) % 128
            elif last_packet + 1 != current_packet:
                missing_packet = last_packet + 1
                while missing_packet != current_packet:
                    print(f"Dropped packet {missing_packet}")
                    missing_packets.append(missing_packet)
                    missing_packet = (missing_packet + 1) % 128
        return missing_packets

    def xfer_packets(self) -> None:
        while self.packet_received:
            delay = (self.last_packet + 128 - (self.packets[0] - 128)) % 128
            if not self.packet_received[0] and delay < 10:
                break
            item = (
                self.packet_received[0],
                self.samples_per_packets[0],
                self.packets[0],
            )
            try:
                self.eeg_queue.put_nowait(item)
            except queue.Full:
                try:
                    self.eeg_queue.get_nowait()
                except queue.Empty:
                    pass
                self.eeg_queue.put_nowait(item)
            self.packet_received.popleft()
            self.samples_per_packets.popleft()
            self.packets.popleft()

    # ------------------------------------------------------------------ #
    # Worker-thread processors                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def worker_process(eeg_queue: queue.Queue, callback: EegCallback) -> None:
        while True:
            try:
                task = eeg_queue.get()
                if task is None:
                    break

                packet_received, data, packet_number = task

                if packet_received and data is not None:
                    samples: list[list[float]] = []
                    n_complete = len(data) // 24
                    if len(data) % 24 != 0:
                        print(
                            f"Warning: packet {packet_number} has {len(data)} bytes "
                            f"(not a multiple of 24). Dropping {len(data) % 24} trailing byte(s)."
                        )
                    if n_complete == 0:
                        print(
                            f"Warning: packet {packet_number} has no complete samples, skipping."
                        )
                        continue
                    for i in range(0, n_complete * 24, 24):
                        sample = []
                        for j in range(0, 24, 3):
                            channel_data = data[i + j : i + j + 3]
                            sample.append(
                                int.from_bytes(channel_data, byteorder="big", signed=True)
                            )
                        samples.append(HermesBleProxy.convert_ads1299_to_microvolts(sample))
                else:
                    samples = [[float("nan")] * 8 for _ in range(10)]

                if callback is not None:
                    callback(samples)

            except Exception as e:
                print(f"Error in EEG worker: {e}")
                break

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
                print(f"[HermesBleProxy] motion_process: {e}")

    @staticmethod
    def convert_ads1299_to_microvolts(
        raw_values: list[int], gain: int = 12, vref: float = 4.5
    ) -> list[float]:
        lsb_uV = (2 * vref * 1e6) / (gain * (2**24))
        return [val * lsb_uV for val in raw_values]
