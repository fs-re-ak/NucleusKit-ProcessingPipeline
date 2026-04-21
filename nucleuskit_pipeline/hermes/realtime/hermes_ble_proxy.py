"""
Hermes V1 BLE client: EEG + motion notifications (ported from the legacy real-time example).

Runs asyncio on a dedicated thread; decodes EEG on worker threads via queues.
"""

from __future__ import annotations

import asyncio
import datetime
import queue
import struct
import sys
import threading
from time import sleep, time
from typing import Callable

from bleak import BleakClient

ACC_SENS = 0.061 / 1000  # 0.061 mg/LSB → g
GYRO_SENS = 8.75 / 1000  # 8.75 mdps/LSB → dps
MAG_SENS = 0.14 / 1000  # 0.14 mgauss/LSB → gauss

EEG_DATA_UUID = "9fa480e1-4967-11e5-a151-0002a5d5c51b"
EEG_CONFIG_UUID = "9fa480e2-4967-11e5-a151-0002a5d5c51b"
EVENT_UUID = "9fa48301-4967-11e5-a151-0002a5d5c51b"
MOTION_UUID = "9fa48201-4967-11e5-a151-0002a5d5c51b"

HERMES_NAME_SUBSTRING = "Hermes V1"

EegCallback = Callable[[list], None] | None
MotionCallback = Callable[[tuple], None] | None


class HermesBleProxy:
    def __init__(
        self,
        mac_address: str,
        eeg_callback: EegCallback = None,
        motion_callback: MotionCallback = None,
    ) -> None:
        self.is_connected = False
        self.client: BleakClient | None = None
        self.last_packet: int | None = None
        self.packets: list[int] = []
        self.samples_per_packets: list[bytes | None] = []
        self.packet_received: list[bool] = []
        self.mac_address = mac_address

        self.eeg_queue: queue.Queue = queue.Queue()
        self.motion_queue: queue.Queue = queue.Queue()

        self.shutdown_event = asyncio.Event()
        self.loop: asyncio.AbstractEventLoop | None = None

        self._connection_error: str | None = None
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

    async def motion_handler(self, _sender: object, data: bytearray) -> None:
        try:
            now = datetime.datetime.now()
            ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw = struct.unpack_from(
                "<hhhhhhhhh", data
            )
            timestamp_epoch = now.timestamp()
            self.motion_queue.put_nowait(
                (timestamp_epoch, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw)
            )
        except Exception as e:
            print(f"[HermesBleProxy] motion_handler: {e}")

    async def config_handler(self, _sender: object, data: bytearray) -> None:
        try:
            print(data)
        except Exception as e:
            print(f"[HermesBleProxy] config_handler: {e}")

    async def notification_handler(self, sender: object, data: bytearray) -> None:
        print(f"Notification from {sender}: {data}")

    async def packet_handler(self, client: BleakClient, _sender: object, data: bytearray) -> None:
        try:
            current_packet = data[0]
            payload = data[1:]

            missing_packets = self.detect_missing_packets(self.last_packet, current_packet)

            if len(missing_packets) > 0:
                for packet in missing_packets:
                    self.packets.append(packet)
                    self.packet_received.append(False)
                    self.samples_per_packets.append(None)

            self.samples_per_packets.append(payload)
            self.packet_received.append(True)
            self.packets.append(current_packet)
            self.last_packet = current_packet

            self.xfer_packets()

        except Exception as e:
            print(f"[HermesBleProxy] packet_handler: {e}")

    def detect_missing_packets(self, last_packet: int | None, current_packet: int) -> list[int]:
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

            self.eeg_queue.put((self.packet_received[0], self.samples_per_packets[0], self.packets[0]))

            self.packet_received = self.packet_received[1:]
            self.samples_per_packets = self.samples_per_packets[1:]
            self.packets = self.packets[1:]

    def _run_async_main(self) -> None:
        # Windows: Proactor loop in a non-main thread breaks many asyncio BLE stacks; use Selector.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.main_task(self.mac_address))

    def trigger_shutdown(self) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.shutdown_event.set)

    async def main_task(self, device_address: str, device_name: str = HERMES_NAME_SUBSTRING) -> None:
        self.client = BleakClient(device_address)
        try:
            try:
                await self.client.connect()
            except Exception as e:
                self._connection_error = str(e)
                self.is_connected = False
                return

            print(f"Connected to {device_name} [{device_address}]")
            self._connection_error = None
            self.is_connected = True
            start = time()

            await self.client.start_notify(EVENT_UUID, self.notification_handler)
            print("Subscribed to button notifications.")

            await self.client.start_notify(MOTION_UUID, self.motion_handler)
            print("Subscribed to motion notifications.")

            await self.client.start_notify(EEG_CONFIG_UUID, self.config_handler)
            print("Subscribed to EEG config notifications.")

            await self.client.start_notify(
                EEG_DATA_UUID,
                lambda sender, data: asyncio.create_task(self.packet_handler(self.client, sender, data)),
            )
            print("Subscribed to EEG data notifications.")

            await self.shutdown_event.wait()

            print(time() - start)

            await self.client.stop_notify(EEG_DATA_UUID)
            await self.client.stop_notify(EVENT_UUID)
            await self.client.stop_notify(MOTION_UUID)
            await self.client.stop_notify(EEG_CONFIG_UUID)
            print("Finished")
            self.is_connected = False

        except Exception as e:
            self._connection_error = str(e)
            self.is_connected = False
            print(f"[HermesBleProxy] main_task error: {e}")

        finally:
            if self.client is not None and self.client.is_connected:
                await self.client.disconnect()
                print("Disconnected from device.")
            self.is_connected = False

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
                        print(f"Warning: packet {packet_number} has no complete samples, skipping.")
                        continue
                    for i in range(0, n_complete * 24, 24):
                        sample = []
                        for j in range(0, 24, 3):
                            channel_data = data[i + j : i + j + 3]
                            sample.append(int.from_bytes(channel_data, byteorder="big", signed=True))
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
                timestamp, ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw, cx_raw, cy_raw, cz_raw = sample

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
    def convert_ads1299_to_microvolts(raw_values: list[int], gain: int = 12, vref: float = 4.5) -> list[float]:
        lsb_uV = (2 * vref * 1e6) / (gain * (2**24))
        return [val * lsb_uV for val in raw_values]
