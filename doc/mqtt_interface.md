# MQTT Interface — Vizia Mobile Device Control

This document describes the MQTT contract used to discover Vizia Mobile Android recorders, monitor their state, and send recording commands and event markers. The description is language-agnostic: any MQTT client that follows the topic layout and JSON envelopes below can participate.

**Reference implementation:** `nucleuskit_toolkit/mqtt/mqtt_controller.py`  
**GUI:** MQTT Controller page in the NucleusKit desktop application

---

## 1. Overview

The interface uses a **publish/subscribe** model on a shared MQTT broker:

| Direction | Purpose |
|-----------|---------|
| **Device → broker** | Periodic **status** packets announce presence, recording state, and battery level |
| **Controller → device** | **Commands** start/stop recording; **tags** inject timestamped event markers into the active session |

Devices are not configured explicitly. A controller **discovers** recorders by subscribing to the status topics and collecting unique device IDs from incoming packets.

Topics are **scoped by experiment** (contract v1): `{experimentId}/{recorderId}/status` and `/control`, so a broker can
restrict each login to its own experiments and a controller discovers an experiment's devices with one wildcard
subscription. Phones without an experiment use the **legacy** topics (`vizia/status`, `{recorderId}/control/`), for
local brokers only. Phones may also publish raw EXG on `{experimentId}/{recorderId}/data/exg` for live processing; the
controller does not use it.

---

## 2. Broker Connection

| Parameter | Default | Notes |
|-----------|---------|-------|
| Protocol | MQTT v3.1.1 | |
| Broker address | — | URL `mqtt://` (plain, port 1883), `mqtts://` (TLS, 8883), `ws://` / `wss://` (WebSocket, path `/mqtt`), or a bare IP / hostname (plain TCP on the configured port) |
| Authentication | anonymous | Username / password when set |
| TLS CA | system CAs | Optional CA certificate file (the CA that signed the broker certificate) |
| Keepalive | `60` s | |
| QoS | `0` | All subscriptions and publishes use QoS 0 |
| Clean session | `true` | |

Controllers and devices must connect to the **same broker**. Use TLS and a login whenever the broker is reachable
outside an isolated local network: the same link can carry raw biosignals.

---

## 3. Topics

### 3.1 Status topic (device publishes, controller subscribes)

| Device has an experiment | Topic | Controller subscribes to |
|---|---|---|
| Yes | `{experimentId}/{recorderId}/status` | `{experimentId}/+/status` |
| No (legacy) | `vizia/status` (shared, configurable) | `vizia/status` |

Controllers subscribe on connect and treat each message as an update for one device: the `{recorderId}` of the topic,
or the `id` field on the legacy shared topic. On experiment topics, a payload `id` that differs from the topic's
recorder is rejected.

### 3.2 Command topic (controller publishes, device subscribes)

| Device has an experiment | Topic |
|---|---|
| Yes | `{experimentId}/{recorderId}/control` |
| No (legacy) | `{recorderId}/control/` (suffix configurable) |

Each device receives commands on the control topic that matches the status topic it was discovered on.

**Example:** device `"lab-phone-01"` of experiment `"exp-1"` receives commands on `exp-1/lab-phone-01/control`; without
an experiment, on `lab-phone-01/control/`.

---

## 4. Message Envelope

All command and tag payloads share a common envelope:

```json
{
  "Type": "<MESSAGE_TYPE>",
  "Content": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `Type` | string | Discriminator: `COMMAND` or `TAG` |
| `Content` | object | Type-specific payload (see sections 5–7) |

Payloads are UTF-8 JSON strings. Malformed JSON on the status topic should be ignored by controllers.

---

## 5. Status Packets

**Publisher:** Vizia Mobile device  
**Topic:** status topic (default `vizia/status`)  
**Direction:** device → controller

### 5.1 Payload format

```json
{"v": 1, "id": "<recorderId>", "experimentId": "<uuid or null>", "sessionId": "<id or null>",
 "recording": <boolean>, "battery": <integer>, "headsetConnected": <boolean>}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique recorder identifier; also used as the `{recorderId}` in command topics |
| `recording` | boolean | `true` while a recording session is active |
| `battery` | integer | Battery level in percent; `-1` if unknown |
| `v`, `experimentId`, `sessionId`, `headsetConnected` | | Added in contract v1 (optional for controllers). Older phones send only `id`, `recording` and `battery` |

### 5.2 Controller behaviour

On each valid status packet, a controller:

1. Parses `id`, `recording`, and `battery`.
2. Upserts a device record keyed by `id`.
3. Updates `last_seen` to the current time.

A device is **discovered** the first time its `id` appears. No separate registration step is required.

---

## 6. Commands

**Publisher:** Controller (or any authorised client)  
**Topic:** `{recorderId}/control/`  
**Direction:** controller → device  
**Type:** `COMMAND`

Commands instruct a recorder to start or stop its recording session.

### 6.1 Start recording

```json
{
  "Type": "COMMAND",
  "Content": {
    "Code": "START_RECORDING"
  }
}
```

### 6.2 Stop recording

```json
{
  "Type": "COMMAND",
  "Content": {
    "Code": "STOP_RECORDING"
  }
}
```

### 6.3 Targeting

| Target | Behaviour |
|--------|-----------|
| Specific `{recorderId}` | Publish once to `{recorderId}/control/` |
| All discovered devices | Publish the same payload to `{recorderId}/control/` for every device ID seen on the status topic |

If the controller is not connected, or no devices have been discovered yet, outbound commands are not sent.

---

## 7. Tags

**Publisher:** Controller (or any authorised client)  
**Topic:** `{recorderId}/control/`  
**Direction:** controller → device  
**Type:** `TAG`

Tags inject an **event marker** into the device's active recording session. They use the same command topic and envelope as commands, but with `Type` set to `TAG`.

### 7.1 Payload format

```json
{
  "Type": "TAG",
  "Content": {
    "Code": "<event_code>",
    "Values": "<optional_detail_string>"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `Code` | string | yes | Short event label (e.g. `STIMULUS_ON`, `TRIAL_START`) |
| `Values` | string | no | Free-text detail; may be empty (`""`) |

### 7.2 Examples

Marker with code only:

```json
{
  "Type": "TAG",
  "Content": {
    "Code": "STIMULUS_ON",
    "Values": ""
  }
}
```

Marker with detail string:

```json
{
  "Type": "TAG",
  "Content": {
    "Code": "TRIAL_START",
    "Values": "block=2 trial=7"
  }
}
```

### 7.3 Targeting

Tags follow the same targeting rules as commands (single device or all discovered devices).

### 7.4 Downstream use

On the device, tags are written into the session event log. After a session is ingested, those events are read from `rawData/` (`rawEvents.csv`, `events_0.csv` or `event.csv`) by re-ak-core (see [events](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/pipeline_events.md)).

---

## 8. Typical Session Flow

```
Controller                          Broker                         Device
    |                                 |                              |
    |--- connect + subscribe -------->|                              |
    |    (vizia/status)               |                              |
    |                                 |<----- status {id, ...} ------|
    |<---- status forwarded ----------|                              |
    |   (device discovered)           |                              |
    |                                 |                              |
    |--- COMMAND START_RECORDING ---->|-----> control/ ------------->|
    |                                 |                              |
    |                                 |<----- status recording=true -|
    |<---- status forwarded ----------|                              |
    |                                 |                              |
    |--- TAG {Code, Values} --------->|-----> control/ ------------->|
    |                                 |                              |
    |--- COMMAND STOP_RECORDING ----->|-----> control/ ------------->|
    |                                 |                              |
    |                                 |<----- status recording=false |
    |<---- status forwarded ----------|                              |
```

---

## 9. Configuration Summary

| Setting | Default | Used by |
|---------|---------|---------|
| Experiment | — | Experiment whose devices to discover on `{experimentId}/+/status` |
| Broker address | — | URL or bare IP / hostname (see section 2) |
| Broker port | `1883` | Only for a bare IP / hostname |
| Username / password | — | Broker login (the password is not saved by the desktop app) |
| CA file | — | CA certificate for TLS |
| Status topic (legacy) | `vizia/status` | Devices without an experiment publish; controllers subscribe |
| Command topic suffix (legacy) | `control/` | Appended after `{recorderId}/` for commands and tags |

The legacy topic names are configurable per deployment. All participants must agree on the same values.

---

## 10. Error Handling

| Condition | Expected behaviour |
|-----------|-------------------|
| Controller not connected to broker | Do not publish commands or tags |
| No devices discovered | Do not publish broadcast (all-devices) commands or tags |
| Malformed status JSON | Log and ignore; do not update device registry |
| Unexpected broker disconnect | Mark disconnected; rely on client reconnect logic |

---

## 11. Related Documentation

| Document | Relevance |
|----------|-----------|
| [re-ak-core events](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/pipeline_events.md) | How session event logs (including MQTT-injected tags) are processed after ingest |
| [`overview.md`](overview.md) | Current offline processing pipeline |
