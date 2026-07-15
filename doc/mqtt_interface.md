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

Devices are not configured explicitly. A controller **discovers** recorders by subscribing to the status topic and collecting unique device IDs from incoming packets.

---

## 2. Broker Connection

| Parameter | Default | Notes |
|-----------|---------|-------|
| Protocol | MQTT v3.1.1 | |
| Port | `1883` | Standard unencrypted MQTT |
| Keepalive | `60` s | |
| QoS | `0` | All subscriptions and publishes use QoS 0 |
| Clean session | `true` | |

Host and port are deployment-specific. Controllers and devices must connect to the **same broker**.

---

## 3. Topics

### 3.1 Status topic (device publishes, controller subscribes)

| Setting | Default |
|---------|---------|
| Topic | `vizia/status` |

Every Vizia Mobile recorder publishes status JSON to this topic. Controllers subscribe once on connect and treat each message as an update for the device identified by the `id` field.

### 3.2 Command topic (controller publishes, device subscribes)

| Setting | Default |
|---------|---------|
| Topic pattern | `{recorderId}/control/` |

- `{recorderId}` is the device identifier from status packets (field `id`).
- The suffix `control/` is configurable but defaults to `control/`.

**Example:** A device with `id` `"lab-phone-01"` receives commands on:

```
lab-phone-01/control/
```

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
{
  "id": "<recorderId>",
  "recording": <boolean>,
  "battery": <integer>
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique recorder identifier; also used as the `{recorderId}` in command topics |
| `recording` | boolean | `true` while a recording session is active |
| `battery` | integer | Battery level in percent; `-1` if unknown |

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

On the device, tags are written into the session event log. After a session is ingested, those events appear in `rawData/rawEvents.csv` and flow through the NucleusKit events pipeline (see [`pipeline_events.md`](pipeline_events.md)).

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
| Status topic | `vizia/status` | Devices publish; controllers subscribe |
| Command topic suffix | `control/` | Appended after `{recorderId}/` for commands and tags |
| Broker port | `1883` | All participants |

Both topic names are configurable per deployment. All participants must agree on the same values.

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
| [`pipeline_events.md`](pipeline_events.md) | How session event logs (including MQTT-injected tags) are processed after ingest |
| [`overview.md`](overview.md) | Full offline processing pipeline |
