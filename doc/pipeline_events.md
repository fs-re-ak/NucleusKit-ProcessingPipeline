# Events Pipeline — Raw Events Processing and Playback Annotations

This document covers two independent event-processing steps that run sequentially in the pipeline:

1. **Event filtering** (`eventProcessor`) — splits `rawEvents.csv` into feature and web event CSVs.
2. **Playback annotation seeding** (`seedPlaybackAnnotations`) — generates `playback_annotations.json` for session review.

---

## Part 1 — Event Filtering

**Module:** `nucleuskit_pipeline/events/processor.py`  
**Entry point:** `eventProcessor(recPath)`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

### 1.1 Purpose

Separates the heterogeneous raw event log into two curated subsets:
- **Feature events** — task-relevant events used for data annotation and analysis.
- **Web events** — browser interaction events used for web-tracking analysis.

### 1.2 Input

| File | Location | Format |
|------|----------|--------|
| `rawEvents.csv` | `rawData/` | CSV: `[timestamp, event_type, payload, …]` |

Events are loaded via `_loadRawEvents`, which normalises timestamps and handles encoding.

### 1.3 Classification Logic

Each raw event row `[timestamp, event_type, payload, …]` is categorised as follows:

| Event type | Feature events | Web events |
|-----------|---------------|-----------|
| `WEB_SCROLL` | No | Yes |
| `WEB_TAB_UPDATE` | Only if `"complete"` in payload | Yes |
| `WEB_INITIAL_SCROLL_POSITION` | No | Yes |
| `NO_KEYPRESS` | Discarded | No |
| `KEYPRESS` | Discarded | No |
| `ANSWER` | Discarded | No |
| All other types | Yes | No |

`WEB_TAB_UPDATE` events with `"complete"` in their payload indicate a completed page navigation and are included in feature events as well as web events, since page-load transitions are relevant to task annotation.

### 1.4 Outputs

| File | Location | Description |
|------|----------|-------------|
| `processedFeatureEvents.csv` | `features/` | Filtered task-relevant events |
| `processedWebFeatureEvents.csv` | `features/` | Web browser interaction events |

Both files are written by `_writeEvents` in the same format as `rawEvents.csv`.

If `rawEvents.csv` is absent or empty, a warning is logged and the step is skipped without writing any output.

---

## Part 2 — Playback Annotation Seeding

**Module:** `nucleuskit_pipeline/events/eventsProcessor.py`  
**Entry point:** `seedPlaybackAnnotations(recPath)`  
**Authors:** RE-AK Technologies Inc., Spring 2026

### 2.1 Purpose

Generates an initial `playback_annotations.json` file that is used by the session playback viewer to display annotated events on the recording timeline. This file is written **once** — if it already exists, the step is skipped entirely to preserve any manual edits made by the researcher.

### 2.2 Input

| File | Location | Format |
|------|----------|--------|
| `event.csv` | `rawData/` | CSV: `[timestamp, event_name, …]` |

The file is read row by row. Rows with fewer than 2 fields or a non-numeric timestamp are silently skipped.

### 2.3 Processing

#### Time zero establishment

The step requires a **`RECORDING_ONSET`** event to establish the recording's time zero. All annotation timestamps are expressed as offsets in seconds from this event. If no `RECORDING_ONSET` row is found, the step is aborted with a warning.

#### Ignored event types

The following administrative events are excluded from the annotation output:

| Event type | Reason |
|------------|--------|
| `RECORDING_ONSET` | Defines t=0; not an analysis annotation |
| `RECIPE_CONFIG` | Experiment configuration marker |
| `SYSTEM_CONFIG` | System configuration marker |
| `CONSENT_INFO` | Consent workflow marker |

All other events become **point annotations** in the output.

#### Annotation object structure

Each annotation point has the following JSON structure:

```json
{
  "id": "<uuid4>",
  "t": 12.345678,
  "label": "EVENT_NAME",
  "visible": true
}
```

- `id`: Unique UUID assigned at generation time.
- `t`: Seconds from `RECORDING_ONSET` (6 decimal places).
- `label`: The event name string from `event.csv`.
- `visible`: Always `true` at generation; may be toggled by the viewer.

### 2.4 Output

| File | Location |
|------|----------|
| `playback_annotations.json` | `features/events/` |

**Schema version:** 1

**Top-level structure:**

```json
{
  "version": 1,
  "points": [ … ],
  "zones":  []
}
```

The `zones` array is initialised empty and is available for manual zone annotations in the playback viewer.

### 2.5 Idempotency

If `features/events/playback_annotations.json` already exists when this step runs, it is **not overwritten**. This design allows researchers to manually add, remove, or adjust annotations in the viewer without those changes being lost on the next pipeline run.

To regenerate the file from scratch, delete it and re-run the pipeline (or run `seedPlaybackAnnotations` directly).

---

## Summary

| Step | Input | Output | Idempotent |
|------|-------|--------|-----------|
| `eventProcessor` | `rawData/rawEvents.csv` | `features/processedFeatureEvents.csv`, `features/processedWebFeatureEvents.csv` | Yes (always overwrites if raw file exists) |
| `seedPlaybackAnnotations` | `rawData/event.csv` | `features/events/playback_annotations.json` | Yes (skips if output exists) |

---

*© RE-AK Technologies Inc.*
