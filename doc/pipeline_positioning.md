# Positioning Pipeline — GPS

This document covers the GPS positioning step (`processGPS`), which runs at the end of the pipeline and resamples GNSS coordinates to the shared 2 Hz timebase. It is optional: if the raw GPS file is absent, the step is skipped with a warning and no output is written.

UWB indoor positioning (`processUWB`) was removed in Nucleus-Kit 2.1.0.

---

## GPS Positioning

**Module:** `nucleuskit_toolkit/position/gps_processor.py`  
**Entry point:** `processGPS(recpath)`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

### 1. Purpose

Resamples raw GNSS (GPS) location data onto the shared 2 Hz pipeline timebase using linear interpolation, producing a clean time-aligned latitude/longitude series.

### 2. Input Data

**Source file** (first match wins):

| Filename | Location |
|----------|----------|
| `gps.csv` | `rawData/` |
| `gps.tmp` | `rawData/` |

**Format:** CSV, three columns (header row optional):

```
unix_timestamp_s, latitude_decimal_degrees, longitude_decimal_degrees
```

If a header row is present its first cell is non-numeric, and the loader skips it automatically. Legacy headerless files are detected and loaded identically.

Timestamps are absolute Unix epoch seconds. The first timestamp is subtracted to produce session-relative seconds.

### 3. Processing

1. Load the raw GPS file into a DataFrame.
2. Convert absolute Unix timestamps to session-relative seconds: `t_rel = t_raw − t_raw[0]`.
3. Build a regular output grid at **0.5 s steps** from `0` to `t_rel[-1]`.
4. Interpolate latitude and longitude independently using `numpy.interp` with `left=NaN, right=NaN`. Points outside the range of the raw data (extrapolation) are emitted as `NaN`.

No artifact rejection is applied. The GPS signal is assumed to be clean from the hardware; large dropouts will appear as `NaN` bins.

### 4. Outputs

### `results/gpsDf.csv`

| Column | Unit | Description |
|--------|------|-------------|
| `Timestamp` | s | Seconds from recording start, 0.5 s steps |
| `Latitude` | decimal degrees | GPS latitude |
| `Longitude` | decimal degrees | GPS longitude |

Float precision: 7 decimal places (~1 cm at the equator). `NaN` written as `NULL`.

### 5. Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/gpsDf.csv` exists | Entire step skipped |
| File absent | Full computation |

---

## Summary

| Pipeline | Input | Output | Rate | Caching |
|----------|-------|--------|------|---------|
| GPS | `rawData/gps.csv` or `gps.tmp` | `results/gpsDf.csv` | 2 Hz (0.5 s) | Yes |

---

*© RE-AK Technologies Inc.*
