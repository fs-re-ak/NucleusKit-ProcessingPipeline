# Positioning Pipelines — GPS and UWB

This document covers two independent positioning pipelines that run at the end of the pipeline:

1. **UWB Indoor Positioning** (`processUWB`) — computes 2-D/3-D position from Ultra-Wideband ranging data using multilateration.
2. **GPS Outdoor Positioning** (`processGPS`) — resamples GNSS coordinates to the shared 2 Hz timebase.

Both pipelines are optional: if the corresponding raw data file is absent, the step is skipped with a warning and no output is written.

---

## Part 1 — UWB Indoor Positioning

**Module:** `nucleuskit_pipeline/position/uwb_processor.py`  
**Entry point:** `processUWB(basepath)`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

### 1.1 Purpose

Computes participant positions in a room-coordinate system from Ultra-Wideband time-of-flight ranging measurements, using a three-pass progressive multilateration approach that maximises coverage even when only a subset of anchors are visible.

### 1.2 Input Data

**Source file** (first match wins):

| Filename | Location |
|----------|----------|
| `uwb_0.csv` | `rawData/` |
| `uwb.tmp` | `rawData/` |

**Raw file format:** Text file, one line per measurement epoch. Lines not containing the keyword `DIST` are discarded (header or non-ranging lines). Each `DIST` line has the structure:

```
timestamp_ms, DIST, n_antennas, [antenna_number, antenna_id, x, y, z, distance] × n_antennas
```

- `timestamp_ms`: Hardware clock in milliseconds. The first line's timestamp is used as the reference (`refTime`) to produce session-relative timestamps.
- `n_antennas`: Number of anchor antennas visible for this epoch.
- For each antenna: its sequential number, a short hexadecimal ID, its known 3-D coordinates (x, y, z) in the room frame (metres), and the measured distance in metres.

### 1.3 Three-pass Multilateration

Position is solved progressively using three passes, each targeting epochs that could not be solved in the preceding pass.

#### Pass 1 — Standard multilateration (≥ 3 anchors) — `_idealSolver` / `_run_firstPass`

When 3 or more anchors are visible, a least-squares minimisation is applied:

```
minimize Σ (‖x − cᵢ‖ − rᵢ)²   over x ∈ ℝ³
```

where `cᵢ` are the known anchor positions and `rᵢ` are the measured distances. The initial guess `x₀` is a distance-weighted centroid:

```
W = [(n−1)·S / (S − rᵢ)]   (with S = Σrᵢ)
x₀ = Σ Wᵢ · cᵢ / Σ Wᵢ
```

Minimisation is performed with `scipy.optimize.minimize` using the Nelder-Mead method (`_uwb_solve`). Epochs that succeed in Pass 1 are marked and not revisited.

#### Pass 2 — Two-anchor solver (`_2AntSolver` / `_run_secondPass`)

For epochs with exactly 2 visible anchors, the same least-squares formulation is applied, but the initial guess `x₀` is set to the **closest solved position** (either from the immediately preceding or immediately following solved epoch, whichever is closer in time). This spatial continuity assumption is valid for a walking subject and substantially constrains the under-determined 2-anchor problem.

Zero distances are replaced with a small regularisation value (`1e-10`) to avoid division issues.

#### Pass 3 — Single-anchor solver (`_1AntSolver` / `_run_thirdPass`)

For epochs with exactly 1 visible anchor, the position is constrained to lie on the circle of radius `r` centred at the anchor's (x, y) position. The closest point on this circle to the best-guess position (from the nearest already-solved epoch) is returned by `find_closest_point_on_circle`. This gives a plausible continuation of the track without multilateration.

#### Z-coordinate assignment (`_bruteForceZ`)

The Z (elevation) coordinate returned by the solver is overridden by a lookup table that maps known antenna IDs to floor levels:

| Set | Anchor IDs (hex) | Assigned Z (m) |
|-----|-----------------|----------------|
| First floor | DCB0, CF01, 9200, 4A82, 4107, 5338, D92B, 8902, 028C, D29D, 4709, 5D17, 00A0 | 2.4 m |
| Second floor | 4496, CF1C, 0CA1, D816, 021A, D71A, D806, 012A, 0111, 8124, 0181, 0B2E, 91A9, D285, CFB0 | 70 m |

If none of the visible antennas match either set, the solver's Z is kept unchanged.

### 1.4 Output Assembly

After all three passes:
- `None` entries (epochs that could not be solved in any pass) are dropped.
- Timestamps are converted from milliseconds to seconds (÷ 1000).
- Results are assembled as a DataFrame with columns `[Timestamp, x, y, z]`.

### 1.5 Outputs

| File | Location | Description |
|------|----------|-------------|
| `positions.csv` | `results/` | `[Timestamp, x, y, z]`; positions in metres at the raw UWB measurement rate |
| `antennas.csv` | `features/` | `[id, x, y, z]` for all unique antennas observed during the session |

**Note:** Unlike all other pipeline outputs, `positions.csv` is **not resampled** to the shared 2 Hz timebase (a TODO comment marks this in the source). Downstream consumers may need to interpolate to align with the 0.5 s grid.

### 1.6 Optional Zone Tagging (`positionToZone`)

`positionToZone(recPath, resources_path)` is a utility (not part of the default pipeline steps) that reads a region-of-interest definition file (`resources/roi.csv` — columns: `label, x, y, w, h`) and tags each position in `positions.csv` with the ROI label of the bounding box it falls within. Output is written to `results/zones.csv`.

### 1.7 Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/positions.csv` exists | Entire step skipped |
| File absent | Full computation |

---

## Part 2 — GPS Positioning

**Module:** `nucleuskit_pipeline/position/gps_processor.py`  
**Entry point:** `processGPS(recpath)`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

### 2.1 Purpose

Resamples raw GNSS (GPS) location data onto the shared 2 Hz pipeline timebase using linear interpolation, producing a clean time-aligned latitude/longitude series.

### 2.2 Input Data

**Source file** (first match wins):

| Filename | Location |
|----------|----------|
| `gps.csv` | `rawData/` |
| `gps.tmp` | `rawData/` |

**Format:** Headerless CSV, three columns:

```
unix_timestamp_s, latitude_decimal_degrees, longitude_decimal_degrees
```

Timestamps are absolute Unix epoch seconds. The first timestamp is subtracted to produce session-relative seconds.

### 2.3 Processing

1. Load the raw GPS file into a DataFrame.
2. Convert absolute Unix timestamps to session-relative seconds: `t_rel = t_raw − t_raw[0]`.
3. Build a regular output grid at **0.5 s steps** from `0` to `t_rel[-1]`.
4. Interpolate latitude and longitude independently using `numpy.interp` with `left=NaN, right=NaN`. Points outside the range of the raw data (extrapolation) are emitted as `NaN`.

No artifact rejection is applied. The GPS signal is assumed to be clean from the hardware; large dropouts will appear as `NaN` bins.

### 2.4 Outputs

### `results/gpsDf.csv`

| Column | Unit | Description |
|--------|------|-------------|
| `Timestamp` | s | Seconds from recording start, 0.5 s steps |
| `Latitude` | decimal degrees | GPS latitude |
| `Longitude` | decimal degrees | GPS longitude |

Float precision: 7 decimal places (~1 cm at the equator). `NaN` written as `NULL`.

### 2.5 Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/gpsDf.csv` exists | Entire step skipped |
| File absent | Full computation |

---

## Summary

| Pipeline | Input | Output | Rate | Caching |
|----------|-------|--------|------|---------|
| UWB | `rawData/uwb_0.csv` or `uwb.tmp` | `results/positions.csv`, `features/antennas.csv` | Raw UWB rate (not resampled) | Yes |
| GPS | `rawData/gps.csv` or `gps.tmp` | `results/gpsDf.csv` | 2 Hz (0.5 s) | Yes |

---

*© RE-AK Technologies Inc.*
