# Cognition Pipeline — EEG-based Cognitive Indexes

**Module:** `nucleuskit_pipeline/hermes/processor/cognition_processor.py`  
**Entry point:** `computeCognitiveIndexes(recpath)`  
**Primary output:** `results/Cognition.csv`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

---

## 1. Purpose

The cognition pipeline transforms raw EEG signals from the Hermes device into four high-level cognitive metrics sampled at 2 Hz. The metrics are designed to quantify moment-to-moment cognitive engagement and its hemispheric distribution.

---

## 2. Input Data

**Source file** (first match wins):

| Filename | Location |
|----------|----------|
| `rawEEG_0.csv` | `rawData/` |
| `eeg.tmp` | `rawData/` |
| `eeg.csv` | `rawData/` |
| `eegRec_0.csv` | `rawData/` |

**Format:** Headerless or single-header CSV. Column 0 is the hardware timestamp (milliseconds, auto-normalised to seconds). Columns 1–8 are the eight EXG channels in the order defined by `HermesConstants`.

**Sampling rate:** 250 Hz

**Channel derivation** (performed by `HermesDataInterface.getEEG()`):

| Output column | Formula | Rationale |
|---------------|---------|-----------|
| `AF7` | `col2 − col5 / 2` | Left frontal, half-referenced to mastoid |
| `AF8` | `col1 − col5 / 2` | Right frontal, half-referenced to mastoid |
| `Temporal` | `col5` | Raw temporal reference |
| `LeftHemi` | `col2` | Left hemisphere raw |
| `RightHemi` | `col1 − col5` | Right hemisphere, differential |

Samples whose absolute value is within 0.1 of ±187 500 are considered hardware-disconnected and are replaced with `NaN` before any processing.

---

## 3. Processing Steps

### Step 1 — Hardware gap detection

Before filtering, the pipeline scans the hardware timestamp vector for inter-sample intervals ≥ 5 s (`GAP_THRESHOLD_S`). Each sample **immediately following** such a gap is flagged in a boolean `gap_sample_mask`. Power-band windows that overlap flagged samples will be emitted as `NaN` rows rather than being computed.

If hardware timestamps are unavailable, a synthetic timestamp array (`index / fs`) is used and gap detection is skipped.

### Step 2 — Signal preprocessing (`preprocess_eeg`)

Two operations are applied sequentially:

#### 2a. Bandpass + notch filtering (`bandpass_filter`)

| Parameter | Value |
|-----------|-------|
| Filter type | 4th-order Butterworth |
| Passband | 0.5–45 Hz |
| Notch | 60 Hz (Q = 30) |
| Implementation | Zero-phase `filtfilt` |

Any `NaN` or `Inf` values in a channel are linearly interpolated before filtering to prevent numerical instability. The filtered array is returned as a DataFrame with the same column names.

#### 2b. Statistical outlier removal (`remove_statistical_outliers`)

A z-score threshold of **3** is applied column-by-column to the filtered data. Rows where any channel exceeds the threshold are removed. The computation is done one column at a time to avoid peak memory allocation on long recordings. The resulting boolean keep-mask is propagated to the parallel timestamp and `hardware_invalid` arrays so they stay aligned.

If the cleaned DataFrame is empty (all rows were outliers), the step fails with an error and the pipeline exits gracefully for this session.

### Step 3 — EEG power band extraction (`compute_eeg_power_bands`)

Welch's periodogram is applied in a **sliding-window** fashion over the preprocessed signal.

| Parameter | Value |
|-----------|-------|
| Window duration | 2 s |
| Window overlap | 75% |
| Step size | 0.5 s (= window × (1 − 0.75)) |
| Welch segment length (`nperseg`) | min(256, window_samples) |
| Frequency bands | delta: 0–4 Hz, theta: 4–8 Hz, alpha: 8–13 Hz, beta: 13–22 Hz, gamma: 30–50 Hz |

For each window:

- The representative **timestamp** is the **median** of the hardware timestamps of the samples in that window. This correctly tracks hardware clock drift without assuming a perfectly constant sample rate.
- If any sample in the window was flagged as hardware-invalid (NaN or gap-adjacent), the entire window is emitted as a **NaN row** for every channel and band, preserving the timeline without computing spurious power estimates.

The output is a long-format DataFrame with columns `[Timestamp, channel, band, power]`, saved to `features/cognition/powerBands.csv`.

### Step 4 — Engagement index computation (`compute_engagement_indexes`)

The power DataFrame is pivoted to `(Timestamp × channel) × band` and the per-channel engagement index is computed:

```
engagement = beta / (alpha + theta + 1e-8)
```

The four cognitive metrics are then derived:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `Intertemporal` | `engagement(Temporal)` | Temporal-lobe engagement |
| `Lateralization` | `log(clip(LeftHemi, 1e-12, ∞)) − log(clip(RightHemi, 1e-12, ∞))` | Hemispheric engagement asymmetry (positive = left-dominant) |
| `Frontal` | `mean(engagement(AF7), engagement(AF8))` | Frontal cortex engagement |
| `Engagement` | `mean(engagement(AF7), engagement(AF8), engagement(Temporal))` | Overall engagement index |

`NaN` windows from the previous step propagate through the arithmetic and appear as `NaN` in all four metrics.

### Step 5 — Resampling to 2 Hz (`_simple_resample`)

The irregular per-window timestamps (spaced ~0.5 s but jittered by hardware clock drift) are snapped onto a strict 0.5 s grid via `numpy.interp`. Output points that fall entirely within a `NaN` gap — i.e., whose two nearest valid source timestamps are more than one output step apart — are set to `NaN` rather than being silently interpolated through.

---

## 4. Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/Cognition.csv` exists | Entire step skipped |
| `features/cognition/powerBands.csv` exists but `Cognition.csv` does not | Band powers loaded from disk; Steps 1–3 are skipped; Steps 4–5 run |
| Neither file exists | Full pipeline from Step 1 |

---

## 5. Outputs

### `results/Cognition.csv`

| Column | Unit / range | Description |
|--------|-------------|-------------|
| `Timestamp` | seconds | Seconds from recording start, 0.5 s steps |
| `Engagement` | dimensionless ratio | Mean frontal + temporal beta/(alpha+theta) |
| `Intertemporal` | dimensionless ratio | Temporal-channel engagement |
| `Lateralization` | log-ratio | Left–right hemispheric engagement log-ratio |
| `Frontal` | dimensionless ratio | Mean frontal (AF7, AF8) engagement |

### `features/cognition/powerBands.csv`

| Column | Description |
|--------|-------------|
| `Timestamp` | Window median hardware timestamp (s) |
| `channel` | EEG channel name |
| `band` | Frequency band (delta / theta / alpha / beta / gamma) |
| `power` | Spectral power (µV²), or `NaN` for hardware-invalid windows |

---

## 6. Error Handling

All major sub-steps return `None` on failure and log a descriptive `ERROR` message. The top-level `computeCognitiveIndexes` catches any unhandled exception, logs it with a full traceback, and returns without writing output files. The orchestrator then continues to the next pipeline step.

---

## 7. Key Constants

| Constant | Value | Location |
|----------|-------|---------|
| `GAP_THRESHOLD_S` | 5.0 s | `cognition_processor.py` |
| `HermesDataInterface.SAMPLING_RATE` | 250 Hz | `data_interface.py` |
| `bandpass_filter` passband | 0.5–45 Hz | `cognition_processor.py` |
| Notch frequency | 60 Hz | `cognition_processor.py` |
| Outlier z-score threshold | 3 | `cognition_processor.py` |
| Welch window | 2 s | `compute_eeg_power_bands` |
| Welch overlap | 75% | `compute_eeg_power_bands` |
| Output timebase | 0.5 s (2 Hz) | `_simple_resample` |

---

*© RE-AK Technologies Inc.*
