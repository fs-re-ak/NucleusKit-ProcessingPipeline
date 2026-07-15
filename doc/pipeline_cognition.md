# Cognition Pipeline — EEG-based Cognitive Indexes

**Module:** `nucleuskit_toolkit/hermes/processor/cognition_processor.py`  
**Entry point:** `computeCognitiveIndexes(recpath, apply_nlms=True)`  
**Primary output:** `results/Cognition.csv`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

---

## 1. Purpose

The cognition pipeline transforms raw EEG signals from the Hermes device into a cognitive engagement metric sampled at 2 Hz. The primary metric (Engagement) is derived exclusively from the temporal electrodes (T9 and T10), following the cleaned pipeline specification in `instructions/EEG_PIPELINE.md`.

> **Deferred metrics:** Focus, CognitiveLoad, Frontal, and Lateralization formulas are implemented in `compute_cognitive_indexes()` but are not emitted in the current output. Their band-power inputs are computed and archived in `features/cognition/temporalBandPowers.csv` for future use.

---

## 2. Input Data

**Source file** (first match wins):

| Filename | Location |
|----------|----------|
| `rawEEG_0.csv` | `rawData/` |
| `eeg.tmp` | `rawData/` |
| `eeg.csv` | `rawData/` |
| `eegRec_0.csv` | `rawData/` |

**Format:** Headerless or single-header CSV. Column 0 is the hardware timestamp
(milliseconds, auto-normalised to seconds). Columns 1–8 are the eight EXG channels
in the order defined by `HermesConstants`.

**Sampling rate:** 250 Hz

**Channel derivation** — performed by `HermesDataInterface.getEEG()` using a
midpoint re-reference against the T9/T10 differential (EAR_R, raw col 5):

| Output column | Formula | Rationale |
|---------------|---------|-----------|
| `AF7` | `col2 − col5 / 2` | Left frontal, half-referenced |
| `AF8` | `col1 − col5 / 2` | Right frontal, half-referenced |
| `T9` | `−col5 / 2` | Left temporal (reference electrode mirrored) |
| `T10` | `col5 / 2` | Right temporal (sensing electrode) |
| `LeftHemi` | `col2` | Raw left hemisphere |
| `RightHemi` | `col1 − col5` | Right hemisphere differential |

The Hermes hardware reference is on T9 (left ear); T10 (right ear / EAR_R) is the
sensing electrode. The raw differential `col5 ≈ T10 − T9`, so after the midpoint
split `T9 = −col5/2` and `T10 = col5/2`.

Samples whose absolute value is within 0.1 of ±187 500 are hardware-disconnected
and are replaced with `NaN` before any processing.

---

## 3. Processing Steps

### Step 1 — Hardware gap detection

The hardware timestamp vector is scanned for inter-sample intervals ≥ 5 s
(`GAP_THRESHOLD_S`). Each sample immediately following such a gap is flagged in a
boolean `gap_sample_mask`.

If hardware timestamps are unavailable, a synthetic array (`index / fs`) is used
and gap detection is skipped.

### Step 2 — Hardware-invalid mask

Before any interpolation or filtering, the per-sample invalid state is captured:

```python
hardware_invalid = eegData.isna().any(axis=1).to_numpy() | gap_sample_mask
```

This mask is used both for artefact rejection (NaN fraction criterion) and for
power-band window invalidation.

### Step 3 — Optional NLMS adaptive decorrelation

When `apply_nlms=True` (the default), a `CausalNLMSFilter` is applied to the raw
8-channel EEG array before the data interface returns it. This causal adaptive
filter decorrelates cross-channel EMG crosstalk in a streaming-compatible manner and
is particularly effective during high-amplitude muscle bursts at T9/T10.

To disable, pass `apply_nlms=False` to `computeCognitiveIndexes`.

### Step 4 — Bandpass + notch filtering (`bandpass_filter`)

| Parameter | Value |
|-----------|-------|
| Filter type | 4th-order Butterworth |
| Passband | 0.5–30 Hz |
| Notch | 60 Hz (Q = 30) — inactive at this highcut |
| Implementation | Zero-phase `filtfilt` |

The 30 Hz upper cutoff was chosen to cover delta, theta, alpha, and beta bands
while excluding the 30–45 Hz high-beta / gamma range dominated by temporal muscle
(EMG) noise at T9/T10. Because the 60 Hz notch frequency lies above the passband,
the notch has no effect and is effectively disabled.

Any `NaN` or `Inf` values in a channel are linearly interpolated before filtering.
The filtered array retains the same column names as the input.

**Note:** Statistical z-score outlier removal (which dropped entire rows) has been
replaced by the epoch-level artefact rejection in Step 5, preserving timeline
alignment.

### Step 5 — Temporal artefact rejection (`reject_temporal_artefacts`)

Artefact detection operates on **1-second non-overlapping epochs** (250 samples) on
**T9 and T10 only**. An epoch is rejected if *either* channel fails *any* of:

| Criterion | Threshold | Signal |
|-----------|-----------|--------|
| NaN fraction (from `hardware_invalid`) | > 20 % | Raw (pre-filter) |
| Peak absolute amplitude | > 500 µV | Filtered |
| Peak-to-peak | > 500 µV | Filtered |

The EMG power-ratio criterion (formerly Criterion 4) was removed because the 30 Hz
bandpass upper cutoff already excludes the 30–45 Hz muscle noise band from the
signal, making the ratio uninformative.

All signal metrics are measured for every epoch regardless of pass/fail, so that
the per-epoch report can be used to calibrate thresholds.

Rejected epochs set the corresponding samples to `True` in `artefact_mask`.
The final invalid mask combines hardware invalids and artefact-rejected samples:

```python
combined_invalid = hardware_invalid | artefact_mask
```

At the 2-second analysis window level, any window where more than **25 %** of
samples are flagged in `combined_invalid` is emitted as an **all-NaN power row**,
preserving the timeline structure.

Rejection statistics are saved to `features/cognition/artefactStats.csv` and
per-epoch metrics to `features/cognition/epochMetrics.csv`. A waveform plot with
artefact regions highlighted is saved to `features/cognition/eegArtefactPlot.png`.

### Step 6 — EEG power band extraction (`compute_eeg_power_bands`)

Power bands are computed on **T9 and T10 only** using Welch's periodogram in a
sliding-window fashion.

| Parameter | Value |
|-----------|-------|
| Window duration | 2 s |
| Window overlap | 75 % |
| Step size | 0.5 s |
| Welch segment length (`nperseg`) | min(256, window_samples) |
| Frequency bands | delta: 0–4 Hz, theta: 4–8 Hz, alpha: 8–13 Hz, beta: 13–22 Hz |

For each window, the **representative timestamp** is the median of the hardware
timestamps in that window, tracking hardware clock drift robustly.

Output: long-format DataFrame `[Timestamp, channel, band, power]` saved to
`features/cognition/powerBands.csv`.

### Step 7 — Cognitive index computation (`compute_cognitive_indexes`)

#### Primary metric — bilateral temporal average (T9 + T10)

For each timestamp the T9 and T10 band powers are averaged:

```
avg_alpha = mean(T9_alpha, T10_alpha)
avg_beta  = mean(T9_beta,  T10_beta)
avg_theta = mean(T9_theta, T10_theta)
avg_delta = mean(T9_delta, T10_delta)
```

| Metric | Formula | Reference |
|--------|---------|-----------|
| `Engagement` | `avg_beta / (avg_alpha + avg_theta)` | Pope et al. (1995) |

`NaN` windows from Step 6 propagate through all arithmetic and appear as `NaN`
in every metric.

Bilateral temporal band averages and relative powers are saved separately to
`features/cognition/temporalBandPowers.csv` (not included in `Cognition.csv`).

### Step 8 — Resampling to 2 Hz (`_simple_resample`)

The per-window timestamps (≈ 0.5 s steps, jittered by hardware clock drift) are
snapped onto a strict 0.5 s grid via `numpy.interp`. Output points that fall
entirely within a `NaN` gap are set to `NaN` rather than being interpolated through.

---

## 4. Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/Cognition.csv` exists | Entire step skipped |
| `features/cognition/powerBands.csv` exists and contains T9/T10 channels | Band powers loaded from cache; Steps 1–6 skipped |
| `powerBands.csv` exists but uses old channel layout (missing T9/T10) | Cache invalidated; full recomputation |
| Neither file exists | Full pipeline from Step 1 |

**Cache invalidation after upgrade:** Sessions processed by an older version of the
pipeline will have `powerBands.csv` with a `Temporal` channel instead of `T9`/`T10`.
The pipeline detects this and automatically recomputes — simply delete the existing
`results/Cognition.csv` to trigger a re-run.

---

## 5. Outputs

### `results/Cognition.csv`

| Column | Unit / range | Description |
|--------|-------------|-------------|
| `Timestamp` | seconds | Seconds from recording start, 0.5 s steps |
| `Engagement` | dimensionless ratio | Bilateral temporal β / (α + θ) |

### `features/cognition/powerBands.csv`

| Column | Description |
|--------|-------------|
| `Timestamp` | Window median hardware timestamp (s) |
| `channel` | EEG channel (T9, T10) |
| `band` | Frequency band (delta / theta / alpha / beta) |
| `power` | Spectral power (µV²), or `NaN` for invalidated windows |

### `features/cognition/artefactStats.csv`

Single-row summary of the epoch-level artefact rejection:

| Column | Description |
|--------|-------------|
| `n_epochs_total` | Total 1-second epochs analysed |
| `n_epochs_rejected` | Epochs rejected by any criterion |
| `pct_epochs_rejected` | Percentage rejected |
| `n_rejected_by_nan` | Rejections triggered by NaN fraction |
| `n_rejected_by_peak_amp` | Rejections triggered by peak amplitude |
| `n_rejected_by_ptp` | Rejections triggered by peak-to-peak |
| `n_samples_flagged` | Total samples in rejected epochs |
| `pct_samples_flagged` | Percentage of total samples flagged |

### `features/cognition/epochMetrics.csv`

Per-epoch metrics for threshold calibration — one row per 1-second epoch:

| Column | Description |
|--------|-------------|
| `epoch_idx` | Epoch index (0-based) |
| `time_s` | Start time of epoch (seconds) |
| `nan_frac` | Fraction of hardware-invalid samples |
| `T9_peak_amp` | Peak absolute amplitude on T9 (µV) |
| `T10_peak_amp` | Peak absolute amplitude on T10 (µV) |
| `T9_ptp` | Peak-to-peak on T9 (µV) |
| `T10_ptp` | Peak-to-peak on T10 (µV) |
| `rejected` | Whether the epoch was rejected (bool) |
| `cause` | First criterion that triggered rejection (`nan` / `peak_amp` / `ptp`) |
| `rejecting_channel` | Channel that caused rejection |

### `features/cognition/temporalBandPowers.csv`

Bilateral temporal band averages and relative powers (features only, not in
`Cognition.csv`):

`Timestamp, avg_beta, avg_alpha, avg_theta, avg_delta, all_power,
rel_beta, rel_alpha, rel_theta, rel_delta`

### `features/cognition/eegArtefactPlot.png`

Waveform plot of the filtered EEG with artefact-rejected regions highlighted in
translucent red. Saved only if it does not already exist (re-delete to regenerate).

---

## 6. Error Handling

All major sub-steps return `None` on failure and log a descriptive `ERROR` message.
The top-level `computeCognitiveIndexes` catches any unhandled exception, logs it
with a full traceback, and returns without writing output files. The orchestrator
continues to the next pipeline step.

---

## 7. Key Constants

| Constant | Value | Location |
|----------|-------|---------|
| `GAP_THRESHOLD_S` | 5.0 s | `cognition_processor.py` |
| `HermesDataInterface.SAMPLING_RATE` | 250 Hz | `data_interface.py` |
| Bandpass passband | 0.5–30 Hz | `bandpass_filter` |
| Notch frequency | 60 Hz (inactive) | `bandpass_filter` |
| Artefact epoch length | 1 s (250 samples) | `reject_temporal_artefacts` |
| NaN fraction threshold | 20 % | `reject_temporal_artefacts` |
| Peak amplitude threshold | 500 µV | `reject_temporal_artefacts` |
| Peak-to-peak threshold | 500 µV | `reject_temporal_artefacts` |
| Window invalidation threshold | > 25 % flagged samples | `compute_eeg_power_bands` |
| Welch window | 2 s | `compute_eeg_power_bands` |
| Welch overlap | 75 % | `compute_eeg_power_bands` |
| Output timebase | 0.5 s (2 Hz) | `_simple_resample` |

---

*© RE-AK Technologies Inc.*
