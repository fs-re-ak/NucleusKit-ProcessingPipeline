# Cognition Pipeline — EEG-based Cognitive Indexes

**Module:** `nucleuskit_pipeline/hermes/processor/cognition_processor.py`  
**Entry point:** `computeCognitiveIndexes(recpath)`  
**Primary output:** `results/Cognition.csv`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

---

## 1. Purpose

The cognition pipeline transforms raw EEG signals from the Hermes device into five
cognitive metrics sampled at 2 Hz. The primary metrics (Engagement, Focus,
CognitiveLoad) are derived exclusively from the temporal electrodes (T9 and T10),
following the cleaned pipeline specification in `instructions/EEG_PIPELINE.md`.
Secondary metrics (Frontal, Lateralization) come from the frontal and hemispheric
channels and run on the same cleaned signal.

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

### Step 3 — Bandpass + notch filtering (`bandpass_filter`)

| Parameter | Value |
|-----------|-------|
| Filter type | 4th-order Butterworth |
| Passband | 0.5–45 Hz |
| Notch | 60 Hz (Q = 30) |
| Implementation | Zero-phase `filtfilt` |

Any `NaN` or `Inf` values in a channel are linearly interpolated before filtering.
The filtered array retains the same column names as the input.

**Note:** Statistical z-score outlier removal (which dropped entire rows) has been
replaced by the epoch-level artefact rejection in Step 4, preserving timeline
alignment.

### Step 4 — Temporal artefact rejection (`reject_temporal_artefacts`)

Artefact detection operates on **1-second non-overlapping epochs** (250 samples) on
**T9 and T10 only**. An epoch is rejected if *either* channel fails *any* of:

| Criterion | Threshold | Signal |
|-----------|-----------|--------|
| NaN fraction (from `hardware_invalid`) | > 20 % | Raw (pre-filter) |
| Peak absolute amplitude | > 75 µV | Filtered |
| Peak-to-peak | > 60 µV | Filtered |
| P(30–45 Hz) / P(1–45 Hz) via Welch | > 0.30 | Filtered (EMG) |

Rejected epochs set the corresponding samples to `True` in `artefact_mask`.
The final invalid mask combines hardware invalids and artefact-rejected samples:

```python
combined_invalid = hardware_invalid | artefact_mask
```

At the 2-second analysis window level, any window where more than **25 %** of
samples are flagged in `combined_invalid` is emitted as an **all-NaN power row**,
preserving the timeline structure.

Rejection statistics are saved to `features/cognition/artefactStats.csv`.

### Step 5 — EEG power band extraction (`compute_eeg_power_bands`)

Welch's periodogram is applied in a sliding-window fashion over the filtered signal.

| Parameter | Value |
|-----------|-------|
| Window duration | 2 s |
| Window overlap | 75 % |
| Step size | 0.5 s |
| Welch segment length (`nperseg`) | min(256, window_samples) |
| Frequency bands | delta: 0–4 Hz, theta: 4–8 Hz, alpha: 8–13 Hz, beta: 13–22 Hz, gamma: 30–50 Hz |

For each window, the **representative timestamp** is the median of the hardware
timestamps in that window, tracking hardware clock drift robustly.

Output: long-format DataFrame `[Timestamp, channel, band, power]` saved to
`features/cognition/powerBands.csv`.

### Step 6 — Cognitive index computation (`compute_cognitive_indexes`)

#### Primary metrics — bilateral temporal averages (T9 + T10)

For each timestamp the T9 and T10 band powers are averaged:

```
avg_alpha = mean(T9_alpha, T10_alpha)
avg_beta  = mean(T9_beta,  T10_beta)
avg_theta = mean(T9_theta, T10_theta)
avg_delta = mean(T9_delta, T10_delta)
avg_gamma = mean(T9_gamma, T10_gamma)
```

| Metric | Formula | Reference |
|--------|---------|-----------|
| `Engagement` | `avg_beta / (avg_alpha + avg_theta)` | Pope et al. (1995) |
| `Focus` | `avg_beta / avg_alpha` | — |
| `CognitiveLoad` | `(avg_theta × avg_beta) / avg_alpha²` | Borghini et al. (2014) |

#### Secondary metrics — frontal and hemispheric channels

```
Frontal       = mean(β/(α+θ) for AF7,    β/(α+θ) for AF8)
Lateralization = log(clip(LeftHemi_eng, 1e-12)) − log(clip(RightHemi_eng, 1e-12))
```

where `LeftHemi_eng` and `RightHemi_eng` are the per-channel engagement ratios
`β / (α + θ)` on the raw hemispheric channels.

`NaN` windows from Step 5 propagate through all arithmetic and appear as `NaN`
in every metric.

Bilateral temporal band averages and relative powers are saved separately to
`features/cognition/temporalBandPowers.csv` (not included in `Cognition.csv`).

### Step 7 — Resampling to 2 Hz (`_simple_resample`)

The per-window timestamps (≈ 0.5 s steps, jittered by hardware clock drift) are
snapped onto a strict 0.5 s grid via `numpy.interp`. Output points that fall
entirely within a `NaN` gap are set to `NaN` rather than being interpolated through.

---

## 4. Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/Cognition.csv` exists | Entire step skipped |
| `features/cognition/powerBands.csv` exists and contains T9/T10 channels | Band powers loaded from cache; Steps 1–5 skipped |
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
| `Focus` | dimensionless ratio | Bilateral temporal β / α |
| `CognitiveLoad` | dimensionless ratio | Bilateral temporal (θ × β) / α² |
| `Frontal` | dimensionless ratio | Mean frontal (AF7, AF8) β / (α + θ) |
| `Lateralization` | log-ratio | log(LeftHemi_eng) − log(RightHemi_eng) |

### `features/cognition/powerBands.csv`

| Column | Description |
|--------|-------------|
| `Timestamp` | Window median hardware timestamp (s) |
| `channel` | EEG channel (AF7, AF8, T9, T10, LeftHemi, RightHemi) |
| `band` | Frequency band (delta / theta / alpha / beta / gamma) |
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
| `n_rejected_by_emg_ratio` | Rejections triggered by EMG power ratio |
| `n_samples_flagged` | Total samples in rejected epochs |
| `pct_samples_flagged` | Percentage of total samples flagged |

### `features/cognition/temporalBandPowers.csv`

Bilateral temporal band averages and relative powers (features only, not in
`Cognition.csv`):

`Timestamp, avg_beta, avg_alpha, avg_theta, avg_delta, avg_gamma, all_power,
rel_beta, rel_alpha, rel_theta, rel_delta, rel_gamma`

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
| Bandpass passband | 0.5–45 Hz | `bandpass_filter` |
| Notch frequency | 60 Hz | `bandpass_filter` |
| Artefact epoch length | 1 s (250 samples) | `reject_temporal_artefacts` |
| NaN fraction threshold | 20 % | `reject_temporal_artefacts` |
| Peak amplitude threshold | 75 µV | `reject_temporal_artefacts` |
| Peak-to-peak threshold | 60 µV | `reject_temporal_artefacts` |
| EMG ratio threshold | 0.30 | `reject_temporal_artefacts` |
| Window invalidation threshold | > 25 % flagged samples | `compute_eeg_power_bands` |
| Welch window | 2 s | `compute_eeg_power_bands` |
| Welch overlap | 75 % | `compute_eeg_power_bands` |
| Output timebase | 0.5 s (2 Hz) | `_simple_resample` |

---

*© RE-AK Technologies Inc.*
