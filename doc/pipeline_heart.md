# Heart Dynamics Pipeline — Shimmer PPG Processing

**Module:** `nucleuskit_pipeline/shimmer/processor/heart.py`  
**Entry point:** `computeHeartDynamics(recPath)`  
**Primary output:** `results/HeartDynamics.csv`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

---

## 1. Purpose

The heart dynamics pipeline derives time-domain heart rate variability (HRV) metrics from the Shimmer photoplethysmography (PPG) channel. The output is a sliding-window HRV time-series at 2 Hz, suitable for direct alignment with all other pipeline outputs.

---

## 2. Input Data

**Source file** (first match wins):

| Filename | Location | PPG column |
|----------|----------|-----------|
| `shimmer.csv` | `rawData/` | column 5 |
| `rawShimmer_0.csv` | `rawData/` | column 5 |
| `ppg.tmp` | `rawData/` | column 1 |
| `ppg.csv` | `rawData/` | column 1 |

Column 0 in all files is the hardware timestamp in milliseconds, which is normalised to seconds from recording start by `normalise_timestamps_to_seconds`.

**Nominal sampling rate:** 51.2 Hz (effective rate may vary slightly due to hardware clock drift)

---

## 3. Processing Steps

### Step 1 — Timestamp-aware resampling (`resample_to_grid`)

The raw signal is projected from its irregular (effective ~48–52 Hz) time base onto a **strict 51.2 Hz nominal-rate grid** using linear interpolation against the original hardware timestamps.

- Hardware gaps ≥ **5 s** (`GAP_THRESHOLD_S`) are detected from the raw timestamp vector and preserved as `NaN` in the resampled output. These represent true sensor disconnects, not sampling jitter.
- A resampling report is saved to `features/ppg/ppg_resample_report.txt` documenting the effective input rate, number of added interpolated samples, and a list of all detected gaps.

**Outputs at this step:**  
`features/ppg/ppg_resampled.csv` (initial, gap-NaN only; overwritten after artifact rejection)

---

### Step 2 — Two-stage artifact rejection (`apply_ppg_artifact_rejection`)

PPG amplitude can vary legitimately between recording segments (e.g. when the sensor is repositioned). A single global threshold would incorrectly reject valid high-amplitude segments. The two-stage approach separates amplitude-related and spectral-quality failure modes.

#### Stage 1 — Spectral Quality Index (SQI)

Each sliding window of **8 s** (`SQI_WINDOW_S`) is scored by:

```
SQI = P_cardiac / P_physio
```

where `P_cardiac` is the power integral in the **0.5–4.0 Hz cardiac band** and `P_physio` is the power integral in the 0.1–10 Hz physiological band. Each sample inherits the **maximum** SQI score across all overlapping windows (50 % overlap), so a sample is only rejected if every window covering it scored below the threshold.

| Parameter | Value | Note |
|-----------|-------|------|
| `SQI_WINDOW_S` | 8.0 s | Gives ~0.125 Hz frequency resolution |
| `SQI_CARDIAC_LO` | 0.5 Hz | 30 bpm lower bound |
| `SQI_CARDIAC_HI` | 4.0 Hz | 240 bpm upper bound + 2nd harmonic |
| `SQI_THRESHOLD` | 0.30 | Min fraction of cardiac-band power |

Windows below the threshold are classified as non-PPG (broadband noise, motion baseline, ADC glitches, flat-line dropout) and their samples are blanked to `NaN`. This gate is **amplitude-invariant**: stable noise at any level is caught because it carries no cardiac-frequency power.

#### Stage 2 — Local MAD glitch detector

Within samples that survived Stage 1, a **rolling MAD-based z-score** is computed over a **60-second** window (`REJECT_WINDOW_S`). Using a local rather than global statistic means that valid PPG at a different amplitude regime is compared to its own neighbourhood, not penalised by a global baseline.

| Parameter | Value |
|-----------|-------|
| `REJECT_WINDOW_S` | 60 s |
| `Z_THRESHOLD` | 4.0 (MAD-z) |
| `REJECT_MARGIN_S` | 5 s blanked around each glitch centre |
| `REJECT_MAX_PCT` | 15 % — abort MAD rejection if it would exceed this fraction |

Sudden spikes and ADC clipping transients (|z| > 4) are flagged, expanded by a ±5 s margin via binary dilation, and blanked to `NaN`. If Stage-2 rejection would exceed 15 % of the total signal, it is aborted — this indicates that the MAD detector has found no genuine glitch and the signal has simply changed amplitude regime.

**After artifact rejection:**  
`features/ppg/ppg_resampled.csv` is **overwritten** with the cleaned signal (NaN at all rejected positions).  
`features/ppg/ppg_rejected.png` — three-colour rejection map (black = valid, orange = SQI-rejected, red = MAD-rejected).

---

### Step 3 — Amplitude normalisation for peak detection (`_normalize_ppg_for_detection`)

NeuroKit2's Elgendi peak-detection algorithm sets its threshold using a **global** mean. When a recording contains amplitude regimes of different levels, the high-amplitude segment dominates the global threshold and peaks in the low-amplitude segment are missed.

A **rolling z-score** normalisation with a **10-second** window (`NORMALIZE_WINDOW_S`) is applied before feeding the signal to NeuroKit2. Each sample is centred and scaled relative to its local neighbourhood, making Elgendi's global threshold effective across all amplitude regimes.

The normalised signal is:
- Used **only** for peak detection
- Saved to `features/ppg/ppg_normalized.csv` (NaN at gap/rejected positions) for visual inspection

---

### Step 4 — Upsampling and peak detection (`_detect_ppg_peaks`)

NeuroKit2's Elgendi algorithm requires ≥ 100 Hz for reliable peak detection. At the native 51.2 Hz, there are only ~44 samples per beat, causing missed peaks and inflated RMSSD.

The normalised signal is upsampled to **256 Hz** (`_UPSAMPLE_RATE`) via cubic interpolation before being passed to `nk.ppg_process`. Peaks detected inside hardware-gap or artifact-rejected regions are discarded. Peak indices are then mapped back to the 51.2 Hz domain.

---

### Step 5 — Physiological IBI filtering and missing-beat fill

#### IBI lower bound (`_filter_physiological_ibi`)

Consecutive peaks separated by less than **300 ms** (`_IBI_MIN_MS` ≈ 200 bpm) are considered duplicate detections. The second peak in each such pair is removed.

#### Adaptive missing-beat filler (`_fill_missing_beats`)

After rolling z-score normalisation the PPG diastolic notch can become prominent enough to be detected as a secondary peak, producing spurious short IBIs (~350–600 ms). These pass the physiological lower bound but bias the rolling median downward. The filler guards against this by only admitting IBIs ≥ `recent_median / ratio` into the history buffer.

For genuine missed beats (IBI > `1.5 × recent_median`), evenly spaced **synthetic peaks** are inserted. Gaps that:
- Overlap the combined gap mask (hardware gap or artifact-rejected region), OR
- Exceed `MISSING_BEAT_MAX_FILL_S` (20 s)

are left unfilled, as the absence of beats reflects legitimate missing signal. Synthetic peaks are flagged in `synth_mask` and rendered as dashed orange lines in `ppg_overview.png`.

| Parameter | Value |
|-----------|-------|
| `_IBI_MIN_MS` | 300 ms (~200 bpm) |
| `_IBI_MAX_MS` | 2000 ms (~30 bpm) |
| `MISSING_BEAT_HISTORY` | 30 beats |
| `MISSING_BEAT_RATIO` | 1.5× recent median |
| `MISSING_BEAT_MAX_FILL_S` | 20 s |
| `MISSING_BEAT_MIN_HISTORY` | 10 beats |

---

### Step 6 — Diagnostic figures

| File | Content |
|------|---------|
| `features/ppg/ppg_overview.png` | NK2-cleaned PPG at 256 Hz; red lines = real peaks; dashed orange = synthetic fills |
| `features/ppg/ppg_spectrogram.png` | STFT spectrogram (0–6 Hz, 8 s window, 87.5 % overlap); white dashed lines mark the cardiac band; translucent red bands mark rejected/gap regions |
| `features/ppg/ppg_rejected.png` | Three-colour rejection map from Stage 1 (orange) and Stage 2 (red) |

---

### Step 7 — Sliding-window HRV computation (`_sliding_hrv_dataframe`)

A sliding window of **30 seconds** at a **2 Hz** output rate (0.5 s steps) computes standard time-domain HRV metrics from the peak-to-peak intervals (IBIs).

Any HRV window whose time span overlaps a sample in the combined gap mask (`gap_mask | reject_mask`) has **all metrics forced to `NaN`**. This prevents a boundary artefact where the long inter-peak interval spanning a gap inflates SDNN and other HRV metrics.

A minimum of **10 beats** is required within the window for metrics to be computed.

| Output metric | Formula |
|--------------|---------|
| `mean_hr` | 60 000 / mean_nn (bpm) |
| `mean_nn` | Mean IBI (ms) |
| `sdnn` | Standard deviation of IBIs (ms) |
| `rmssd` | √ mean(ΔNN²) (ms) |
| `pnn50` | % successive IBI differences > 50 ms |
| `cvsd` | RMSSD / mean_nn (dimensionless) |
| `cvnn` | SDNN / mean_nn (dimensionless) |
| `n_beats` | Number of beats in the window |

---

## 4. Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/HeartDynamics.csv` exists | Entire step skipped |
| File absent | Full computation from raw PPG |

To force recomputation, delete `results/HeartDynamics.csv` and optionally the `features/ppg/` directory.

---

## 5. Outputs

### `results/HeartDynamics.csv`

| Column | Unit | Description |
|--------|------|-------------|
| `Timestamp` | s | Seconds from recording start, 0.5 s steps |
| `mean_hr` | bpm | Mean heart rate in the 30 s window |
| `mean_nn` | ms | Mean normal-to-normal interval |
| `sdnn` | ms | Standard deviation of NN intervals |
| `rmssd` | ms | Root mean square of successive differences |
| `pnn50` | % | Fraction of NN differences > 50 ms |
| `cvsd` | dimensionless | Coefficient of variation (RMSSD / mean_NN) |
| `cvnn` | dimensionless | Coefficient of variation (SDNN / mean_NN) |
| `n_beats` | count | Number of beats used to compute the window |

`NaN` for windows with fewer than 10 beats or overlapping gap/rejected regions.

### Feature files in `features/ppg/`

| File | Description |
|------|-------------|
| `ppg_resampled.csv` | Timestamp + Value columns; artifact-cleaned; NaN in rejected/gap regions |
| `ppg_normalized.csv` | Timestamp + Value; rolling z-score; NaN in rejected/gap regions |
| `ppg_resample_report.txt` | Text summary of resampling statistics and gap list |
| `ppg_rejected.png` | Three-colour artifact rejection map |
| `ppg_overview.png` | Cleaned PPG waveform with annotated real and synthetic peaks |
| `ppg_spectrogram.png` | STFT spectrogram with cardiac-band markers and rejection overlay |

---

## 6. Re-running Artifact Rejection Only (`fix_ppg_session`)

`nucleuskit_pipeline/shimmer/processor/ppg_fixer.py` provides `fix_ppg_session(recpath)` as a standalone utility that:

1. Loads the existing `ppg_resampled.csv` and its timestamps.
2. Re-applies `apply_ppg_artifact_rejection` with the current constants.
3. Overwrites `ppg_resampled.csv` with the newly cleaned signal.
4. Re-runs peak detection and HRV computation, and overwrites `HeartDynamics.csv`.

This allows re-tuning the artifact rejection constants (e.g. `SQI_THRESHOLD`, `Z_THRESHOLD`) without rerunning the full pipeline.

---

## 7. Key Constants Summary

| Constant | Value | Description |
|----------|-------|-------------|
| `SAMPLING_RATE` | 51.2 Hz | Native Shimmer PPG rate |
| `GAP_THRESHOLD_S` | 5.0 s | Minimum gap duration flagged as disconnect |
| `SQI_WINDOW_S` | 8.0 s | SQI analysis window |
| `SQI_CARDIAC_LO` | 0.5 Hz | Cardiac band lower bound |
| `SQI_CARDIAC_HI` | 4.0 Hz | Cardiac band upper bound |
| `SQI_THRESHOLD` | 0.30 | Min cardiac-band power fraction |
| `Z_THRESHOLD` | 4.0 | MAD-z glitch detection threshold |
| `REJECT_MARGIN_S` | 5.0 s | Blanking margin around glitch |
| `REJECT_MAX_PCT` | 15 % | MAD-rejection safety cap |
| `REJECT_WINDOW_S` | 60.0 s | Rolling MAD window |
| `_UPSAMPLE_RATE` | 256 Hz | Rate for NeuroKit2 peak detection |
| `NORMALIZE_WINDOW_S` | 10.0 s | Rolling z-score window for NK2 input |
| `_IBI_MIN_MS` | 300 ms | Physiological lower bound (~200 bpm) |
| `_IBI_MAX_MS` | 2000 ms | Physiological upper bound (~30 bpm) |
| `MISSING_BEAT_RATIO` | 1.5 | Long-gap threshold multiplier |
| `MISSING_BEAT_MAX_FILL_S` | 20.0 s | Maximum gap to attempt synthetic fill |
| HRV sliding window | 30.0 s | Width of HRV analysis window |
| HRV minimum beats | 10 | Minimum peaks required per window |
| Output timebase | 0.5 s (2 Hz) | Step of HRV output series |

---

*© RE-AK Technologies Inc.*
