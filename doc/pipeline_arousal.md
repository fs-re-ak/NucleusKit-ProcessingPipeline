# Arousal Pipeline — Shimmer EDA / GSR Processing

**Module:** `nucleuskit_pipeline/shimmer/processor/eda.py`  
**Entry point:** `computeArousal(recPath)`  
**Primary output:** `results/Arousal.csv`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

---

## 1. Purpose

The arousal pipeline decomposes the raw electrodermal activity (EDA / galvanic skin response / GSR) signal from the Shimmer device into its tonic (slow baseline, skin conductance level — SCL) and phasic (fast sympathetic response, skin conductance response — SCR) components. The decomposition uses the `cvxEDA` convex optimisation algorithm (Greco et al., *IEEE Trans. Biomed. Eng.*, 2015). Discrete SCR events are also detected and reported. The output is a 2 Hz time-series suitable for alignment with all other pipeline outputs.

---

## 2. Input Data

**Source file** (first match wins):

| Filename | Location | EDA column |
|----------|----------|-----------|
| `shimmer.csv` | `rawData/` | column 4 |
| `rawShimmer_0.csv` | `rawData/` | column 4 |
| `gsr.tmp` | `rawData/` | column 1 |
| `gsr.csv` | `rawData/` | column 1 |

Column 0 is the hardware timestamp in milliseconds, normalised to seconds from recording start.  
`shimmer.csv` / `rawShimmer_0.csv` store the EDA value as **raw GSR resistance in kΩ**. The two-column legacy files (`gsr.tmp`, `gsr.csv`) store the EDA value directly.

**Nominal sampling rate:** 51.2 Hz

---

## 3. Processing Steps

### Step 1 — Timestamp-aware resampling (`resample_to_grid`)

Identical to the PPG pipeline: the raw signal is projected onto a **strict 51.2 Hz nominal-rate grid** using linear interpolation against the original hardware timestamps. Hardware gaps ≥ **5 s** are preserved as `NaN` (see `shimmer/processor/resampler.py`).

**Note:** Interpolation is performed in **resistance space (kΩ)**, where it is physically valid. Unit conversion to conductance follows immediately after, ensuring that the interpolated mid-gap values are never used in the analysis.

**Output:** `features/eda/eda_resampled.csv`, `features/eda/eda_resample_report.txt`

---

### Step 2 — Unit conversion: kΩ → µS

After resampling, the resistance signal is converted to **conductance**:

```
EDA_µS = 1000 / EDA_kΩ
```

All downstream processing (artifact rejection, cvxEDA, SCR detection) operates in **µS (microsiemens)**. This is the standard unit for EDA analysis because:
- Physiological responses produce **positive deflections** (rising conductance).
- The MAD artifact floor (0.05 µS) has a meaningful physiological interpretation.

`NaN` values from gap regions propagate correctly through the division.

---

### Step 3 — MAD-based artifact rejection (`reject_eda_artifacts`)

A **rolling MAD** (median absolute deviation) criterion is applied to the conductance signal to detect and linearly interpolate over transient artifacts.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ARTIFACT_WINDOW` | 5.0 s | Rolling window for local median and MAD |
| `ARTIFACT_MAD_THR` | 5.0 | Threshold in multiples of local MAD |
| MAD floor | 0.05 µS | Prevents flat segments from flagging all subsequent samples |
| Buffer expansion | 1 s | Outlier mask is expanded by 1 s via a rolling max |

Samples flagged as outliers are replaced by **linear interpolation** rather than NaN, so the cvxEDA solver receives a continuous signal. After interpolation, samples within hardware-gap regions (from `gap_mask`) are **restored to NaN** so true disconnects are never filled in.

---

### Step 4 — cvxEDA tonic/phasic decomposition (`_decompose_eda`)

#### Downsampling

Before decomposition, the 51.2 Hz artifact-rejected signal is downsampled to **4 Hz** (`CVXEDA_RATE`) using `scipy.signal.resample`. EDA is a very slow signal (SCR onset ~1–2 s, offset ~5–15 s); 4 Hz is the standard rate for EDA analysis and is the rate recommended in the cvxEDA reference paper.

`NaN` values are forward-filled before resampling so the signal is continuous for the solver. The gap mask is downsampled in parallel so gap regions can be restored to NaN in the output.

#### Z-score normalisation

The downsampled signal is z-score normalised before passing to cvxEDA:

```
EDA_norm = (EDA_4Hz − mean(EDA_4Hz)) / (std(EDA_4Hz) + 1e-12)
```

This is applied after artifact rejection, so the mean and std are not distorted by transient spikes.

#### cvxEDA decomposition

`cvxEDA(EDA_norm, 1.0 / CVXEDA_RATE)` decomposes the signal into:

- **Tonic component** (SCL): a slowly varying baseline reflecting slow changes in sweat gland activity and sympathetic arousal.
- **Phasic component** (SCR): a fast residual representing discrete sympathetic activation events.

The solver returns both components at 4 Hz. For technical details of the convex formulation, refer to: Greco, A. et al. (2015). "cvxEDA: A convex optimization approach to electrodermal activity processing." *IEEE Trans. Biomed. Eng.*, 63(4), 797–804.

#### Post-decomposition filtering

A 3rd-order Butterworth lowpass filter with a **0.5 Hz cutoff** (`PHASIC_CUTOFF`) is applied to the phasic component via zero-phase `filtfilt` to attenuate solver-induced high-frequency noise. The tonic component is not filtered.

Gap regions are restored to `NaN` in both outputs using the downsampled gap mask.

---

### Step 5 — SCR event detection (`_detect_scr_events`)

Discrete skin conductance responses are detected as **peaks in the 4 Hz phasic component** using `scipy.signal.find_peaks` before resampling back to the original rate.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `_SCR_MIN_HEIGHT` | 0.05 (z-score units) | Minimum peak amplitude |
| `_SCR_MIN_PROMINENCE` | 0.05 (z-score units) | Minimum peak prominence |
| `_SCR_MIN_DISTANCE_S` | 1.0 s | Minimum distance between consecutive SCR events |

Peaks that fall inside hardware-gap windows (flagged by the downsampled gap mask) are discarded. The function returns:
- SCR onset times in seconds from recording start
- Peak amplitudes (z-score units)
- Peak prominences (z-score units)

---

### Step 6 — Diagnostic figure

`_save_eda_figure` saves `features/eda/eda_overview.png`:

- Black line: z-scored artifact-rejected EDA signal (51.2 Hz, for visual fidelity).
- Blue line: tonic component (resampled back to 51.2 Hz).
- Red vertical lines: SCR event times.
- NaN regions appear as natural breaks in the trace.

---

### Step 7 — Downsample to 2 Hz and write outputs

The tonic and phasic arrays (both resampled to 51.2 Hz via `scipy.signal.resample`) are downsampled to the shared **2 Hz / 0.5 s timebase** using `pandas.Series.resample("500ms").mean()`. Each 0.5 s bin averages the ~26 original 51.2 Hz samples it contains. `NaN` samples (large gaps) produce `NaN` output bins automatically.

---

## 4. Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| `results/Arousal.csv` exists | Entire step skipped |
| File absent | Full computation from raw EDA |

---

## 5. Outputs

### `results/Arousal.csv`

| Column | Unit | Description |
|--------|------|-------------|
| `Timestamp` | s | Seconds from recording start, 0.5 s steps |
| `TonicEDA` | z-score (µS-derived) | Slow tonic EDA component (SCL proxy) |
| `PhasicEDA` | z-score (µS-derived) | Fast phasic EDA component (SCR envelope) |

`NaN` rows correspond to hardware-gap regions.

### Feature files in `features/eda/`

| File | Description |
|------|-------------|
| `eda_resampled.csv` | Timestamp + Value; 51.2 Hz grid; resistance in kΩ (before conductance conversion) |
| `eda_resample_report.txt` | Resampling statistics and gap list |
| `eda_overview.png` | EDA signal, tonic trace, and SCR event markers |
| `SCR_events.csv` | Timestamp (s), Amplitude (z-score), Prominence (z-score) for each detected SCR |

---

## 6. Key Constants Summary

| Constant | Value | Description |
|----------|-------|-------------|
| `SAMPLING_RATE` | 51.2 Hz | Native Shimmer EDA rate |
| `GAP_THRESHOLD_S` | 5.0 s | Minimum gap duration flagged as disconnect |
| `CVXEDA_RATE` | 4 Hz | Rate for cvxEDA decomposition |
| `PHASIC_CUTOFF` | 0.5 Hz | Lowpass cutoff for phasic post-filter |
| `ARTIFACT_WINDOW` | 5.0 s | Rolling MAD window |
| `ARTIFACT_MAD_THR` | 5.0 | MAD threshold multiplier |
| `_SCR_MIN_HEIGHT` | 0.05 (z-score) | SCR peak minimum amplitude |
| `_SCR_MIN_PROMINENCE` | 0.05 (z-score) | SCR peak minimum prominence |
| `_SCR_MIN_DISTANCE_S` | 1.0 s | Minimum SCR-to-SCR interval |
| Output timebase | 0.5 s (2 Hz) | Step of output series |

---

## 7. Reference

Greco, A., Valenza, G., Lanata, A., Scilingo, E. P., & Citi, L. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. *IEEE Transactions on Biomedical Engineering*, 63(4), 797–804. https://doi.org/10.1109/TBME.2015.2474131

---

*© RE-AK Technologies Inc.*
