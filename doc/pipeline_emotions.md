# Emotions Pipeline — EMG-based Emotion Classification

**Module:** `nucleuskit_pipeline/hermes/processor/emotions_processor.py`  
**Entry point:** `computeEmotions(recpath)`  
**Primary output:** `results/Emotions.csv`  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter 2026

---

## 1. Purpose

The emotions pipeline infers moment-to-moment facial emotion probabilities from 8-channel surface EMG recorded by the Hermes device. Classification is performed using a bundled two-stage machine-learning model. The output is a set of per-emotion probability time-series sampled at 2 Hz.

---

## 2. Input Data

**Source file** (first match wins):

| Filename | Location |
|----------|----------|
| `rawEEG_0.csv` | `rawData/` |
| `eeg.tmp` | `rawData/` |
| `eeg.csv` | `rawData/` |
| `eegRec_0.csv` | `rawData/` |

**Format:** Headerless or single-header CSV. Column 0 is the hardware timestamp (milliseconds, auto-normalised). Columns 1–8 are the eight EMG channels, stored in the order defined by `HermesConstants`.

**Sampling rate:** 250 Hz

**Channel order** (feature-columns.json order used by the classifier):

| Index (0-based) | Channel |
|-----------------|---------|
| 0 | AF8 |
| 1 | AF7 |
| 2 | CHEEK_R |
| 3 | CHEEK_L |
| 4 | EAR_R |
| 5 | AFz |
| 6 | BROW_L |
| 7 | NOSE |

---

## 3. Hardware Disconnect Invalidation

Before any processing, samples whose absolute value is within 1 of the hardware saturation value **187 500** or within 1 of **0** are replaced with `NaN`. These represent electrode disconnects or ADC-level saturation events.

```python
DISCONNECT_VALUE = 187500.0
NULL_WINDOW_NAN_THRESHOLD = 0.10   # 10 % of samples NaN => null window
```

---

## 4. Classifier Architecture

The bundled model lives in `hermes/classifier/weights/`. It is a **two-stage discriminant**:

| Stage | Model | Role |
|-------|-------|------|
| Stage 1 — Artefact gate | Binary classifier | Distinguishes genuine facial EMG from movement/noise artefacts |
| Stage 2a — Neutral gate | LDA | Separates neutral from emotional states |
| Stage 2b — Emotion classifier | LDA / KNN | Multi-class emotion probability estimate |

Model weights are loaded via `TwoStageClassifier.load(clf_dir)`. `cooldown_windows=0` disables the post-prediction cool-down so every window is classified independently.

**Feature vector** fed to the model (9 elements):

1. L2-normalised per-channel RMS values for each of the 8 EMG channels.
2. `AVG_RMS`: the un-normalised mean across all 8 channels.

---

## 5. Streaming Classification (`StreamingEMGClassifier`)

The primary computation path uses a sample-by-sample streaming classifier:

| Parameter | Value |
|-----------|-------|
| Window length | 1.0 s (= 250 samples at 250 Hz) |
| Step size | 0.5 s (= 125 samples) |
| Sampling rate | 250 Hz |
| Cooldown windows | 0 (disabled for offline batch) |

The classifier maintains an internal ring buffer. Every time the buffer advances by one step (`step_samples = 125`), it emits a full window result: emotion probabilities, per-channel RMS, and the model input feature vector.

### Per-window timestamp

The representative timestamp of each output window is the **median** of the hardware timestamps buffered during that window. This is robust to hardware clock drift. When hardware timestamps are unavailable (file read failure), the classifier's sample-count clock is used as a fallback.

### Null window detection

If more than `NULL_WINDOW_NAN_THRESHOLD` (10 %) of the samples in the current window buffer were `NaN` (hardware-disconnected), the window is flagged as a **null window** and all emotion columns are emitted as `NaN`. This prevents spurious classifications on disconnected epochs.

---

## 6. Processing Steps

### Step 1 — Load EMG

`loadEXG(recpath, re_reference=False)` reads the raw EXG file, normalises timestamps to seconds from start, and returns a 2-D NumPy array of shape `(n_samples, 8)`. Re-referencing is disabled for the emotions path (it is used only for EEG-band cognition).

### Step 2 — Invalidate hardware artifacts

`_invalidate_invalid_eeg_samples(eeg_data)` sets disconnect-value and zero-value samples to `NaN` in-place.

### Step 3 — Sample-by-sample streaming classification

For each sample:

1. Append a boolean `nan_flag` to the NaN-flag deque and the hardware timestamp to the timestamp deque (both sized to `window_samples`).
2. Replace `NaN` with `0.0` (via `np.nan_to_num`) before pushing the sample into the classifier's internal buffer.
3. When the classifier emits a result (every 125 samples):
   - Compute the null-window flag from the NaN-flag deque.
   - If null: append `NaN` row to emotion, RMS, and model-input lists.
   - If valid: append classification probabilities, per-channel RMS, and L2-normalised feature vector.

### Step 4 — Resample to 2 Hz

When hardware timestamps are available, the raw emotion DataFrame (timestamped at hardware-clock midpoints) is snapped to the shared 0.5 s grid via `_simple_resample`. `NaN` windows from Step 3 remain `NaN` after resampling; output bins whose two bracketing valid source timestamps are more than 1.0 s apart are also forced to `NaN`.

### Step 5 — Write outputs

Three files are written (see Section 7).

### Step 6 — Diagnostic report

`generate_report(recpath)` from `emotions_report.py` saves per-emotion jitter plots and a pie chart of the predicted label distribution.

---

## 7. Incremental Caching and RMS-driven Recomputation

| Condition | Behaviour |
|-----------|-----------|
| `Emotions.csv` **and** `emotionClassifierInputs.csv` **and** `rmsSignals.csv` all exist | Entire step skipped |
| `rmsSignals.csv` exists but emotion pair is missing | Run `_compute_emotions_from_rms_csv`: re-run classifier on cached RMS without loading raw EXG |
| Neither file exists | Full raw-EXG path |

The RMS-driven path (`_compute_emotions_from_rms_csv`) loads `rmsSignals.csv`, applies `normalize_rms_dataframe` (channel gain normalisation), and iterates over each row to call `TwoStageClassifier.infer` directly. This is significantly faster on re-runs and supports manual RMS edits (e.g. channel-fixer corrections) being propagated to the final emotion output without reprocessing the raw signal.

---

## 8. Outputs

### `results/Emotions.csv`

| Column | Unit / range | Description |
|--------|-------------|-------------|
| `Timestamp` | seconds | Seconds from recording start, 0.5 s steps |
| `Neutral` | 0–1 probability | — |
| `Happiness` | 0–1 probability | — |
| `Anger` | 0–1 probability | — |
| `Surprise` | 0–1 probability | — |
| `Contempt` | 0–1 probability | — |
| `Disgust` | 0–1 probability | — |
| `Fear` | 0–1 probability | — |
| `Sadness` | 0–1 probability | — |

`NaN` rows correspond to null windows (hardware disconnects exceeding the 10 % threshold). Written with `na_rep="NULL"`.

### `features/emotions/rmsSignals.csv`

One row per classifier window (0.5 s step). Columns: `Timestamp`, followed by `AF8`, `AF7`, `CHEEK_R`, `CHEEK_L`, `EAR_R`, `AFz`, `BROW_L`, `NOSE`. Values are raw (un-normalised) RMS in the hardware amplitude units. `NaN` rows correspond to null windows.

### `features/emotions/emotionClassifierInputs.csv`

One row per valid (non-null) classifier window. Columns:

- `Timestamp`
- All feature columns as defined by `clf.feature_columns` / `clf.model_feature_columns` (L2-normalised RMS + `AVG_RMS`)
- `PredictedLabel` (string emotion name)
- `PredictedConfidence` (float, 0–1)

This file is the primary traceability artefact linking raw features to classifier decisions.

---

## 9. Key Constants

| Constant | Value | Location |
|----------|-------|---------|
| `DISCONNECT_VALUE` | 187 500 | `emotions_processor.py` |
| `NULL_WINDOW_NAN_THRESHOLD` | 0.10 | `emotions_processor.py` |
| Window length | 1.0 s | `StreamingEMGClassifier` |
| Step size | 0.5 s | `StreamingEMGClassifier` |
| Sampling rate | 250 Hz | `StreamingEMGClassifier` |
| Output timebase | 0.5 s (2 Hz) | `_simple_resample` |

---

## 10. Classifier Weights Location

```
nucleuskit_pipeline/hermes/classifier/weights/
├── config.json
├── feature_columns.json
├── artefact_config.json
├── stage1_<model>.pkl        # Artefact gate
├── stage2_<model>.pkl        # Emotion classifier
└── stage2_cov.npy            # Covariance matrix for LDA
```

To retrain or update the model, replace these files and ensure `feature_columns.json` lists the columns in the exact order expected by the feature extraction step.

---

*© RE-AK Technologies Inc.*
