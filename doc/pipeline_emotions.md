# Emotions Pipeline — EMG-based Emotion Classification

**Package:** `nucleuskit_pipeline/hermes/processor/emotions_processor/`
**Entry point:** `computeEmotions(recpath)`
**Primary output:** `results/Emotions.csv`
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter–Spring 2026

---

## 1. Purpose

The emotions pipeline infers moment-to-moment facial emotion probabilities from 8-channel surface EMG recorded by the Hermes device. Classification is performed using a pluggable two-stage machine-learning model. The output is a set of per-emotion probability time-series sampled at 2 Hz.

---

## 2. Package Layout

```
hermes/processor/emotions_processor/
├── __init__.py          # Public API: computeEmotions, EMOTION_COLUMNS
├── constants.py         # Shared constants (thresholds, column lists)
├── session.py           # computeEmotions() — orchestration & I/O
├── pipeline.py          # run_window_pipeline() — single classification loop
├── report.py            # Visual statistics report generator
├── interface/
│   ├── model.py         # EmotionModel ABC + EmotionWindowResult dataclass
│   ├── windows.py       # EmotionWindow dataclass + WindowSource Protocol
│   └── streaming.py     # StreamingWindowSource, RmsCsvWindowSource
└── models/
    ├── __init__.py      # Model registry: get_model(name)
    └── v12/
        ├── model.py     # V12EmotionModel (concrete EmotionModel)
        ├── inference.py # TwoStageClassifier
        ├── config.py    # HardwareConfig, ExperimentConfig
        └── weights/     # *.pkl, *.json — bundled V12 weights
```

---

## 3. Input Data

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

## 4. Hardware Disconnect Invalidation

Before any processing, samples whose absolute value is within 1 of the hardware saturation value **187 500** are replaced with `NaN`. These represent electrode disconnects or ADC-level saturation events.

```python
# constants.py
DISCONNECT_VALUE = 187500.0
NULL_WINDOW_NAN_THRESHOLD = 0.10   # 10 % of samples NaN => null window
```

---

## 5. Classifier Architecture

The bundled model lives in `models/v12/weights/`. It is a **two-stage discriminant**:

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

## 6. Unified Window Pipeline

All classification paths share a single loop in `pipeline.py`. Window *production* (filtering, windowing, RMS extraction) is separated from *inference* (model feature construction and classification):

```
WindowSource  ──────────────────────────────┐
  StreamingWindowSource (raw EMG)           │   run_window_pipeline()
  RmsCsvWindowSource (existing rmsSignals)  │   ─────────────────────
                                            └── for window in source:
                                                  if invalid → NaN rows
                                                  else → model.infer_from_rms(rms)
                                                         → emotion / RMS / input rows
```

### StreamingWindowSource

Accepts pre-loaded EMG `(N, 8)` + optional hardware timestamps. Internally applies a 4th-order Butterworth bandpass (15–45 Hz), maintains a 250-sample ring buffer, and emits one `EmotionWindow` per step (125 samples = 0.5 s). Per-window timestamp is the median of the hardware timestamps in the buffer; falls back to the sample-count clock when hardware timestamps are unavailable. Windows where more than `NULL_WINDOW_NAN_THRESHOLD` of samples were NaN are emitted as invalid windows.

### RmsCsvWindowSource

Reads an existing `rmsSignals.csv`, normalises column names via `hermes/rms_columns.py`, and yields one `EmotionWindow` per row. Rows with any NaN channel value are emitted as invalid windows.

### EmotionModel (pluggable)

`V12EmotionModel.infer_from_rms(channel_rms)` constructs the 9-element feature vector (L2 normalise + AVG_RMS) and calls `TwoStageClassifier.infer`, returning an `EmotionWindowResult`.

---

## 7. Processing Steps

### Step 1 — Load EMG (streaming path only)

`loadEXG(recpath, re_reference=False)` reads the raw EXG file, normalises timestamps to seconds from start, and returns a 2-D NumPy array of shape `(n_samples, 8)`.

### Step 2 — Invalidate hardware artefacts

`_invalidate_disconnected_samples(eeg_data)` sets disconnect-value samples to `NaN` in-place.

### Step 3 — Optional NLMS decorrelation

`CausalNLMSFilter.push_batch(eeg_data)` applies causal adaptive decorrelation (streaming path, enabled by default via `apply_nlms=True`).

### Step 4 — Window pipeline

`run_window_pipeline(source, model)` runs the unified loop described in Section 6.

### Step 5 — Resample to 2 Hz (streaming path with hardware timestamps)

The raw emotion DataFrame (timestamped at hardware-clock midpoints) is snapped to the shared 0.5 s grid via `_simple_resample`. `NaN` windows remain `NaN` after resampling.

### Step 6 — Write outputs

Three files are written (see Section 9).

### Step 7 — Diagnostic report

`generate_report(recpath)` saves per-emotion jitter plots and a pie chart of the predicted label distribution.

---

## 8. Incremental Caching and RMS-driven Recomputation

| Condition | Behaviour |
|-----------|-----------|
| All three output files exist | Entire step skipped |
| `rmsSignals.csv` exists but emotion pair is missing | Use `RmsCsvWindowSource`: re-run classifier on cached RMS without loading raw EXG |
| Neither file exists | Full raw-EXG path via `StreamingWindowSource` |

The RMS-driven path is significantly faster on re-runs and supports manual RMS edits (e.g. channel-fixer corrections) propagating to the final emotion output without reprocessing the raw signal.

---

## 9. Outputs

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

One row per classifier window (0.5 s step). Columns: `Timestamp`, followed by `AF8`, `AF7`, `CHEEK_R`, `CHEEK_L`, `EAR_R`, `AFz`, `BROW_L`, `NOSE`. Values are raw (un-normalised) RMS. `NaN` rows correspond to null windows.

### `features/emotions/emotionClassifierInputs.csv`

One row per valid (non-null) classifier window. Columns:

- `Timestamp`
- All feature columns as defined by `model.feature_columns` (L2-normalised RMS + `AVG_RMS`)
- `PredictedLabel` (string emotion name)
- `PredictedConfidence` (float, 0–1)

---

## 10. Key Constants

| Constant | Value | Location |
|----------|-------|---------|
| `DISCONNECT_VALUE` | 187 500 | `constants.py` |
| `NULL_WINDOW_NAN_THRESHOLD` | 0.10 | `constants.py` |
| Window length | 1.0 s | `StreamingWindowSource` |
| Step size | 0.5 s | `StreamingWindowSource` |
| Sampling rate | 250 Hz | `StreamingWindowSource` |
| Output timebase | 0.5 s (2 Hz) | `_simple_resample` |

---

## 11. Adding a New Model

1. Create `models/<name>/` with a subclass of `EmotionModel` and its `weights/` directory.
2. Register it in `models/__init__.py`:

```python
from nucleuskit_pipeline.hermes.processor.emotions_processor.models.v13 import V13EmotionModel
_REGISTRY["v13"] = V13EmotionModel
```

3. Select it at runtime:

```python
computeEmotions(recpath, model_name="v13")
```

No changes to the pipeline loop or session orchestration are required.

---

## 12. Classifier Weights Location

```
models/v12/weights/
├── config.json
├── feature_columns.json
├── artefact_config.json
├── stage1_neutral_detector.pkl   # Artefact gate
├── stage2_lda.pkl                # Neutral gate (LDA)
├── stage2_knn.pkl                # Emotion classifier (KNN)
├── stage2_cov.npy                # Covariance matrix for Mahalanobis distance
└── label_encoder.pkl
```

To retrain or update the model, replace these files and ensure `feature_columns.json` lists the columns in the exact order expected by the feature extraction step.

---

*© RE-AK Technologies Inc.*
