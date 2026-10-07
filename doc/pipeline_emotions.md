# Emotions Pipeline — EMG-based Emotion Classification

**Package:** `nucleuskit_toolkit/hermes/processor/emotions_processor/`
**Entry point:** `computeEmotions(recpath)`
**Primary output:** `results/Emotions.csv`
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter–Spring 2026

---

## 1. Purpose

The emotions pipeline infers moment-to-moment facial emotion probabilities from
8-channel surface EMG recorded by the Hermes device.  Classification is
performed using **classical-emotion 2.5.0** — a logistic-regression model
trained on the `extended_nm_a` feature set (exp-0040, 24 subjects, 48.2 %
LOSO accuracy).  The output is a set of per-emotion probability time-series
sampled at 2 Hz.

---

## 2. Package Layout

```
hermes/processor/emotions_processor/
├── __init__.py          # Public API: computeEmotions, EMOTION_COLUMNS
├── constants.py         # Shared constants (thresholds, column lists)
├── session.py           # computeEmotions() — orchestration & I/O
├── pipeline.py          # run_window_pipeline() — two-pass classification loop
├── report.py            # Visual statistics report generator
├── interface/
│   ├── model.py         # EmotionModel ABC + EmotionWindowResult dataclass
│   ├── windows.py       # EmotionWindow dataclass + WindowSource Protocol
│   └── streaming.py     # StreamingWindowSource (batch filtfilt)
└── models/
    ├── __init__.py                 # Model registry: get_model(name)
    ├── classical_emotion/          # Adapter wrapping the vendored exg_emotion package
    │   ├── __init__.py
    │   └── model.py                # ClassicalEmotionAdapter
    └── classical-emotion-2.5.0/   # Vendored release artifact (pipeline.joblib, exg_emotion/)
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

**Format:** Headerless or single-header CSV.  Column 0 is the hardware
timestamp (milliseconds, auto-normalised).  Columns 1–8 are the eight EMG
channels in the order defined by `HermesConstants`.

**Sampling rate:** 250 Hz

**Channel order** (Hermes hardware order = classical-emotion training order):

| Index | Hermes name | Training name |
|-------|-------------|---------------|
| 0 | AF8 | Supraorbital_L |
| 1 | AF7 | Supraorbital_R |
| 2 | CHEEK_R | Zygomatic_L |
| 3 | CHEEK_L | Zygomatic_R |
| 4 | EAR_R | Temporal_EEG |
| 5 | AFz | Glabella |
| 6 | BROW_L | Temporal_R |
| 7 | NOSE | Nasolabial |

The names differ cosmetically; the hardware physical mapping is identical.  No
channel permutation is applied.

---

## 4. Hardware Disconnect Invalidation

Before any processing, samples whose absolute value is within 1 of the hardware
saturation value **187 500** are replaced with `NaN`.

```python
# constants.py
DISCONNECT_VALUE = 187500.0
NULL_WINDOW_NAN_THRESHOLD = 0.10   # 10 % of samples NaN => null window
```

---

## 5. Classifier Architecture

The bundled model lives in `models/classical-emotion-2.5.0/pipeline.joblib`.

| Property | Value |
|----------|-------|
| Experiment | exp-0040 |
| Feature set | `extended_nm_a` (25 features) |
| Classifier | logistic regression (`C=1.0`, sklearn `StandardScaler`) |
| Training data | 24 subjects, 1 811 label-matched 2 s windows |
| LOSO accuracy | 48.2 % ± 12.1 % |
| LOSO macro F1 | 43.6 % ± 11.2 % |

**25-D feature vector `extended_nm_a`:**

| Offset | Count | Feature | Normalisation |
|--------|-------|---------|---------------|
| 0–7 | 8 | RMS all channels | per-subject z-score |
| 8–14 | 7 | Waveform length, EMG only (no EAR_R) | per-subject z-score |
| 15–21 | 7 | Median frequency 20–120 Hz, EMG only | none |
| 22 | 1 | Zygomatic (L−R)/(L+R) asymmetry | scale-free |
| 23 | 1 | Envelope corr Glabella × Supraorbital_L | scale-free |
| 24 | 1 | NM-A (Glabella − Nasolabial)/(Glabella + Nasolabial) | scale-free |

**Output labels (8-way after CONTEMPT_LEFT / CONTEMPT_RIGHT merge):**
`ANGER`, `CONTEMPT`, `DISGUST`, `FEAR`, `HAPPINESS`, `NEUTRAL`, `SADNESS`, `SURPRISE`

---

## 6. Unified Window Pipeline

```
StreamingWindowSource (raw EMG, already bandpass-filtered)
    │
    │  Pass 1 — collect all windows
    ▼
run_window_pipeline()
    │  Calibrate: set_subject_context(valid_windows) — per-subject z-score
    │
    │  Pass 2 — classify each window
    │    if invalid → NaN row
    │    else → model.predict(samples)
    │           → EmotionWindowResult
    ▼
emotion_rows / rms_rows / model_input_rows
```

### StreamingWindowSource

Accepts pre-loaded, NaN-invalidated EMG `(N, 8)` + optional hardware timestamps.

1. Fills NaN positions with 0 (so the filter does not diverge; original NaN
   flags are preserved).
2. Applies a 4th-order Butterworth bandpass (15–40 Hz) via `scipy.signal.sosfiltfilt`
   **on the full recording at once** (zero-phase, no phase lag).
3. Slides a 500-sample (2.0 s) window over the result in 125-sample (0.5 s)
   steps, yielding one `EmotionWindow` per hop.  Window timestamp is the
   median of the hardware timestamps in the window, or the sample-count
   midpoint when no hardware timestamps are available.

A window where more than `NULL_WINDOW_NAN_THRESHOLD` (10 %) of samples were
originally invalid is emitted as an `is_invalid=True` window.

**NLMS is not applied before this step.**  The classical-emotion 2.5.0 model
was trained on bandpass-only windows; NLMS costs ~14 pp LOSO.

### Two-pass calibration

After all windows are collected, `run_window_pipeline` stacks the **valid**
windows into a `(n_valid, 8, 500)` array and calls `model.set_subject_context`.
This computes per-subject mean and standard deviation for RMS and waveform-
length features (required by the `extended_nm_a` z-score normalisation).
Invalid windows are excluded from calibration.

### EmotionModel (pluggable)

`ClassicalEmotionAdapter.predict(samples)` extracts the 25-D `extended_nm_a`
feature vector, classifies with the sklearn pipeline, merges
CONTEMPT_LEFT/RIGHT, and maps the uppercase labels to title-case
`EMOTION_COLUMNS` keys.

---

## 7. Processing Steps

### Step 1 — Load EMG

`loadEXG(recpath, re_reference=False)` reads the raw EXG file, normalises
timestamps to seconds from start, and returns `(timestamps, eeg_data)` of
shapes `(N,)` and `(N, 8)`.

### Step 2 — Invalidate hardware artefacts

`_invalidate_disconnected_samples(eeg_data)` sets disconnect-value samples to
`NaN` in-place.

### Step 3 — Window pipeline

`run_window_pipeline(source, model)` runs the two-pass loop (collect → calibrate
→ classify) described in Section 6.

### Step 4 — Resample to 2 Hz (when hardware timestamps available)

The raw emotion DataFrame (timestamped at window midpoints) is snapped to the
shared 0.5 s grid via `_simple_resample`.  `NaN` windows remain `NaN` after
resampling.

### Step 5 — Write outputs

Three files are written (see Section 9).

### Step 6 — Diagnostic report

`generate_report(recpath)` saves per-emotion jitter plots and a pie chart of
the predicted label distribution.

---

## 8. Incremental Caching

| Condition | Behaviour |
|-----------|-----------|
| All three output files exist | Entire step skipped |
| Any file is missing | Full raw-EXG path via `StreamingWindowSource` |

The RMS-replay path (from `rmsSignals.csv`) is no longer supported.
`rmsSignals.csv` is now a write-only diagnostic output.  When emotions need
recomputing, the full EXG is always reloaded.  This ensures subject
calibration uses the actual per-subject statistics for every recomputation.

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

`NaN` rows correspond to null windows.  Written with `na_rep="NULL"`.

### `features/emotions/rmsSignals.csv`

One row per classifier window (0.5 s step).  Columns: `Timestamp`, followed
by `AF8`, `AF7`, `CHEEK_R`, `CHEEK_L`, `EAR_R`, `AFz`, `BROW_L`, `NOSE`.
Values are raw (un-normalised) RMS computed from the 2.0 s filtered window.
`NaN` rows correspond to null windows.

### `features/emotions/emotionClassifierInputs.csv`

One row per valid (non-null) classifier window.  Columns:

- `Timestamp`
- 25 feature columns (`rms_*`, `wl_*`, `mdf_*`, `zyg_asymmetry`,
  `ant_corr`, `nm_ratio_a`) as defined by `model.feature_columns`
- `PredictedLabel` (title-case emotion name)
- `PredictedConfidence` (float, 0–1)

---

## 10. Key Constants

| Constant | Value | Location |
|----------|-------|---------|
| `DISCONNECT_VALUE` | 187 500 | `constants.py` |
| `NULL_WINDOW_NAN_THRESHOLD` | 0.10 | `constants.py` |
| Window length | 2.0 s (500 samples) | `StreamingWindowSource` |
| Step size | 0.5 s (125 samples) | `StreamingWindowSource` |
| Bandpass | 15–40 Hz | `StreamingWindowSource` |
| Sampling rate | 250 Hz | `StreamingWindowSource` |
| Output timebase | 0.5 s (2 Hz) | `_simple_resample` |

---

## 11. Adding a New Model

1. Create `models/<name>/` with a subclass of `EmotionModel`.
2. Implement `load`, `reset`, `set_subject_context`, `predict`,
   `feature_columns`, and `emotion_labels`.
3. Register it in `models/__init__.py`:

```python
from nucleuskit_toolkit.hermes.processor.emotions_processor.models.mymodel import MyModel
_REGISTRY["my-model"] = MyModel
```

4. Select it at runtime:

```python
computeEmotions(recpath, model_name="my-model")
```

No changes to the pipeline loop or session orchestration are required.

---

## 12. Model Artifact Location

```
models/classical-emotion-2.5.0/
├── pipeline.joblib        # Fitted sklearn Pipeline (StandardScaler + LogReg)
├── manifest.json          # Model metadata (name, version, input spec, labels)
├── config.json            # Training configuration (feature_set, channels, …)
├── labels.json            # Human-readable label list
├── metrics.json           # LOSO evaluation results
└── exg_emotion/           # Self-contained feature extraction + runtime package
    ├── core/              # interfaces.py, types.py, validation.py, exceptions.py
    ├── models/classical_v1/  # ClassicalEmotionModel, feature extraction
    └── runtime/           # Predictor, loader, registry
```

The `exg_emotion` package is imported at runtime via `sys.path.insert` in
`ClassicalEmotionAdapter.load()`.

---

*© RE-AK Technologies Inc.*
