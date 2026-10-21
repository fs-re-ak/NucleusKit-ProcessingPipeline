# Emotions Pipeline — EMG-based Emotion Classification

> **Since Nucleus-Kit 2.0.0** this processing is done by re-ak-core (`reak_core.femg` and `reak_core.models`). This page describes Nucleus-Kit 1.1 and will move to re-ak-core's documentation; the module paths below are those of 1.1. Every change in the results since 1.1 is listed in re-ak-core's [CHANGELOG](https://github.com/fs-re-ak/re-ak-core/blob/main/CHANGELOG.md).

**Package:** `nucleuskit_toolkit/hermes/processor/emotions_processor/`
**Entry point:** `computeEmotions(recpath)`
**Primary output:** `results/Emotions.csv`
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter–Spring 2026

---

## 1. Purpose

The emotions pipeline infers moment-to-moment facial emotion intensity from
8-channel surface EMG recorded by the Hermes device.  Classification defaults
to **classical-emotion 2.5.0** — a logistic-regression model trained on the
`extended_nm_a` feature set (exp-0040).  Other vendored releases (for example
2.6.0) can be selected from offline processing or via
`computeEmotions(..., model_name=...)`.  The output is a set of per-emotion
intensity time-series sampled at 2 Hz.  2.5.0 writes class probabilities;
2.6.0 writes a one-hot of the winning class.

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
    ├── __init__.py                 # get_model / list_models
    ├── catalog.py                  # Discover release folders + preprocess specs
    ├── classical_emotion/          # Adapter wrapping a vendored exg_emotion package
    │   ├── __init__.py
    │   └── model.py                # ClassicalEmotionAdapter
    ├── classical-emotion-2.5.0/    # Vendored release (15–40 Hz)
    └── classical-emotion-2.6.0/    # Vendored release (60 Hz notch + 20–110 Hz)
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

**Channel order** (Hermes hardware order = classical-emotion training order).
Raw files may be headerless or use `ch1`..`ch8` in this order. Written RMS
columns use the output names below.

| Index | Raw column | Electrode | Output RMS column |
|-------|------------|-----------|-------------------|
| 0 | ch1 | AF8 | HEAD_R_RMS |
| 1 | ch2 | AF7 | HEAD_L_RMS |
| 2 | ch3 | CHEEK_R | CHEEK_R_RMS |
| 3 | ch4 | CHEEK_L | CHEEK_L_RMS |
| 4 | ch5 | EAR_R | EAR_R_RMS |
| 5 | ch6 | AFz | FOREHEAD_L_RMS |
| 6 | ch7 | BROW_L | BROW_RMS |
| 7 | ch8 | NOSE | NOSE_RMS |

Electrode names stay internal (cognition, NLMS, realtime). Feature files and
the emotions report use the output RMS names. Raw `ch1`..`ch8` are accepted
in this order, including when the header lists them out of order.

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

The default model lives in `models/classical-emotion-2.5.0/pipeline.joblib`.
Additional releases sit beside it (for example `classical-emotion-2.6.0/`).

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
2. Applies the selected release's filter: optional 60 Hz notch, then a
   4th-order Butterworth bandpass via `scipy.signal.sosfiltfilt` **on the
   full recording at once** (zero-phase).  2.5.0 uses 15–40 Hz and no notch;
   2.6.0 uses a 60 Hz notch then 20–110 Hz.
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
| `Neutral` | 0–1 | Emotion intensity |
| `Happiness` | 0–1 | Emotion intensity |
| `Anger` | 0–1 | Emotion intensity |
| `Surprise` | 0–1 | Emotion intensity |
| `Contempt` | 0–1 | Emotion intensity |
| `Disgust` | 0–1 | Emotion intensity |
| `Fear` | 0–1 | Emotion intensity |
| `Sadness` | 0–1 | Emotion intensity |

For **2.5.0**, each intensity is the class probability (the row sums to 1).
For **2.6.0**, intensity is the argmax: the winning class is `1` and every
other class is `0`.  The winning class is the model's emitted label (argmax
after the contempt merge; if that argmax is fear, the next-best class is
used and fear stays `0`).  `PredictedConfidence` in
`emotionClassifierInputs.csv` remains the winning-class probability for
both releases.

`NaN` rows correspond to null windows.  Written with `na_rep="NULL"`.

### `features/emotions/rmsSignals.csv`

One row per classifier window (0.5 s step).  Columns: `Timestamp`, followed
by `HEAD_R_RMS`, `HEAD_L_RMS`, `CHEEK_R_RMS`, `CHEEK_L_RMS`, `EAR_R_RMS`,
`FOREHEAD_L_RMS`, `BROW_RMS`, `NOSE_RMS` (raw columns `ch1`..`ch8`).
Values are raw (un-normalised) RMS computed from the 2.0 s filtered window.
`NaN` rows correspond to null windows.

### `features/emotions/emotionClassifierInputs.csv`

One row per valid (non-null) classifier window.  Columns:

- `Timestamp`
- 25 feature columns (`rms_*`, `wl_*`, `mdf_*`, `zyg_asymmetry`,
  `ant_corr`, `nm_ratio_a`) as defined by `model.feature_columns`
- `PredictedLabel` (title-case emotion name)
- `PredictedConfidence` (float, 0–1)
- L2-normalised per-channel RMS: `HEAD_R_RMS` … `NOSE_RMS` (same order as
  `ch1`..`ch8`). The emotions report reads these columns.

---

## 10. Key Constants

| Constant | Value | Location |
|----------|-------|---------|
| `DISCONNECT_VALUE` | 187 500 | `constants.py` |
| `NULL_WINDOW_NAN_THRESHOLD` | 0.10 | `constants.py` |
| Window length | 2.0 s (500 samples) | `StreamingWindowSource` |
| Step size | 0.5 s (125 samples) | `StreamingWindowSource` |
| Bandpass / notch | From the selected release `config.json`; default 15–40 Hz, no notch | `StreamingWindowSource` |
| Sampling rate | 250 Hz | `StreamingWindowSource` |
| Output timebase | 0.5 s (2 Hz) | `_simple_resample` |

---

## 11. Adding a New Model

### Drop-in classical-emotion release

1. Copy the release directory into `models/` so it contains `manifest.json`,
   `config.json`, `pipeline.joblib`, and `exg_emotion/` (plus any extra
   artifacts such as `prior_ratios.json`).
2. Declare `bandpass_hz` and, if needed, `notch_hz` in `config.json`.
   If `bandpass_hz` is omitted, the pipeline uses 15–40 Hz and no notch.
3. Add a matching `[tool.setuptools.package-data]` entry in `pyproject.toml`
   so installed wheels include the artifacts.
4. Select it by directory name:

```python
computeEmotions(recpath, model_name="classical-emotion-2.6.0")
```

The offline GUI combo and `python -m nucleuskit_toolkit --emotion-model …`
list every discovered release.  The registry key is the directory name.
`classical-emotion` remains an alias for `classical-emotion-2.5.0`.

### Custom EmotionModel subclass

1. Create `models/<name>/` with a subclass of `EmotionModel`.
2. Implement `load`, `reset`, `set_subject_context`, `predict`,
   `feature_columns`, and `emotion_labels`.  Override `preprocess_spec`
   when the model is not trained on 15–40 Hz windows.
3. Wire it through `get_model` (or keep using a drop-in release folder).
4. Select it at runtime:

```python
computeEmotions(recpath, model_name="my-model")
```

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

models/classical-emotion-2.6.0/   # same layout, plus prior_ratios.json
```

The `exg_emotion` package is imported at runtime via `sys.path.insert` in
`ClassicalEmotionAdapter.load()`.  Switching versions unloads the previous
`exg_emotion*` modules so the two releases do not share one import.

---

*© RE-AK Technologies Inc.*
