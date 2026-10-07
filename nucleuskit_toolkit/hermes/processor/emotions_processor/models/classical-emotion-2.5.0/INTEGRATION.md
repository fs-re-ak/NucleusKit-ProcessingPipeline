# classical-emotion v2.5.0 — Integration guide

Canon z-score export of the Hermes EXG emotion classifier, retrained on the
current 24-subject labels (same feature geometry as 2.4.0 / exp-0039). This is
**not** the 2.3.0 low-power rest-reference model.

| | |
|---|---|
| Experiment | **exp-0040** |
| Feature set | `extended_nm_a` (25 features) |
| Classifier | logistic regression (`C=1.0`) + `StandardScaler` |
| LOSO (24 subjects) | **48.2% ± 12.1% accuracy**, **43.6% ± 11.2% macro F1** |
| Do **not** apply | NLMS decorrelation |

Copy this **entire directory** (or unzip `classical-emotion-2.5.0.zip`) into the
consuming project. It is self-contained: fitted pipeline, feature extraction,
validation, and the predictor API. The training repository is not required.

---

## 1. Install

```bash
pip install -r requirements.txt
```

`numpy>=1.24`, `scikit-learn>=1.3`, `joblib>=1.3`.

The checkpoint was pickled with **scikit-learn 1.6.x**. Use 1.6.x in production
if `joblib.load` warns about version mismatch.

---

## 2. Where this model sits in the pipeline

```
raw 8-ch EXG
    → hardware invalidation / NaN handling
    → 15–40 Hz bandpass (4th-order Butterworth, filtfilt) on the full recording
    → slice 2.0 s windows (500 samples at 250 Hz)
    → this model
```

This package **does not** filter, resample, or window. It classifies one
already-filtered window at a time.

**Do not run NLMS (or any other cross-channel decorrelation) before this
model.** Canon training used bandpass-only windows. NLMS cost ~14 pp LOSO.

---

## 3. Input contract

| Property | Value |
|---|---|
| Shape | `[8, 500]` per window — **channels × samples** |
| Dtype | finite `float32` or `float64` (no NaN/Inf) |
| Sample rate | 250 Hz |
| Duration | 2.0 s |
| Band | already 15–40 Hz |

**Row order must match training:**

```python
CHANNELS = [
    "Supraorbital_L",  # ch1
    "Supraorbital_R",  # ch2
    "Zygomatic_L",     # ch3
    "Zygomatic_R",     # ch4
    "Temporal_EEG",    # ch5
    "Glabella",        # ch6
    "Temporal_R",      # ch7
    "Nasolabial",      # ch8
]
```

Batch of one subject's windows for calibration: `[n_windows, 8, 500]`.

---

## 4. Subject calibration (required)

RMS and waveform-length features are **per-subject z-scored** across that
subject's windows (full-recording mean/std). This is not optional.

1. Collect every window for **one subject / one recording**.
2. Call `set_subject_context(windows)` **once** for that subject.
3. Then `predict()` each window.
4. When the subject (or recording) changes, call `set_subject_context` again
   with the new subject's windows. Never mix subjects in one calibration.

**Offline (matches training):** pass the whole session.

**Streaming:** this release is a full-recording z-score. A short prefix will
not match training. Accumulate the session (or a long trailing buffer), then
calibrate before you trust scores. Do not pair this checkpoint with the
low-power rest reference (`extended_nm_a_lp` / 2.3.0).

Calling `predict` before `set_subject_context` raises `RuntimeError`.

---

## 5. Minimal integration

```python
import sys
from pathlib import Path

import numpy as np

release = Path("path/to/classical-emotion/2.5.0")  # this folder
sys.path.insert(0, str(release))

from exg_emotion.runtime.predictor import Predictor

pred = Predictor.from_release_dir(release)

# subject_windows: float array [n_windows, 8, 500], one recording
pred.set_subject_context(subject_windows)

for i in range(subject_windows.shape[0]):
    out = pred.predict(subject_windows[i])
    print(out.label, out.probabilities[out.label])
```

`out` is an `EmotionPrediction`:

| Field | Meaning |
|---|---|
| `label` | Winning class (8-way after contempt merge) |
| `probabilities` | `dict[str, float]`, sums to 1.0 |
| `model_name` | `"classical-emotion"` |
| `model_version` | `"2.5.0"` |
| `usable` | always `True` in this release |
| `quality` | `None` (SQI not implemented) |

---

## 6. Output labels

Public classes (8):

`ANGER`, `CONTEMPT`, `DISGUST`, `FEAR`, `HAPPINESS`, `NEUTRAL`, `SADNESS`, `SURPRISE`

Internally the head is 9-way (`CONTEMPT_LEFT` / `CONTEMPT_RIGHT`). After
argmax those two are merged into **CONTEMPT** and their probabilities are
summed. You never see the laterality labels at the API.

---

## 7. Features (for debugging, not for you to compute)

25-D vector, then sklearn `StandardScaler` + logreg:

| Offset | Count | What | Norm |
|---|---|---|---|
| 0–7 | 8 | RMS, all 8 channels | per-subject z-score |
| 8–14 | 7 | Waveform length, EMG only (no Temporal_EEG) | per-subject z-score |
| 15–21 | 7 | Median frequency 20–120 Hz, EMG only | none |
| 22 | 1 | Zygomatic `(L−R)/(L+R)` | scale-free |
| 23 | 1 | Envelope corr Glabella × Supraorbital_L | scale-free |
| 24 | 1 | NM-A `(Glabella − Nasolabial)/(Glabella + Nasolabial)` | scale-free |

---

## 8. Integration checklist

- [ ] Bandpass 15–40 Hz on the **full recording**, then slice; do not filter each 2 s window in isolation if that is not what the training pipeline did (`sosfiltfilt` on the whole take).
- [ ] Shape `[8, 500]`, channel order as `CHANNELS` above.
- [ ] No NLMS / no extra spatial whitening.
- [ ] `set_subject_context` per subject from that subject's own windows.
- [ ] Finite samples only (drop or interpolate NaNs before predict).
- [ ] Handle 8-class output (`CONTEMPT`, not left/right).
- [ ] Do not load this zip against the 2.3.0 low-power checkpoint or vice versa.

---

## 9. Provenance

- Dataset: `hermes-7ccc7bf4f8e6` (24 subjects, 25 sessions, 1811 label-matched windows)
- Checkpoint: `pipeline.joblib` from exp-0040
- Config: `configs/classical-011.yaml`
- Same feature geometry as 2.4.0 / exp-0039; retrained on the current reviewed labels
