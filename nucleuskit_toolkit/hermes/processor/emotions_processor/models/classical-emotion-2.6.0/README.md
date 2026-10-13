# classical-emotion v2.6.0 — Integration guide

Gprior_field export of the Hermes EXG emotion classifier. Same 25-D
`extended_nm_a` + logreg as 2.5.0. Amplitude scale is **gated p95 with a
field cold-start prior**, on **20–110 Hz** windows (60 Hz notch). This is
**not** the 2.5.0 full-session mean/std recipe and **not** 15–40 Hz.

| | |
|---|---|
| Experiment | **exp-0044** |
| Feature set | `extended_nm_a` (25 features) |
| Amplitude | gated p95 + field prior (`prior_ratios.json`) |
| Classifier | logistic regression (`C=1.0`) + `StandardScaler` |
| Paired LOSO vs field 2.5.0 | **47.5%** vs **43.4%** (p_greater=0.018) |
| 5% sparse | **46.6%** (within 1 point of dense) |
| Do **not** apply | NLMS decorrelation |

Copy this **entire directory** (or unzip `classical-emotion-2.6.0.zip`) into the
consuming project. It is self-contained. `prior_ratios.json` is a required
artifact: 2.6.0's behavior depends on it.

---

## Honest caveats

1. **Fear is effectively undetected** (~0.11 recall vs ~11% chance) on every
   calibration variant, including 2.5.0. That is a model limitation, not a
   scaler bug. This release **does not emit FEAR as the predicted label**.
   If argmax is FEAR, the next-best class is returned, `fear_suppressed=True`,
   `low_confidence=True`, and FEAR stays in `probabilities`. Datastories must
   not claim fear until there is spontaneous fear data.
2. **Numbers are still posed LOSO.** Field SURPRISE share drops (MCQ 35%→11%,
   Odyssey 43%→7%, MUTEK 37%→10%), but some of that may be the gate. A small
   hand-labeled field set is still required for a true field number.
3. **Mild selection optimism.** The winner was picked from several variants
   on the same 24 folds, so p=0.018 is a bit generous. Not a blocker.

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
    → 60 Hz notch, then 20–110 Hz bandpass (4th-order Butterworth, filtfilt)
      on the full recording
    → slice 2.0 s windows (500 samples at 250 Hz)
    → this model
```

This package **does not** filter, resample, or window. It classifies one
already-filtered window at a time.

**Do not run NLMS.** Do not use 15–40 Hz windows with this checkpoint.

---

## 3. Input contract

| Property | Value |
|---|---|
| Shape | `[8, 500]` per window — **channels × samples** |
| Dtype | finite `float32` or `float64` (no NaN/Inf) |
| Sample rate | 250 Hz |
| Duration | 2.0 s |
| Band | already 20–110 Hz after 60 Hz notch |

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

RMS/WL location is the **session rest median**. Scale is a shrink of
`rest × prior_ratio` toward gated p95 of windows above `2 × rest median`.
The prior ratio table is `prior_ratios.json` (field p95/rest, 57 MCQ /
Odyssey / MUTEK sessions).

1. Collect every window for **one subject / one recording** (20–110 Hz).
2. Call `set_subject_context(windows)` **once**.
3. Read `pred.last_calibration` (`expressive_fraction`, `prior_weight`).
4. Then `predict()` each window.
5. When the recording changes, call `set_subject_context` again.

`prior_weight` is 0 when the session is all rest (pure prior) and approaches
1 as gated windows accumulate (`n / (n + 10)`).

Calling `predict` before `set_subject_context` raises `RuntimeError`.

---

## 5. Minimal integration

```python
import sys
from pathlib import Path

import numpy as np

release = Path("path/to/classical-emotion/2.6.0")  # this folder
sys.path.insert(0, str(release))

from exg_emotion.runtime.predictor import Predictor

pred = Predictor.from_release_dir(release)

# subject_windows: float array [n_windows, 8, 500], 20-110 Hz + notch
pred.set_subject_context(subject_windows)
print(pred.last_calibration)
# {'recipe': 'gated_prior', 'expressive_fraction': ..., 'prior_weight': ..., ...}

for i in range(subject_windows.shape[0]):
    out = pred.predict(subject_windows[i])
    if out.fear_suppressed:
        # do not write a fear claim; FEAR mass is still in out.probabilities
        pass
    print(out.label, out.probabilities[out.label])
```

`out` is an `EmotionPrediction`:

| Field | Meaning |
|---|---|
| `label` | Winning class (8-way after contempt merge; never FEAR) |
| `probabilities` | `dict[str, float]`, sums to 1.0 (includes FEAR) |
| `fear_suppressed` | argmax was FEAR; label is the next-best class |
| `low_confidence` | True when fear was suppressed |
| `model_name` | `"classical-emotion"` |
| `model_version` | `"2.6.0"` |
| `usable` | `True` |
| `quality` | `None` (SQI not implemented) |

---

## 6. Output labels

Public classes (8):

`ANGER`, `CONTEMPT`, `DISGUST`, `FEAR`, `HAPPINESS`, `NEUTRAL`, `SADNESS`, `SURPRISE`

FEAR is in `probabilities` only. Internally the head is 9-way
(`CONTEMPT_LEFT` / `CONTEMPT_RIGHT`), merged to **CONTEMPT** after argmax.

---

## 7. Features (for debugging, not for you to compute)

25-D vector, then sklearn `StandardScaler` + logreg:

| Offset | Count | What | Norm |
|---|---|---|---|
| 0–7 | 8 | RMS, all 8 channels | rest median / gated-p95+prior |
| 8–14 | 7 | Waveform length, EMG only (no Temporal_EEG) | rest median / gated-p95+prior |
| 15–21 | 7 | Median frequency 20–120 Hz, EMG only | none |
| 22 | 1 | Zygomatic `(L−R)/(L+R)` | scale-free |
| 23 | 1 | Envelope corr Glabella × Supraorbital_L | scale-free |
| 24 | 1 | NM-A `(Glabella − Nasolabial)/(Glabella + Nasolabial)` | scale-free |

---

## 8. Integration checklist

- [ ] Notch 60 Hz, then bandpass **20–110 Hz** on the full recording; slice 2 s.
- [ ] Shape `[8, 500]`, channel order as `CHANNELS` above.
- [ ] No NLMS / no extra spatial whitening.
- [ ] Ship `prior_ratios.json` next to `pipeline.joblib`.
- [ ] `set_subject_context` per recording; log `expressive_fraction` and `prior_weight`.
- [ ] Do not emit FEAR as a datastory (`fear_suppressed`).
- [ ] Handle 8-class output (`CONTEMPT`, not left/right).
- [ ] Do not load this zip against the 2.5.0 15–40 Hz checkpoint or vice versa.

---

## 9. Provenance

- Dataset: `hermes-7ccc7bf4f8e6` labels (24 subjects, 25 sessions, 1811
  label-matched HOLD windows), rebuilt at 20–110 Hz + 60 Hz notch
- Prior: field gated p95 / rest median, 57 sessions, MCQ / Odyssey / MUTEK
- Checkpoint: `pipeline.joblib` from exp-0044
- Protocol LOSO: `reports/calibration/release.md` (Gprior_field)
