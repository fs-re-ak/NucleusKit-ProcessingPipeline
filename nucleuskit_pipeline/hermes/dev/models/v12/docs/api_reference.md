# ModelV12 — API Reference

## Primary Interface: `StreamingEMGClassifier`

**Module:** `interface/realtime_classifier.py`

Real-time interface — accepts one raw EMG sample at a time.

### Constructor

```python
StreamingEMGClassifier(
    clf_dir: str,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
    sampling_rate: int = 250,
)
```

### Methods

#### `push_sample(raw_sample) -> dict[str, float] | None`

Feed one raw EEG frame.  Returns a probability dict when a classification fires.

#### `push_batch(raw_batch) -> Iterator[(float, dict[str, float])]`

Generator — feed a 2-D batch, yields `(time_sec, proba_dict)` pairs.

#### `reset() -> None`

Clear buffer and filter state between recordings.

---

## Secondary Interface: `TwoStageClassifier`

**Module:** `interface/inference.py`

Accepts pre-computed feature vectors.

### Loading

```python
from inference import TwoStageClassifier
clf = TwoStageClassifier.load("path/to/classifier/")
```

### Methods

#### `predict(x) -> tuple[str, float]`

Returns `(label, confidence)`.  The artefact gate is checked first.

#### `predict_proba(x) -> dict[str, float]`

Full probability distribution.  Artefact windows return `{"Neutral": 1.0, others: 0.0}`.

#### `predict_batch(X) -> tuple[np.ndarray, np.ndarray]`

Batch prediction — artefact gate applied per row.

#### `predict_batch_detailed(X) -> tuple[labels, confidences, artefact_mask, p_active]`

Like `predict_batch` but also returns the per-window artefact boolean mask
and Stage 1 active probability (useful for diagnostics and analysis scripts).

#### `is_artefact(x) -> bool`

Returns `True` if `AVG_RMS > artefact_threshold` for the given feature vector.

### Properties

| Property | Type | Description |
|---|---|---|
| `feature_columns` | `list[str]` | Ordered feature names |
| `emotion_classes` | `list[str]` | Stage 2 emotion labels |
| `all_classes` | `list[str]` | All labels including Neutral |
| `artefact_gate_enabled` | `bool` | True when artefact_config.json was loaded |
| `artefact_threshold` | `float \| None` | AVG_RMS rejection threshold |

### Tunable attributes

| Attribute | Default | Description |
|---|---|---|
| `theta` | `0.5` | Stage 1 Active/Neutral decision threshold |
| `lda_weight` | `0.6` | Stage 2 LDA ensemble weight |
| `knn_weight` | `0.4` | Stage 2 KNN ensemble weight |
| `artefact_threshold` | from JSON | Override at runtime to tune sensitivity |

---

## Feature vector specification

9-element input vector:

| Index | Name | Description |
|---|---|---|
| 0–7 | `<CHANNEL>_RMS` | L2-normalised per-channel RMS |
| 8 | `AVG_RMS` | Mean RMS (raw, pre-normalisation) — used by artefact gate |

---

## Dependencies

```
scikit-learn >= 1.6.1
scipy        >= 1.17.0
numpy        >= 2.3.5
joblib       >= 1.4.2
```

Install: `pip install -r requirements.txt`
