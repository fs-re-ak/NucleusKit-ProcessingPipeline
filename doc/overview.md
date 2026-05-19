# NucleusKit Processing Pipeline — Technical Overview

**Package:** `nucleuskit_pipeline`  
**Entry point:** `python -m nucleuskit_pipeline` (GUI) or `--session <dir>` (headless CLI)  
**Authors:** Fred Simard — RE-AK Technologies Inc., Winter/Spring 2026

---

## 1. Purpose

The NucleusKit Processing Pipeline is an **offline analytics engine** that transforms raw multi-modal recordings produced by a Nucleus-Kit session into a set of clean, time-aligned physiological and cognitive feature files. It is designed to run against a session directory on disk and is idempotent: each step checks whether its outputs already exist and skips re-computation when they are present.

---

## 2. Pipeline Orchestrator

**Module:** `nucleuskit_pipeline/pipeline.py`  
**Class:** `NucleusKitProcessingPipeline`

The orchestrator assembles an ordered list of processing steps during `configureAsDefault()` and executes them sequentially in `processSession(jobDetails)`. Any exception thrown by a single step is caught, logged, and does not abort subsequent steps.

### Step execution order

| # | Function | Module | Description |
|---|----------|--------|-------------|
| 0 | `ensure_session_rawdata_layout` | `session/layout.py` | Migrate flat sessions into the canonical `rawData/` sub-folder layout |
| 1 | `convertPOVMovieClip` *(optional)* | `camera/processing.py` | Convert POV camera UUID file from MKV to MP4 (requires POV config) |
| 2 | `prepareDirectory` | `session/layout.py` | Create `features/`, `results/`, `meta/`, and processing log |
| 3 | `ensure_session_video_rotated_180` | `camera/processing.py` | Apply ffmpeg 180° rotation to `rawData/video.mp4` (idempotent marker) |
| 4 | `extractMetaInfo` | `session/meta/info.py` | Compute session duration and readability → `features/metainfo.json` |
| 5 | `computeHeartDynamics` | `shimmer/processor/heart.py` | PPG → heart rate, HRV metrics → `results/HeartDynamics.csv` |
| 6 | `computeArousal` | `shimmer/processor/eda.py` | EDA → tonic/phasic decomposition → `results/Arousal.csv` |
| 7 | `computeCognitiveIndexes` | `hermes/processor/cognition_processor.py` | EEG → power bands → engagement indexes → `results/Cognition.csv` |
| 8 | `computeEmotions` | `hermes/processor/emotions_processor.py` | EMG → two-stage classifier → emotion probabilities → `results/Emotions.csv` |
| 9 | `eventProcessor` | `events/processor.py` | Split `rawEvents.csv` into feature and web event CSVs |
| 10 | `seedPlaybackAnnotations` | `events/eventsProcessor.py` | Generate `features/events/playback_annotations.json` |
| 11 | `processUWB` | `position/uwb_processor.py` | UWB multilateration → `results/positions.csv` |
| 12 | `processGPS` | `position/gps_processor.py` | GPS interpolation → `results/gpsDf.csv` |

---

## 3. Session Directory Layout

A processed session folder has the following canonical structure:

```
<session>/
├── rawData/                    # Raw hardware outputs (read-only by the pipeline)
│   ├── shimmer.csv             # Multi-channel Shimmer export (timestamp, EDA, PPG, …)
│   ├── rawEEG_0.csv            # Hermes EEG/EMG recording (timestamp + 8 channels)
│   ├── rawEvents.csv           # Raw event log
│   ├── event.csv               # Processed event timestamps (for playback seeding)
│   ├── gps.csv / gps.tmp       # GPS log (unix_ts, lat, lon)
│   ├── uwb_0.csv / uwb.tmp     # UWB DIST-frame log
│   └── video.mp4               # Optional POV recording
│
├── features/                   # Intermediate computations (traceability artefacts)
│   ├── ppg/
│   │   ├── ppg_resampled.csv           # Artifact-cleaned PPG at 51.2 Hz
│   │   ├── ppg_normalized.csv          # Rolling z-score normalised PPG
│   │   ├── ppg_resample_report.txt     # Gap and rate statistics
│   │   ├── ppg_rejected.png            # Three-colour artifact rejection map
│   │   ├── ppg_overview.png            # Waveform + annotated peaks
│   │   └── ppg_spectrogram.png         # STFT spectrogram (0–6 Hz)
│   ├── eda/
│   │   ├── eda_resampled.csv           # Resampled EDA at 51.2 Hz (kΩ)
│   │   ├── eda_resample_report.txt
│   │   ├── eda_overview.png            # EDA, tonic component, SCR events
│   │   └── SCR_events.csv              # Discrete SCR event timestamps and amplitudes
│   ├── cognition/
│   │   └── powerBands.csv              # Per-window, per-channel, per-band EEG power
│   ├── emotions/
│   │   ├── rmsSignals.csv              # Per-window per-channel EMG RMS
│   │   └── emotionClassifierInputs.csv # L2-normed RMS + classifier predictions
│   └── events/
│       ├── processedFeatureEvents.csv
│       ├── processedWebFeatureEvents.csv
│       └── playback_annotations.json
│
├── results/                    # Final output files consumed by downstream tools
│   ├── HeartDynamics.csv       # 2 Hz: mean_hr, mean_nn, sdnn, rmssd, pnn50, cvsd, cvnn
│   ├── Arousal.csv             # 2 Hz: TonicEDA, PhasicEDA
│   ├── Cognition.csv           # 2 Hz: Engagement, Intertemporal, Lateralization, Frontal
│   ├── Emotions.csv            # 2 Hz: Neutral, Happiness, Anger, Surprise, …
│   ├── gpsDf.csv               # 2 Hz: Latitude, Longitude
│   └── positions.csv           # 2 Hz: X, Y (UWB)
│
└── meta/                       # Session metadata
    └── metainfo.json
```

---

## 4. Shared Output Timebase

All result files share the same **2 Hz / 0.5-second timebase**. The `Timestamp` column in every result CSV represents **seconds from the start of the recording**. This alignment enables direct multi-modal fusion without additional resampling on the consumer side.

Hardware gaps (sensor disconnects ≥ 5 s) are propagated as `NaN` through every pipeline and are visible in all output files, preserving temporal fidelity.

---

## 5. Incremental Processing (Caching)

Each pipeline step begins by checking whether its primary output file already exists:

- If the output exists → step is skipped with an `INFO` log message.
- If the output is missing → full computation is performed and outputs are written.

This makes the pipeline safe to re-run after partial failures. To force recomputation of a specific step, delete the corresponding output file(s) from `results/` or `features/`.

---

## 6. Individual Pipeline Documentation

| File | Pipeline |
|------|----------|
| [pipeline_heart.md](pipeline_heart.md) | Shimmer PPG → Heart Rate Dynamics |
| [pipeline_arousal.md](pipeline_arousal.md) | Shimmer EDA → Arousal (cvxEDA) |
| [pipeline_cognition.md](pipeline_cognition.md) | Hermes EEG → Cognitive Indexes |
| [pipeline_emotions.md](pipeline_emotions.md) | Hermes EMG → Emotion Classification |
| [pipeline_events.md](pipeline_events.md) | Raw Events → Feature / Web Events + Playback Annotations |
| [pipeline_positioning.md](pipeline_positioning.md) | GPS and UWB Positioning |

---

## 7. Hardware Inputs Summary

| Hardware | File(s) | Nominal Rate | Channels used |
|----------|---------|-------------|---------------|
| Shimmer | `shimmer.csv` / `rawShimmer_0.csv` | ~51.2 Hz | col 4 = EDA (kΩ), col 5 = PPG |
| Hermes | `rawEEG_0.csv` / `eeg.tmp` | 250 Hz | col 0 = timestamp, cols 1–8 = EXG |
| GPS | `gps.csv` / `gps.tmp` | variable | unix_ts, latitude, longitude |
| UWB | `uwb_0.csv` / `uwb.tmp` | variable | DIST-frame: tag, antenna, distance |

---

*© RE-AK Technologies Inc.*
