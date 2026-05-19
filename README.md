# Nucleus-Kit Processing Pipeline

Desktop application for **Nucleus-Kit** sessions: **offline** analytics (ingest session folders on disk, run the bundled processing steps, write outputs under each session's `features/` and `results/`), plus real-time streaming, playback review, and MQTT device control. The **offline** pipeline can be run from the GUI or from the command line.

## Table of contents

- [Hardware mapping](#hardware-mapping)
- [Main menu (graphical mode)](#main-menu-graphical-mode)
  - [Tools submenu](#tools-submenu)
- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [Session layout](#session-layout)
  - [Top-level folders](#top-level-folders)
  - [rawData/ — typical inputs](#rawdata--typical-inputs)
  - [features/ — notable outputs](#features--notable-outputs)
  - [results/ — primary pipeline tables](#results--primary-pipeline-tables)
- [Optional POV / ffmpeg configuration](#optional-pov--ffmpeg-configuration)
- [Technical documentation](#technical-documentation)
- [Trust and bundled models](#trust-and-bundled-models)
- [Tests](#tests)
- [License](#license)

## Hardware mapping

| Device | Signals |
|--------|---------|
| **Hermes headset** | EEG and EMG (including features used by the emotion model) |
| **Shimmer wristband** | PPG and EDA (GSR processing in code maps to the Shimmer EDA channel) |

## Main menu (graphical mode)

After `python -m nucleuskit_pipeline`, the home screen offers:

| Item | Purpose |
|------|---------|
| **Real-time viewer** | Connect to a Hermes headset over BLE and an optional Shimmer over serial: scan, stream, plot EEG / motion / Shimmer signals, and record into a session-style folder. |
| **Offline processing** | Pick a session directory and run the full analytics pipeline (log output in the window). |
| **Playback mode** | Review a processed session: optional `rawData/video.mp4` plus time-aligned plots from `results/*.csv`, with editable event annotations when available. |
| **Tools** | Opens a submenu of maintenance utilities (see below). |
| **MQTT Controller** | Connect to a broker (e.g. Vizia Mobile), discover devices from status traffic, and send recording-related MQTT commands. |
| **Settings** | Application preferences (theme: light, dark, or system), persisted under **REAK / NucleusKitPipeline** via `QSettings`. |

### Tools submenu

| Item | Purpose |
|------|---------|
| **Channel Fixer** | Repair a single bad RMS channel in an already-processed session using the bundled channel-fix model. |
| **Channel gain adjustment** | Adjust per-channel RMS gain and zero offset against a reference distribution; updates working emotion RMS features. |
| **Revert to original** | Restore working RMS features from the frozen baseline under `features/emotions/original/`. |
| **PPG Fixer** | Re-run PPG artifact rejection and HRV computation on an already-processed session (useful when tuning rejection thresholds without reprocessing the full pipeline). |

## Requirements

- Python 3.10+
- Core dependencies are declared in [`pyproject.toml`](pyproject.toml). [`requirements.txt`](requirements.txt) is a flat install file that covers all dependencies including the optional GUI extras in one go.

## Install

From the repository root:

```bash
pip install .
```

The **desktop interface** uses PySide6 (Qt 6). Install it with the optional `gui` extra:

```bash
pip install ".[gui]"
```

For development (tests + linter):

```bash
pip install ".[dev]"
```

To work on the GUI locally, combine extras:

```bash
pip install ".[dev,gui]"
```

## Run

**Graphical mode** (main menu above):

```bash
python -m nucleuskit_pipeline
```

Requires `pip install ".[gui]"`.

Replace the packaged logo anytime with your own PNG at [`nucleuskit_pipeline/ui/resources/branding/logo.png`](nucleuskit_pipeline/ui/resources/branding/logo.png) (same path in an installed package).

**Headless mode** (automation / CI):

```bash
python -m nucleuskit_pipeline --session "D:\path\to\session_folder"
```

Optional POV / ffmpeg JSON (merged after any config files found in the current working directory):

```bash
python -m nucleuskit_pipeline --session "D:\path\to\session" --config "D:\path\to\config.json"
```

After install, the same entry point is available as:

```bash
nucleuskit-pipeline --session "D:\path\to\session_folder"
```

## Session layout

The pipeline treats the **session root** as a single directory (the folder you select in Offline processing or pass to `--session`).

### Top-level folders

| Folder | Role |
|--------|------|
| **`rawData/`** | Canonical location for recordings and acquisition-side files. The GUI expects this folder to exist and be non-empty after any automatic layout fix (see below). |
| **`features/`** | Intermediate and traceability outputs: meta summary, per-processor feature exports, processed events, optional playback annotation seed. |
| **`results/`** | Primary analytics tables, logs, and optional playback-side annotation copies. |
| **`meta/`** | Reserved for session-level assets such as preview images (created when needed). |
| **`experimentConfigs/`** | Reserved for experiment or run configuration artifacts. |

On startup, if **`rawData/` is missing**, the pipeline may **create it and move** every other top-level entry into it, except the reserved names above (`rawData`, `features`, `results`, `meta`, `experimentConfigs`). That normalizes older layouts where files lived at the session root.

### `rawData/` — typical inputs

Exact filenames vary by firmware and recording path; common patterns include:

- **Hermes EEG / EMG:** `rawEEG_0.csv` or `eeg.tmp`
- **Shimmer:** `shimmer.csv`, or legacy `rawShimmer_0.csv` / `gsr.tmp`
- **Video:** `video.mp4` (used for playback and optional one-time 180° rotation during processing)
- **Events:** `rawEvents.csv` (processed into `features/processedFeatureEvents.csv` and related); `event.csv` (used to seed `features/events/playback_annotations.json` when that file does not yet exist)
- **Position:** UWB / GPS CSVs as expected by the position processors (under `rawData/`)

### `features/` — notable outputs

- **`features/metainfo.json`** — session duration and stream presence flags from meta extraction
- **`features/ppg/`** — PPG intermediate files: `ppg_resampled.csv` (artifact-cleaned signal), `ppg_normalized.csv` (rolling z-score used for peak detection), `ppg_resample_report.txt`, and diagnostic figures `ppg_rejected.png`, `ppg_overview.png`, `ppg_spectrogram.png`
- **`features/eda/`** — EDA intermediate files: `eda_resampled.csv` (resistance in kΩ), `eda_resample_report.txt`, `eda_overview.png`, and `SCR_events.csv` (discrete skin conductance response events)
- **`features/cognition/powerBands.csv`** — EEG band powers feeding cognition metrics
- **`features/emotions/`** — e.g. `rmsSignals.csv`, `emotionClassifierInputs.csv`, plus `original/` when using RMS editing tools
- **`features/events/playback_annotations.json`** — point/zone annotations for playback (seeded from `rawData/event.csv` once, then user-editable in the app)
- **`features/processedFeatureEvents.csv`** / **`features/processedWebFeatureEvents.csv`** — derived from `rawData/rawEvents.csv` when present
- **`features/antennas.csv`** — UWB-related export when that pipeline runs

Other steps may still write auxiliary artifacts directly under `features/`.

### `results/` — primary pipeline tables

Written by the default offline pipeline (when the corresponding raw inputs exist):

- **`processingLog.txt`** — append-only run log
- **`Emotions.csv`**, **`Cognition.csv`**, **`Arousal.csv`**, **`HeartDynamics.csv`**
- **`positions.csv`**, **`zones.csv`** (UWB), **`gpsDf.csv`** (GPS)

Playback mode looks for metric CSVs here and, for synchronized video, uses **`rawData/video.mp4`**. Event annotations are **loaded** from **`features/events/playback_annotations.json`**. When you edit annotations in Playback and save, the app writes that file and **mirrors** the same JSON to **`results/playback_annotations.json`** for convenience.

## Optional POV / ffmpeg configuration

POV movie conversion uses optional settings from, in order:

1. `hermes_standalone_config.json` in the **current working directory** (legacy name, still supported)
2. `nucleuskit_pipeline_config.json` in the **current working directory** (overrides legacy keys if both exist)
3. An extra JSON path passed with `--config` (CLI only)
4. Environment variables: `HERMES_POV_DATA_ROOT`, `HERMES_FFMPEG_DIR`, `HERMES_POV_SCREEN_ID`

The graphical **Offline processing** run uses the same merge rules from the **process working directory** and environment; it does not prompt for a config file path.

JSON keys: `pov_data_root`, `ffmpeg_dir`, `screen_id`.

## Technical documentation

The [`doc/`](doc/) folder contains detailed technical descriptions of each processing pipeline for traceability and scientific reference:

| File | Pipeline |
|------|----------|
| [`doc/overview.md`](doc/overview.md) | Pipeline orchestration, step order, session layout, shared 2 Hz timebase |
| [`doc/pipeline_heart.md`](doc/pipeline_heart.md) | Shimmer PPG → Heart Rate Dynamics (SQI + MAD artifact rejection, NeuroKit2, HRV) |
| [`doc/pipeline_arousal.md`](doc/pipeline_arousal.md) | Shimmer EDA → Arousal (cvxEDA tonic/phasic decomposition, SCR detection) |
| [`doc/pipeline_cognition.md`](doc/pipeline_cognition.md) | Hermes EEG → Cognitive Indexes (Welch power bands, engagement metrics) |
| [`doc/pipeline_emotions.md`](doc/pipeline_emotions.md) | Hermes EMG → Emotion Classification (two-stage LDA/KNN classifier) |
| [`doc/pipeline_events.md`](doc/pipeline_events.md) | Raw Events → Feature / Web Events + Playback Annotations |
| [`doc/pipeline_positioning.md`](doc/pipeline_positioning.md) | GPS and UWB Indoor Positioning (multilateration) |

## Trust and bundled models

The repository includes **joblib/pickle** model artifacts for emotion inference and channel repair. Treat them like executable data: only use models shipped with this project or that you trust.

## Tests

```bash
pytest
```

## License

See [LICENSE](LICENSE).
