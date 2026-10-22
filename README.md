# Nucleus-Kit Processing Toolkit

Desktop application for **Nucleus-Kit** sessions: **offline** analytics (ingest session folders on disk, run the bundled processing steps, write outputs under each session's `features/` and `results/`), plus real-time streaming, playback review, and MQTT device control. The **offline** pipeline can be run from the GUI or from the command line.

Since version 2.0.0 the processing itself (heart dynamics, arousal, cognition, emotions, events, emotion models and the RMS / PPG repair tools) is [re-ak-core](https://github.com/fs-re-ak/re-ak-core), RE-AK's open processing core, shared with the re-akcas cloud processor. Nucleus-Kit adds the desktop application and its local steps: POV movie conversion, session video rotation and GPS. Every change in the results compared with Nucleus-Kit 1.1 is listed in re-ak-core's [CHANGELOG](https://github.com/fs-re-ak/re-ak-core/blob/main/CHANGELOG.md).

## Table of contents

- [Hardware mapping](#hardware-mapping)
- [Graphical interface](#graphical-interface)
  - [Tools submenu](#tools-submenu)
- [Requirements](#requirements)
- [Install](#install)
  - [From PyPI](#from-pypi)
  - [From source (development)](#from-source-development)
  - [Upgrade and uninstall](#upgrade-and-uninstall)
- [Run](#run)
  - [Desktop shortcut (Windows)](#desktop-shortcut-windows)
  - [Troubleshooting](#troubleshooting)
- [Session layout](#session-layout)
  - [Top-level folders](#top-level-folders)
  - [rawData/ — typical inputs](#rawdata--typical-inputs)
  - [features/ — notable outputs](#features--notable-outputs)
  - [results/ — primary toolkit tables](#results--primary-pipeline-tables)
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

## Graphical interface

After `python -m nucleuskit_toolkit` (or `nucleuskit-toolkit`), the window shows a sidebar on the left and the selected
page on the right. Every page has a title, a one-line description and a **Help** link to this documentation. A page
opened for the first time shows a loading screen while its libraries load (a few seconds on the first launch); they
are also loaded in the background right after start-up.

| Sidebar item | Purpose |
|------|---------|
| **Process** | Run the pipeline on one session folder or on every session of a dataset folder. The selected folder is checked as you type (read only): session name, duration and recorded signals, or for a dataset the number of sessions and how many are incomplete. Run is enabled once the folder is valid; while running it becomes **Cancel**, which finishes the step in progress and skips the rest. In a dataset, a failing session does not stop the others. |
| **Real-time viewer** | Connect to a Hermes headset over BLE (devices appear in the list as soon as the scan sees them) and an optional Shimmer over serial: stream, plot EEG / motion / Shimmer signals, and record into a session-style folder. |
| **Playback** | Review a processed session: optional `rawData/video.mp4` plus time-aligned plots from `results/*.csv`, with editable event annotations when available. |
| **Tools** | Maintenance utilities for processed sessions (see below). |
| **MQTT Controller** | Connect to a broker (e.g. Vizia Mobile), discover devices from status traffic, and send recording-related MQTT commands. |
| **Settings** | Theme (light, dark, or system), persisted under **REAK / NucleusKitToolkit** via `QSettings`. |
| **Documentation** | Opens the technical documentation in the browser. |
| **About** | Nucleus-Kit and re-ak-core versions, available emotion models, license and links. |

Leaving a page is refused while it is busy (a run in progress, a device connected or connecting, unsaved playback
annotations are offered for saving first).

### Tools submenu

| Item | Purpose |
|------|---------|
| **Channel Fixer** | Repair a single bad RMS channel in an already-processed session using the bundled channel-fix model. |
| **Channel gain adjustment** | Adjust per-channel RMS gain and zero offset against a reference distribution; updates working emotion RMS features. |
| **Revert to original** | Restore working RMS features from the frozen baseline under `features/emotions/original/`. |
| **PPG Fixer** | Re-run PPG artifact rejection and HRV computation on an already-processed session (useful when tuning rejection thresholds without reprocessing the full pipeline). |
| **EEG Regression Denoising** | Remove shared muscle noise from the EEG by regression on the EMG reference channels, then recompute the cognition outputs. |
| **Recording Merger** | Join up to 5 recordings into one session, with empty CSV rows and black video for the gaps. |

## Requirements

- **Python 3.10, 3.11 or 3.12** (3.12 recommended). Check with `python --version` (`py --version` on Windows).
  On Windows, install it from [python.org](https://www.python.org/downloads/windows/) and tick **Add python.exe to
  PATH** in the installer.
- **ffmpeg** (optional): only for POV movie conversion and session video rotation. Point Nucleus-Kit to it with
  `ffmpeg_dir` (see [Optional POV / ffmpeg configuration](#optional-pov--ffmpeg-configuration)).
- Dependencies are declared in [`pyproject.toml`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/pyproject.toml);
  [`requirements.txt`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/requirements.txt) pins the
  exact versions that were tested, GUI extras included.

Install Nucleus-Kit in its **own virtual environment**: its emotion models need scikit-learn 1.6, which must not be
upgraded by another tool installed next to it.

## Install

### From PyPI

**Windows (PowerShell)**

```powershell
python -m venv "$env:USERPROFILE\nucleuskit"
& "$env:USERPROFILE\nucleuskit\Scripts\python.exe" -m pip install --upgrade pip
& "$env:USERPROFILE\nucleuskit\Scripts\python.exe" -m pip install "nucleuskit-toolkit[gui]"
& "$env:USERPROFILE\nucleuskit\Scripts\nucleuskit-toolkit.exe" --help
```

**macOS / Linux**

```bash
python3 -m venv ~/nucleuskit
~/nucleuskit/bin/python -m pip install --upgrade pip
~/nucleuskit/bin/python -m pip install "nucleuskit-toolkit[gui]"
~/nucleuskit/bin/nucleuskit-toolkit --help
```

The `gui` extra installs the desktop interface (PySide6 / Qt 6), Bluetooth and serial streaming and MQTT. Without it
only headless processing (`--session`) is available. [re-ak-core](https://github.com/fs-re-ak/re-ak-core), the
processing core, is installed automatically.

With [pipx](https://pipx.pypa.io), which creates the environment for you and puts the commands on your PATH:

```bash
pipx install "nucleuskit-toolkit[gui]"
```

### From source (development)

Clone this repository and [re-ak-core](https://github.com/fs-re-ak/re-ak-core) side by side, then install both in
editable mode (changes in either clone apply without reinstalling). This is also the way to install before a version is
published on PyPI.

**Windows (PowerShell)**

```powershell
git clone https://github.com/fs-re-ak/re-ak-core.git
git clone https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline.git
cd NucleusKit-ProcessingPipeline
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ..\re-ak-core
.\.venv\Scripts\python.exe -m pip install -e ".[dev,gui]"
.\.venv\Scripts\python.exe -m pytest
```

**macOS / Linux**

```bash
git clone https://github.com/fs-re-ak/re-ak-core.git
git clone https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline.git
cd NucleusKit-ProcessingPipeline
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ../re-ak-core
.venv/bin/python -m pip install -e ".[dev,gui]"
.venv/bin/python -m pytest
```

The re-ak-core clone satisfies Nucleus-Kit's `re-ak-core>=1.0.1,<1.1` requirement, so pip keeps it instead of
downloading the published one. To reproduce the tested environment exactly, install `requirements.txt` instead of the
`[dev,gui]` extras (it needs re-ak-core on PyPI).

### Upgrade and uninstall

```bash
python -m pip install --upgrade "nucleuskit-toolkit[gui]"
python -m pip uninstall nucleuskit-toolkit re-ak-core
```

Use the environment's `python` (for example `"$env:USERPROFILE\nucleuskit\Scripts\python.exe"`), or delete the
environment folder to remove everything. With pipx: `pipx upgrade nucleuskit-toolkit` / `pipx uninstall nucleuskit-toolkit`.

## Run

Run the commands with the environment's executables (as below), or activate the environment first
(`.\.venv\Scripts\Activate.ps1` in PowerShell, `source .venv/bin/activate` on macOS / Linux) and drop the path.

**Graphical mode** (see [Graphical interface](#graphical-interface); needs the `gui` extra):

```powershell
& "$env:USERPROFILE\nucleuskit\Scripts\nucleuskit-toolkit.exe"
```

The same as `python -m nucleuskit_toolkit`. From a source checkout, `launch_app.bat` runs `python -m nucleuskit_toolkit`
with the first `python` on the PATH: activate the environment before running it.

**Headless mode** (one session, then exit; for automation and CI):

```powershell
nucleuskit-toolkit --session "D:\path\to\session_folder"
```

Options:

| Option | Effect |
|---|---|
| `--session DIR` | Process this session folder and exit (no GUI) |
| `--config JSON` | Optional POV / ffmpeg settings, merged after the config files of the current directory |
| `--emotion-model NAME` | Emotion model release (default `classical-emotion-2.6.0`; `nucleuskit-toolkit --help` lists them) |

Outputs are written inside the session folder (see [Session layout](#session-layout)). Existing outputs are reused:
delete a file under `results/` to recompute that step.

Replace the packaged logo anytime with your own PNG at [`nucleuskit_toolkit/ui/resources/branding/logo.png`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/nucleuskit_toolkit/ui/resources/branding/logo.png) (same path in an installed package).

### Desktop shortcut (Windows)

1. Right-click the desktop, **New → Shortcut**.
2. Location: the environment's `pythonw.exe` followed by `-m nucleuskit_toolkit`, for example
   `C:\Users\<you>\nucleuskit\Scripts\pythonw.exe -m nucleuskit_toolkit` (`pythonw` opens no console window).
3. Name it **Nucleus-Kit**, then in **Properties → Start in**, choose the folder that holds your
   `nucleuskit_toolkit_config.json`, if you use one.

### Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Call the environment's `python.exe` / `nucleuskit-toolkit.exe` by its path, as above, instead of activating |
| `No matching distribution found for re-ak-core` | That version is not on PyPI yet: install [from source](#from-source-development) |
| pip reports a conflict on `scikit-learn` | Another package needs a newer scikit-learn: install Nucleus-Kit in its own environment |
| `ModuleNotFoundError: No module named 'PySide6'` | Install the `gui` extra: `pip install "nucleuskit-toolkit[gui]"` |
| `python` is not recognized (Windows) | Use `py -3.12` instead of `python`, or reinstall Python with **Add python.exe to PATH** |

## Session layout

The pipeline treats the **session root** as a single directory (the folder you select on the Process page or pass to `--session`).

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
- **Events:** `rawEvents.csv` (processed into `features/events/processedFeatureEvents.csv` and related); `event.csv` (used to seed `features/events/playback_annotations.json` when that file does not yet exist)
- **Position:** GPS CSV as expected by the GPS processor (under `rawData/`)

### `features/` — notable outputs

- **`features/metainfo.json`** — session duration and stream presence flags from meta extraction
- **`features/ppg/`** — PPG intermediate files: `ppg_resampled.csv` (artifact-cleaned signal), `ppg_normalized.csv` (rolling z-score used for peak detection), `ppg_resample_report.txt`, and diagnostic figures `ppg_rejected.png`, `ppg_overview.png`, `ppg_spectrogram.png`
- **`features/eda/`** — EDA intermediate files: `eda_resampled.csv` (resistance in kΩ), `eda_resample_report.txt`, `eda_overview.png`, and `SCR_events.csv` (discrete skin conductance response events)
- **`features/cognition/`** — EEG band powers (`powerBands.csv`), bilateral temporal averages (`temporalBandPowers.csv`), epoch-level artefact rejection summary (`artefactStats.csv`), per-epoch signal metrics (`epochMetrics.csv`), and artefact waveform plot (`eegArtefactPlot.png`)
- **`features/emotions/`** — e.g. `rmsSignals.csv`, `emotionClassifierInputs.csv`, plus `original/` when using RMS editing tools
- **`features/events/playback_annotations.json`** — point/zone annotations for playback (seeded from `rawData/event.csv` once, then user-editable in the app)
- **`features/events/processedFeatureEvents.csv`** / **`features/events/processedWebFeatureEvents.csv`** — derived from `rawData/rawEvents.csv` when present

Other steps may still write auxiliary artifacts directly under `features/`.

### `results/` — primary toolkit tables

Written by the default offline pipeline (when the corresponding raw inputs exist). Existing outputs are reused: delete them to recompute a step.

- **`processingLog.txt`** — append-only run log
- **`Emotions.csv`**, **`Cognition.csv`**, **`Arousal.csv`**, **`HeartDynamics.csv`**
- **`emotion_model.json`** — emotion model and post-processing used for `Emotions.csv`
- **`processing.json`** — re-ak-core version, model, status of each step and a signal-quality summary
- **`gpsDf.csv`** (GPS)

Playback mode looks for metric CSVs here and, for synchronized video, uses **`rawData/video.mp4`**. Event annotations are **loaded** from **`features/events/playback_annotations.json`**. When you edit annotations in Playback and save, the app writes that file and **mirrors** the same JSON to **`results/playback_annotations.json`** for convenience.

## Optional POV / ffmpeg configuration

POV movie conversion uses optional settings from, in order:

1. `hermes_standalone_config.json` in the **current working directory** (legacy name, still supported)
2. `nucleuskit_pipeline_config.json` in the **current working directory** (legacy name, still supported)
3. `nucleuskit_toolkit_config.json` in the **current working directory** (overrides legacy keys if both exist)
4. An extra JSON path passed with `--config` (CLI only)
5. Environment variables: `HERMES_POV_DATA_ROOT`, `HERMES_FFMPEG_DIR`, `HERMES_POV_SCREEN_ID`

A run from the **Process** page uses the same merge rules from the **process working directory** and environment; it does not prompt for a config file path.

JSON keys: `pov_data_root`, `ffmpeg_dir`, `screen_id`.

## Technical documentation

The [`doc/`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/tree/main/doc/) folder contains detailed technical descriptions of each processing toolkit for traceability and scientific reference. The heart, arousal, cognition, emotions and events processing now lives in re-ak-core: those pages describe Nucleus-Kit 1.1 and will move to re-ak-core's documentation; re-ak-core's CHANGELOG lists what changed since.

| File | Pipeline |
|------|----------|
| [`doc/overview.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/overview.md) | Pipeline orchestration, step order, session layout, shared 2 Hz timebase |
| [`doc/pipeline_heart.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/pipeline_heart.md) | Shimmer PPG → Heart Rate Dynamics (SQI + MAD artifact rejection, NeuroKit2, HRV) |
| [`doc/pipeline_arousal.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/pipeline_arousal.md) | Shimmer EDA → Arousal (cvxEDA tonic/phasic decomposition, SCR detection) |
| [`doc/pipeline_cognition.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/pipeline_cognition.md) | Hermes EEG → Cognitive Indexes (Welch power bands, epoch-level artefact rejection, Engagement) |
| [`doc/pipeline_emotions.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/pipeline_emotions.md) | Hermes EMG → Emotion Classification (AVG_RMS artefact gate, two-stage LDA/KNN classifier) |
| [`doc/pipeline_events.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/pipeline_events.md) | Raw Events → Feature / Web Events + Playback Annotations |
| [`doc/pipeline_positioning.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/pipeline_positioning.md) | GPS Positioning |
| [`doc/mqtt_interface.md`](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/doc/mqtt_interface.md) | Vizia Mobile MQTT interface — status, commands, and tags |

## Trust and bundled models

The emotion models are shipped by re-ak-core as **joblib/pickle** artifacts, checked against the checksums of their release manifest before loading. Treat them like executable data: only use models shipped with re-ak-core or that you trust.

## Tests

```bash
pytest
```

`tests/test_open_core_references.py` checks that Nucleus-Kit gives exactly re-ak-core's outputs on re-ak-core's synthetic sessions; it needs a `re-ak-core` checkout next to this repository (or `RE_AK_CORE=<path>`) and is skipped otherwise.

## License

Copyright (C) 2026 RE-AK Technologies Inc.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](https://github.com/fs-re-ak/NucleusKit-ProcessingPipeline/blob/main/LICENSE) for the full text.

SPDX-License-Identifier: `GPL-3.0-or-later`. Versions up to 1.1 were released under the MIT License.
