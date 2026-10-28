# Nucleus-Kit Processing Toolkit

Desktop application for **Nucleus-Kit** sessions: **offline** analytics (ingest session folders on disk, run the bundled processing steps, write outputs under each session's `features/` and `results/`), plus real-time streaming, playback review, and MQTT device control. The **offline** pipeline can be run from the GUI or from the command line.

Since version 2.0.0 the processing itself (heart dynamics, arousal, cognition, emotions, events, emotion models and the RMS / PPG repair tools) is [re-ak-core](https://github.com/fs-re-ak/re-ak-core), RE-AK's open processing core, shared with the re-akcas cloud processor. Nucleus-Kit adds the desktop application and its local steps: POV movie conversion, session video rotation and GPS. Every change in the results compared with Nucleus-Kit 1.1 is listed in re-ak-core's [CHANGELOG](https://github.com/fs-re-ak/re-ak-core/blob/main/CHANGELOG.md).

## Table of contents

- [Requirements](#requirements)
- [Install](#install)
  - [From PyPI](#from-pypi)
  - [From source (development)](#from-source-development)
  - [FFmpeg](#ffmpeg)
  - [Upgrade and uninstall](#upgrade-and-uninstall)
- [Run](#run)
  - [Desktop shortcut (Windows)](#desktop-shortcut-windows)
  - [Troubleshooting](#troubleshooting)
- [Hardware mapping](#hardware-mapping)
- [Graphical interface](#graphical-interface)
  - [Tools submenu (experimental)](#tools-submenu)
- [Session layout](#session-layout)
  - [Top-level folders](#top-level-folders)
  - [rawData/ — typical inputs](#rawdata--typical-inputs)
  - [features/ — notable outputs](#features--notable-outputs)
  - [results/ — primary toolkit tables](#results--primary-pipeline-tables)
- [Optional POV movie conversion](#optional-pov-movie-conversion)
- [Technical documentation](#technical-documentation)
- [Trust and bundled models](#trust-and-bundled-models)
- [Tests](#tests)
- [License](#license)

## Requirements

- **Python 3.10, 3.11 or 3.12** (3.12 recommended). Check with `python --version` (`py --version` on Windows).
  On Windows, install it from [python.org](https://www.python.org/downloads/windows/) and tick **Add python.exe to
  PATH** in the installer.
- **FFmpeg** (a system program, not a pip package) for session video, the Recording Merger and optional POV
  conversion. Biosignal processing runs without it. Install it with the other setup steps: [FFmpeg](#ffmpeg).
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

### FFmpeg

FFmpeg is a system program. pip does not install it, and it does not go inside the virtual environment. Install it
on the machine after the Python steps above.

Nucleus-Kit calls it only for video. If a video step cannot find it, that step is skipped and the log says so; heart,
arousal, cognition, emotion and GPS processing still run. The Recording Merger does not skip: it stops until both
`ffmpeg` and `ffprobe` are on PATH.

| Use | What it does | Where Nucleus-Kit looks |
|-----|----------------|-------------------------|
| **Session video** | While processing, rotates `rawData/video.mp4` 180° once (`hflip,vflip`, re-encoded with libx264). | `ffmpeg_dir` if you set it, otherwise `ffmpeg` on PATH |
| **Playback** | When you open a session, remuxes `rawData/video.mp4` so the file header is at the start and seeking works on long recordings. | Same as session video |
| **Recording Merger** | Joins up to five videos and inserts black-screen gaps (libx264 and AAC). | PATH only. `ffmpeg_dir` is not read. Both `ffmpeg` and `ffprobe` must be on PATH. |
| **POV conversion** | Optional. Re-muxes one camera file from MKV to MP4 without re-encoding. | `ffmpeg_dir` is required. PATH alone is not enough. See [Optional POV movie conversion](#optional-pov-movie-conversion). |

Use a normal release build. The rotation and the Recording Merger re-encode with **libx264** (and AAC for merged
audio). A build compiled without those encoders will fail those steps.

After installing, open a **new** terminal and check:

```powershell
ffmpeg -version
ffprobe -version
```

Both commands should print a version line. Restart Nucleus-Kit as well: a copy that was already open still has the
old PATH.

#### Windows

**winget** (built into Windows 10 and 11):

```powershell
winget install --id Gyan.FFmpeg.Essentials -e --source winget
```

Close PowerShell, open it again, and run the two checks above. This is the [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
essentials build (libx264 and AAC). winget registers `ffmpeg` and `ffprobe` for you. If a new terminal still cannot
find them, add the `bin` folder that contains `ffmpeg.exe` to PATH, as in the manual steps.

**Manual install**, if winget is unavailable or did not put the programs on PATH:

1. Download **ffmpeg-release-essentials.zip** from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (also linked from
   [ffmpeg.org](https://ffmpeg.org/download.html#build-windows)). That build includes `ffmpeg.exe`, `ffprobe.exe`,
   libx264 and AAC.
2. Extract the zip. Inside a versioned folder there is a `bin` folder. That `bin` folder is the one that contains
   `ffmpeg.exe` and `ffprobe.exe`.
3. Add that `bin` folder to your user PATH: **Settings → System → About → Advanced system settings →
   Environment Variables**. Under **User variables**, select **Path**, **Edit**, **New**, and paste the `bin` folder.
   Paste the folder that contains `ffmpeg.exe`, not the folder above it and not the path of `ffmpeg.exe` itself.
4. Open a new PowerShell and run `ffmpeg -version` and `ffprobe -version`.

If you cannot change PATH, session video and playback can still use a configured folder. The Recording Merger
cannot; it only searches PATH.

#### macOS

```bash
brew install ffmpeg
```

Homebrew puts `ffmpeg` and `ffprobe` on PATH, with libx264.

#### Linux

Install the `ffmpeg` package from your distribution so `ffmpeg` and `ffprobe` are on PATH. On Debian and Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg
```

#### Using a folder when FFmpeg is not on PATH

Session video rotation and playback also accept a directory. `ffmpeg_dir` is the folder that **contains** the
program (`ffmpeg.exe` on Windows, `ffmpeg` on macOS and Linux). It is not the path of the program file.

Put this in `nucleuskit_toolkit_config.json` in the directory you start Nucleus-Kit from. For a desktop shortcut,
that is the **Start in** folder (see [Desktop shortcut (Windows)](#desktop-shortcut-windows)):

```json
{
  "ffmpeg_dir": "C:/ffmpeg/bin"
}
```

The environment variable `HERMES_FFMPEG_DIR` is the same folder and overrides the JSON file.

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
| `--config JSON` | Extra POV / FFmpeg settings. Applied after JSON files in the current directory and before environment variables ([Optional POV movie conversion](#optional-pov-movie-conversion)) |
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
| Log says ffmpeg was not found, or the Recording Merger asks for `ffmpeg` / `ffprobe` | Install FFmpeg, then restart the terminal and Nucleus-Kit ([FFmpeg](#ffmpeg)) |

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
| **Tools** | Experimental only. Utilities for processed sessions (see below). |
| **MQTT Controller** | Connect to a broker (e.g. Vizia Mobile), discover devices from status traffic, and send recording-related MQTT commands. |
| **Settings** | Theme (light, dark, or system), persisted under **REAK / NucleusKitToolkit** via `QSettings`. |
| **Documentation** | Opens the technical documentation in the browser. |
| **About** | Nucleus-Kit and re-ak-core versions, available emotion models, license and links. |

Leaving a page is refused while it is busy (a run in progress, a device connected or connecting, unsaved playback
annotations are offered for saving first).

<a id="tools-submenu"></a>

### Tools submenu (experimental)

Experimental only. These utilities sit outside the supported processing path.

| Item | Purpose |
|------|---------|
| **Channel Fixer** | Repair a single bad RMS channel in an already-processed session using the bundled channel-fix model. |
| **Channel gain adjustment** | Adjust per-channel RMS gain and zero offset against a reference distribution; updates working emotion RMS features. |
| **Revert to original** | Restore working RMS features from the frozen baseline under `features/emotions/original/`. |
| **PPG Fixer** | Re-run PPG artifact rejection and HRV computation on an already-processed session (useful when tuning rejection thresholds without reprocessing the full pipeline). |
| **EEG Regression Denoising** | Remove shared muscle noise from the EEG by regression on the EMG reference channels, then recompute the cognition outputs. |
| **Recording Merger** | Join up to 5 recordings into one session, with empty CSV rows and black video for the gaps. Needs `ffmpeg` and `ffprobe` on PATH ([FFmpeg](#ffmpeg)). |

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

## Optional POV movie conversion

This step is only for a POV camera file that has a UUID name and **no extension**. Nucleus-Kit renames it to
`.mkv`, re-muxes it to `.mp4` with `ffmpeg -c copy`, and puts the result back under the original name. It runs
only when all three settings below are set. If any one is missing, the step is skipped.

| Setting | JSON key | Environment variable | Meaning |
|---------|----------|----------------------|---------|
| Data folder | `pov_data_root` | `HERMES_POV_DATA_ROOT` | Folder that contains the UUID-named file |
| File name | `screen_id` | `HERMES_POV_SCREEN_ID` | That file's name, without an extension |
| FFmpeg folder | `ffmpeg_dir` | `HERMES_FFMPEG_DIR` | Folder that contains the FFmpeg program (see [FFmpeg](#ffmpeg)) |

The source path is `<pov_data_root>/<screen_id>`.

Example `nucleuskit_toolkit_config.json`:

```json
{
  "pov_data_root": "D:/recordings/pov",
  "screen_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "ffmpeg_dir": "C:/ffmpeg/bin"
}
```

Settings are merged in this order. A later source replaces the same key from an earlier one:

1. `hermes_standalone_config.json` in the current working directory (legacy name, still read)
2. `nucleuskit_pipeline_config.json` in the current working directory (legacy name, still read)
3. `nucleuskit_toolkit_config.json` in the current working directory
4. The file passed with `--config` (command line only)
5. The environment variables in the table above

The **Process** page does not ask for a config path. It reads the JSON files in its working directory and the
environment variables. From the command line, `--config` adds one more file between those JSON files and the
environment variables.

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
