# Nucleus-Kit Processing Pipeline

Offline analysis for **Nucleus-Kit** recording sessions: ingest session folders on disk, run the bundled analytics steps, and write results under each session’s `results/` directory.

## Hardware mapping

| Device | Signals |
|--------|---------|
| **Hermes headset** | EEG and EMG (including features used by the emotion model) |
| **Shimmer wristband** | PPG and EDA (GSR processing in code maps to the Shimmer EDA channel) |

## Requirements

- Python 3.10+
- Dependencies are listed in [`pyproject.toml`](pyproject.toml) (also mirrored in [`requirements.txt`](requirements.txt)).

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

**Graphical mode** (main menu: offline processing, settings with light/dark/system theme, placeholders for other modes):

```bash
python -m nucleuskit_pipeline
```

Requires `pip install ".[gui]"`. The app persists theme under **REAK / NucleusKitPipeline** via `QSettings`.

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

The GUI expects a **session directory** that contains at least a `rawData/` folder with the recorded files for that session. The pipeline creates or uses `features/`, `results/`, `meta/`, and `experimentConfigs/` as needed.

Processor-specific feature exports (for traceability) live under subfolders of `features/`, for example:

- `features/cognition/powerBands.csv` — EEG band powers used for cognitive metrics
- `features/emotions/rmsSignals.csv` — per-channel raw window RMS (same window as the classifier)
- `features/emotions/emotionClassifierInputs.csv` — exact model input vector (L2-normalised RMS + `AVG_RMS`) plus `PredictedLabel` / `PredictedConfidence` for each window

Other steps may still write auxiliary artifacts directly under `features/` (for example preprocessing QC files).

## Optional POV / ffmpeg configuration

POV movie conversion uses optional settings from, in order:

1. `hermes_standalone_config.json` in the **current working directory** (legacy name, still supported)
2. `nucleuskit_pipeline_config.json` in the **current working directory** (overrides legacy keys if both exist)
3. An extra JSON path passed with `--config` or chosen in the GUI
4. Environment variables: `HERMES_POV_DATA_ROOT`, `HERMES_FFMPEG_DIR`, `HERMES_POV_SCREEN_ID`

JSON keys: `pov_data_root`, `ffmpeg_dir`, `screen_id`.

## Trust and bundled models

The repository includes **joblib/pickle** model artifacts for emotion inference. Treat them like executable data: only use models shipped with this project or that you trust.

## Tests

```bash
pytest
```

## License

See [LICENSE](LICENSE).
