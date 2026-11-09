# Nucleus-Kit processing — technical overview

Offline orchestration for one session folder. The desktop application, real-time viewer, playback and MQTT controller are described in the [README](../README.md).

**Package:** `nucleuskit_toolkit`  
**Entry point:** `python -m nucleuskit_toolkit` (GUI) or `--session <dir>` (headless)  
**Processing:** [re-ak-core](https://github.com/fs-re-ak/re-ak-core) `Pipeline.default()`, plus the local steps below

---

## What runs

`nucleuskit_toolkit/pipeline.py` (`build_pipeline`) starts from re-ak-core's default pipeline and inserts Nucleus-Kit steps around it.

POV conversion runs **before** that pipeline, and only when `pov_data_root`, `screen_id` and `ffmpeg_dir` are all set. A failure there is logged and does not stop the pipeline.

| Order | Step | Where | Output |
|---|---|---|---|
| — | POV movie conversion | `nucleuskit_toolkit/camera/processing.py` | Re-mux of one UUID-named camera file, when configured |
| 1 | `prepare` | `reak_core` | `features/`, `results/`, `meta/`, `experimentConfigs/`, `results/processingLog.txt` |
| 2 | `video_rotation` | `nucleuskit_toolkit/camera/processing.py` | `rawData/video.mp4` rotated 180° once |
| 3 | `metainfo` | `reak_core.io` | `features/metainfo.json` |
| 4 | `heart_dynamics` | `reak_core.ppg` | `results/HeartDynamics.csv` |
| 5 | `arousal` | `reak_core.eda` | `results/Arousal.csv` |
| 6 | `cognition` | `reak_core.eeg` | `results/Cognition.csv` |
| 7 | `emotions` | `reak_core.femg` | `results/Emotions.csv` |
| 8 | `feature_events` | `reak_core.sync` | `features/events/processedFeatureEvents.csv`, `processedWebFeatureEvents.csv` |
| 9 | `playback_annotations` | `reak_core.sync` | `features/events/playback_annotations.json` (created once) |
| 10 | `gps` | `nucleuskit_toolkit/position/gps_processor.py` | `results/gpsDf.csv` |
| after | emotions report | `reak_core` exporter | `features/emotions/report_jitter_<emotion>.png` and `report_emotion_distribution.png`, when classifier inputs exist |

`video_rotation` is omitted when the run asks to skip it. A cancelled run finishes the step in progress and records the rest as skipped in `results/processing.json`. An exception in one step is logged and the following steps still run.

UWB indoor positioning was removed in Nucleus-Kit 2.1.0. Sessions are not given `results/positions.csv` or `results/zones.csv`.

## Session layout

The session root is the folder passed to `--session` or chosen on the Process page. When `rawData/` is missing, re-ak-core creates it and moves every top-level entry into it except `rawData`, `features`, `results`, `meta` and `experimentConfigs`.

| Folder | Role |
|---|---|
| `rawData/` | Recordings. Common names: `eeg.csv` or `rawEEG_0.csv`, `shimmer.csv`, `event.csv` or `rawEvents.csv`, `gps.csv`, `video.mp4` |
| `features/` | Intermediate files, including `features/metainfo.json` |
| `results/` | `Emotions.csv`, `Cognition.csv`, `Arousal.csv`, `HeartDynamics.csv`, `gpsDf.csv`, `emotion_model.json`, `processing.json`, `processingLog.txt` |
| `meta/`, `experimentConfigs/` | Created for session assets and run configuration. `metainfo.json` is not written here |

Result tables share a 2 Hz timebase. `Timestamp` is seconds from the start of the recording. Existing step outputs are reused; delete the step's result file to recompute it. `results/processing.json` records the core version, the emotion model, each step's status and a signal-quality summary.

## Algorithm pages

Heart, arousal, cognition, emotions and events are documented with the core:

| Page | Step |
|---|---|
| [Overview](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/overview.md) | Session contract, invalid samples, timebase |
| [Heart](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/pipeline_heart.md) | Shimmer PPG |
| [Arousal](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/pipeline_arousal.md) | Shimmer EDA |
| [Cognition](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/pipeline_cognition.md) | Hermes EEG |
| [Emotions](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/pipeline_emotions.md) | Hermes fEMG |
| [Events](https://github.com/fs-re-ak/re-ak-core/blob/main/doc/pipeline_events.md) | Feature events and playback annotations |

Pages that stay in this repository:

| Page | Step |
|---|---|
| [pipeline_positioning.md](pipeline_positioning.md) | GPS |
| [mqtt_interface.md](mqtt_interface.md) | Vizia Mobile MQTT |

The Nucleus-Kit 1.1 descriptions, including module paths that no longer exist, are in [archive/](archive/).
