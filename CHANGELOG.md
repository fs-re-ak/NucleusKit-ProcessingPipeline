# Changelog

## 2.1.0

### Removed

- UWB indoor positioning (`position/uwb_processor.py`, the `uwb` pipeline step): it is no longer part of Nucleus-Kit.
  Sessions are no longer given `results/positions.csv`, `results/zones.csv` or `features/antennas.csv`. GPS is unchanged.

### Fixed

- Lint: `ruff check` is clean.

## 2.0.0

The processing is now [re-ak-core](https://github.com/fs-re-ak/re-ak-core) 1.0.0a1, RE-AK's open processing core,
shared with the re-akcas cloud processor.

### Changed

- Heart dynamics, arousal, cognition, emotions, events, the emotion models (2.5.0, 2.6.0) and the maintenance tools
  (channel fixer, channel gain, revert to original, PPG fixer) come from re-ak-core. Nucleus-Kit keeps the desktop
  application, real-time streaming, playback, MQTT, recording merge, POV / video handling, UWB positioning and GPS.
- Results differ from 1.1 where the open-core decisions changed the processing. Every difference, measured on
  synthetic sessions, is listed in re-ak-core's [CHANGELOG](https://github.com/fs-re-ak/re-ak-core/blob/main/CHANGELOG.md).
  The main ones: invalid EXG samples are those at the 375,000 µV full scale or non-finite; processed events are
  written to `features/events/`; `rmsSignals.csv` is always written with raw `HEAD_*_RMS` columns.
- New outputs: `results/processing.json` (core version, model, step statuses, signal-quality summary) and
  `results/emotion_model.json`.
- re-ak-core messages appear in the console and in the GUI log.
- License: GPL-3.0-or-later (was MIT up to 1.1).

### Removed

- The processing modules now in re-ak-core (`hermes/processor`, `hermes/tools`, `shimmer/processor`,
  `shimmer/vendors`, `events`, `session`, `shared`, `hermes/rms_columns.py`, `hermes/realtime/nlms_filter.py`) and
  their tests, now covered by re-ak-core's.
- Python dependencies only used by that code: `joblib`, `cvxopt` and `neurokit2` (installed by re-ak-core).

## 1.1

Last version with its own processing code (MIT License). Its outputs are the `nk` references of re-ak-core.
