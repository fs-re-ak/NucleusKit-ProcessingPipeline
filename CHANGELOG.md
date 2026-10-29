# Changelog

## 2.2.1

### Fixed

- Emotions no longer show a slow triangular oscillation caused by the headset clock drift: requires re-ak-core
  1.0.2, which writes `results/Emotions.csv` without blending adjacent classifier labels. Sessions reprocessed
  with this version give different `Emotions.csv` values (same columns).

## 2.2.0

### Changed

- New application layout: a sidebar (Process, Real-time viewer, Playback, Tools, MQTT Controller, Settings,
  Documentation, About) replaces the main menu and the "Main menu" buttons; every page has a title, a one-line
  description and a Help link.
- New visual design with refreshed dark and light themes: cards, step numbers, visible check boxes and radio buttons,
  outlined secondary buttons; orange is reserved for the Run pipeline action.
- **Process** page (formerly Offline processing) redesigned: one column with mode cards, the processing options
  ending with Run pipeline as the main action, and the processing log below. The selected folder is checked in the
  background without moving any file (session name, duration, recorded signals; for a dataset, the number of
  sessions and of incomplete ones). Run stays disabled until the folder is valid, and invalid folders are explained
  under the path field.
- The NLMS option is labelled "EEG channels only": it never applies to the emotion (fEMG) features.
- The emotion model selector marks the recommended release; options start from their defaults each time the page
  opens.
- Opening a page for the first time shows a loading screen while its libraries load, and those libraries are loaded
  in the background after start-up, so the Real-time viewer no longer looks frozen on first use.
- The Hermes Bluetooth scan lists devices as soon as they are found, shows a spinner and the number found, and can
  be stopped; connecting stops the scan.
- Plots follow the light / dark theme with readable axes. The real-time EEG is a stacked display in one ink colour
  with channel names on the left axis, time in seconds and a 100 µV scale bar; motion traces are X red, Y teal,
  Z yellow with a legend. Playback uses the product emotion palette, one fixed colour per emotion or metric (Fear
  and Contempt shared a colour before), with darker variants of the pale colours in the light theme; series with
  no data in the session are not shown, and the legend text stays in the text colour.

### Added

- **Cancel** while processing: the step in progress finishes, the remaining steps (and, for a dataset, the remaining
  sessions) are skipped and recorded as cancelled in `results/processing.json`.
- **About** page.

### Removed

- The main menu page and the progress bar of the processing page.

## 2.1.0

### Changed

- Published on PyPI: `pip install "nucleuskit-toolkit[gui]"`. re-ak-core is now a regular dependency
  (`re-ak-core>=1.0.1,<1.1`, from PyPI) instead of a Git URL, so Git is no longer needed to install.
- Dependencies updated to their latest versions (numpy 2.5, pandas 2.3, matplotlib 3.11, OpenCV 5, PySide6 6.12,
  pyqtgraph 0.14, bleak 3); scipy stays on 1.17 and scikit-learn on 1.6, as in re-ak-core.
- SPDX license metadata, project URLs and classifiers.

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
