"""Apply trained linear repair models to a session's RMS CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from nucleuskit_pipeline.hermes.processor.rms_original_ops import (
    append_operation,
    ensure_baseline_snapshot,
)

from .paths import emotions_dir, report_dir, report_html, working_csv
from .report import write_comparison_report
from .rms_columns import CANONICAL_CHANNEL_NAMES, normalize_rms_dataframe


def _resolve_models_dir(models_dir: Optional[Path]) -> Path:
    if models_dir is not None:
        return Path(models_dir).resolve()
    root = Path(__file__).resolve().parent.parent
    rel = root / "models"
    if rel.is_dir():
        return rel
    dev = root / "trainers" / "chanFixer" / "models"
    if dev.is_dir():
        return dev
    return rel


def _clear_report_artifacts(report_path: Path) -> None:
    if not report_path.is_dir():
        return
    for p in report_path.iterdir():
        if p.is_file():
            p.unlink()


def fix_session(
    rec_path: str | Path,
    target_channel_idx: int,
    *,
    models_dir: str | Path | None = None,
    ignored_channels: Optional[Sequence[int]] = None,
    clip_low: float = 0.0,
    clip_high: float = 30.0,
) -> Path:
    """
    Repair one RMS channel using other channels as features.

    Paths (under ``rec_path``):
      - Working file: ``features/emotions/rmsSignals.csv``
      - Frozen baseline (first tool use): ``features/emotions/original/rmsSignals.csv``
      - Report: ``features/emotions/chanFixReport/report.html``

    The HTML report compares the working file **immediately before this repair**
    to the repaired result. Re-runs operate on the current working CSV; use
    **Revert to original** in the GUI to restore the frozen baseline first.
    """
    rec_path = Path(rec_path).resolve()
    emo = emotions_dir(rec_path)
    rep = report_dir(rec_path)
    work = working_csv(rec_path)
    html_out = report_html(rec_path)

    if not work.is_file():
        raise FileNotFoundError(f"Missing RMS file (nothing to repair): {work}")

    ensure_baseline_snapshot(rec_path)

    before_df = normalize_rms_dataframe(pd.read_csv(work))

    emo.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    _clear_report_artifacts(rep)

    models_path = _resolve_models_dir(
        Path(models_dir) if models_dir is not None else None
    )
    model_file = models_path / f"chan{target_channel_idx}_fixer.joblib"
    if not model_file.is_file():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = joblib.load(model_file)

    df = before_df
    if df.shape[1] < 2:
        raise ValueError("Expected Timestamp column plus at least one channel.")

    columns = list(df.columns)
    channel_names = list(CANONICAL_CHANNEL_NAMES)
    n_chan = len(channel_names)
    if not (0 <= target_channel_idx < n_chan):
        raise IndexError(
            f"target_channel_idx={target_channel_idx} out of range for {n_chan} channels"
        )

    ignored = set(ignored_channels or [])
    other_idxs = [
        i
        for i in range(n_chan)
        if i != target_channel_idx and i not in ignored
    ]

    rms_data = df.values
    timestamps = rms_data[:, 0]
    data = rms_data[:, 1:].astype(float)

    X_hat = data[:, other_idxs]
    data[:, target_channel_idx] = model.predict(X_hat)

    ch = data[:, target_channel_idx]
    ch[(ch < clip_low) | (ch > clip_high)] = np.nan
    data[:, target_channel_idx] = ch

    out = np.concatenate((timestamps.reshape(-1, 1), data), axis=1)
    repaired = pd.DataFrame(out, columns=columns)
    repaired = repaired.interpolate()

    repaired.to_csv(work, index=False)

    write_comparison_report(
        html_out,
        before_df,
        repaired,
        target_channel_idx,
        channel_names,
    )

    append_operation(rec_path, f"channel_fix target={channel_names[target_channel_idx]}")

    return html_out
