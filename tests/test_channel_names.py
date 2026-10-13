"""Raw ch1–ch8 headers and the RMS names written by the emotions pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nucleuskit_toolkit.hermes.constants import CHANNEL_NAMES, OUTPUT_RMS_COLUMNS
from nucleuskit_toolkit.hermes.processor.data_interface import load_exg_matrix
from nucleuskit_toolkit.hermes.processor.emotions_processor.report import CHANNEL_RMS_COLUMNS
from nucleuskit_toolkit.hermes.rms_columns import as_output_rms_frame, normalize_rms_dataframe


def test_output_names_match_the_emotions_report() -> None:
    assert list(OUTPUT_RMS_COLUMNS) == CHANNEL_RMS_COLUMNS
    assert len(OUTPUT_RMS_COLUMNS) == len(CHANNEL_NAMES) == 8


def test_load_exg_reorders_ch_columns(tmp_path: Path) -> None:
    path = tmp_path / "eeg.csv"
    # Timestamp, then ch8 .. ch1 so file order is reversed.
    header = "Timestamp," + ",".join(f"ch{i}" for i in range(8, 0, -1))
    # One sample: time=5, ch8=80 ... ch1=10
    values = ",".join(str(10 * i) for i in range(8, 0, -1))
    path.write_text(f"{header}\n5,{values}\n", encoding="utf-8")

    data = load_exg_matrix(str(path))
    assert data.shape == (1, 9)
    assert data[0, 0] == 5
    # Hardware order is ch1..ch8 → 10, 20, ..., 80
    np.testing.assert_allclose(data[0, 1:], [10, 20, 30, 40, 50, 60, 70, 80])


def test_load_exg_headerless_stays_positional(tmp_path: Path) -> None:
    path = tmp_path / "eeg.csv"
    path.write_text("1,10,20,30,40,50,60,70,80\n", encoding="utf-8")
    data = load_exg_matrix(str(path))
    np.testing.assert_allclose(data[0], [1, 10, 20, 30, 40, 50, 60, 70, 80])


def test_normalize_accepts_output_names_and_ch_columns() -> None:
    header = ["Timestamp", *OUTPUT_RMS_COLUMNS]
    raw = pd.DataFrame([dict(zip(header, [0.0, 1, 2, 3, 4, 5, 6, 7, 8]))])
    out = normalize_rms_dataframe(raw)
    assert list(out.columns) == ["Timestamp", *CHANNEL_NAMES]
    assert list(out.iloc[0, 1:]) == [1, 2, 3, 4, 5, 6, 7, 8]

    ch = pd.DataFrame(
        [[0.0, 8, 7, 6, 5, 4, 3, 2, 1]],
        columns=["Timestamp", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"],
    )
    named = normalize_rms_dataframe(ch)
    assert list(named.iloc[0, 1:]) == [8, 7, 6, 5, 4, 3, 2, 1]


def test_as_output_rms_frame_uses_report_names() -> None:
    df = pd.DataFrame(
        [[0.0, *range(1, 9)]],
        columns=["Timestamp", *CHANNEL_NAMES],
    )
    written = as_output_rms_frame(df)
    assert list(written.columns) == ["Timestamp", *OUTPUT_RMS_COLUMNS]
