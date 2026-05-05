"""Path helpers for session-relative RMS feature files."""

from pathlib import Path


def emotions_dir(rec_path: Path) -> Path:
    return Path(rec_path) / "features" / "emotions"


def report_dir(rec_path: Path) -> Path:
    return emotions_dir(rec_path) / "chanFixReport"


def working_csv(rec_path: Path) -> Path:
    return emotions_dir(rec_path) / "rmsSignals.csv"


def report_html(rec_path: Path) -> Path:
    return report_dir(rec_path) / "report.html"
