"""CLI: python -m channel_fixer rec_path target_channel_idx [--models-dir PATH]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .fix_session import fix_session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair one RMS channel using features/emotions/rmsSignals.csv."
    )
    parser.add_argument(
        "rec_path",
        type=Path,
        help="Session root (contains features/emotions/rmsSignals.csv)",
    )
    parser.add_argument(
        "target_channel_idx",
        type=int,
        help="0-based channel index (Timestamp is column 0 in CSV, not counted)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory with chan{i}_fixer.joblib (default: ./models or trainers/chanFixer/models)",
    )
    parser.add_argument("--clip-low", type=float, default=0.0)
    parser.add_argument("--clip-high", type=float, default=30.0)
    parser.add_argument(
        "--ignore-channel",
        type=int,
        action="append",
        default=None,
        help="Other channel index to exclude from features (repeatable)",
    )
    args = parser.parse_args()

    out = fix_session(
        args.rec_path,
        args.target_channel_idx,
        models_dir=args.models_dir,
        ignored_channels=args.ignore_channel,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )
    print(out)


if __name__ == "__main__":
    main()
