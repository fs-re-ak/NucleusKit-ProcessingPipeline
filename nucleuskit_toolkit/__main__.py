"""Entry point: ``python -m nucleuskit_toolkit`` or ``nucleuskit-toolkit`` CLI."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")


def main(argv: list[str] | None = None) -> int:
    from nucleuskit_toolkit.logging_utils import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(
        description="Nucleus-Kit offline processing toolkit (Hermes EEG/EMG, Shimmer PPG/EDA)."
    )
    parser.add_argument(
        "--session",
        metavar="DIR",
        help="Process this session folder and exit (no GUI).",
    )
    parser.add_argument(
        "--config",
        metavar="JSON",
        help="Optional POV / ffmpeg JSON (merged after cwd defaults).",
    )
    ns = parser.parse_args(argv)

    if ns.session:
        from nucleuskit_toolkit.pipeline import NucleusKitProcessingToolkit
        from nucleuskit_toolkit.session_job import session_job_from_folder

        job = session_job_from_folder(ns.session, pov_config_json=ns.config)
        pipe = NucleusKitProcessingToolkit(None)
        pipe.processSession(job)
        return 0

    from nucleuskit_toolkit.app import run_app

    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
