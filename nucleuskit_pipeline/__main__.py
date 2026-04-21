"""Entry point: ``python -m nucleuskit_pipeline`` or ``nucleuskit-pipeline`` CLI."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")


def main(argv: list[str] | None = None) -> int:
    from nucleuskit_pipeline.logging_utils import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(
        description="Nucleus-Kit offline processing pipeline (Hermes EEG/EMG, Shimmer PPG/EDA)."
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
        from nucleuskit_pipeline.pipeline import NucleusKitProcessingPipeline
        from nucleuskit_pipeline.session_job import session_job_from_folder

        job = session_job_from_folder(ns.session, pov_config_json=ns.config)
        pipe = NucleusKitProcessingPipeline(None)
        pipe.processSession(job)
        return 0

    from nucleuskit_pipeline.app import run_app

    run_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
