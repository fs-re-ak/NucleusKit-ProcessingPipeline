"""Smoke test: package imports."""

from __future__ import annotations


def test_import_package_version() -> None:
    import nucleuskit_pipeline

    assert nucleuskit_pipeline.__version__


def test_import_main_main() -> None:
    from nucleuskit_pipeline.__main__ import main

    assert callable(main)


def test_import_channel_fixer() -> None:
    from nucleuskit_pipeline.hermes.processorDevelopment.channel_fixer_release.channel_fixer import (
        CANONICAL_CHANNEL_NAMES,
        fix_session,
    )

    assert len(CANONICAL_CHANNEL_NAMES) == 8
    assert callable(fix_session)
