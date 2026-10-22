"""Cancelling a run: the remaining steps are skipped, nothing raises."""

from __future__ import annotations

import os
import threading

from nucleuskit_toolkit.pipeline import build_pipeline


def test_cancelled_pipeline_skips_every_remaining_step(tmp_path) -> None:
    session = tmp_path / "S1"
    os.makedirs(session / "rawData")
    cancel = threading.Event()
    cancel.set()
    result = build_pipeline(skip_video_rotation=True, cancel=cancel).run(str(session))
    steps = [s for s in result.steps if not s.name.startswith("export:")]
    assert steps
    assert all(step.status == "skipped" and step.detail.get("reason") == "cancelled" for step in steps)


def test_cancel_after_first_step(tmp_path) -> None:
    session = tmp_path / "S2"
    os.makedirs(session / "rawData")
    cancel = threading.Event()
    pipeline = build_pipeline(skip_video_rotation=True, cancel=cancel)
    first = pipeline.steps[0]
    original_run = first.run

    def run_then_cancel(sess, context):  # type: ignore[no-untyped-def]
        res = original_run(sess, context)
        cancel.set()
        return res

    first.run = run_then_cancel
    result = pipeline.run(str(session))
    assert result.steps[0].detail.get("reason") != "cancelled"
    steps = [s for s in result.steps if not s.name.startswith("export:")]
    assert all(step.detail.get("reason") == "cancelled" for step in steps[1:])
