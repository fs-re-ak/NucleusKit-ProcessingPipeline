"""
Nucleus-Kit analytics toolkit (offline session processing).

The processing itself is re-ak-core's default pipeline (``reak_core.pipeline``): heart dynamics,
arousal, cognition, emotions, events and the emotions report. Nucleus-Kit adds its local steps
around it: POV movie conversion, session video rotation and GPS.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from reak_core.io.session import Session
from reak_core.pipeline import FunctionStep, Pipeline, RunContext, StepResult

from nucleuskit_toolkit.camera import convertPOVMovieClip, ensure_session_video_rotated_180
from nucleuskit_toolkit.config import resolve_pov_settings
from nucleuskit_toolkit.logging_utils import printError, printInfo, printWarning
from nucleuskit_toolkit.position import processGPS


class BasePipeline(ABC):
    """Minimal base from analyticsEngine.pipelines.BasePipeline."""

    def __init__(self, configuration=None):
        self.configured = False
        self.pipeName = "Unknown"

    @abstractmethod
    def processSession(self, jobDetails):
        pass

    def updateSession(self, jobDetails):
        return False

    def configure(self, config_name=None):
        pass

    def get_pipeline_name(self):
        return self.pipeName


_STEP_LOG = {
    "prepare": "[pipeline] Preparing session folders",
    "video_rotation": "[pipeline] Checking session video orientation",
    "metainfo": "[pipeline] Reading recording length",
    "heart_dynamics": "[pipeline] Computing heart dynamics",
    "arousal": "[pipeline] Computing arousal",
    "cognition": "[pipeline] Computing cognitive indexes",
    "emotions": "[pipeline] Computing emotions",
    "feature_events": "[pipeline] Processing events",
    "playback_annotations": "[pipeline] Seeding playback annotations",
    "gps": "[pipeline] Processing GPS",
}


def _local_step(name: str, fn) -> FunctionStep:
    """A Nucleus-Kit step backed by ``fn(session_path)``."""

    def run(session: Session, context: RunContext) -> dict:
        fn(session.root)
        return {}

    return FunctionStep(name, run)


class _LoggedStep:
    """Prints the Nucleus-Kit progress line before running the wrapped step.

    When ``cancel`` is set, the step is skipped instead: a cancelled run finishes the step in progress, then
    skips the remaining ones (recorded as ``skipped`` / ``cancelled`` in ``results/processing.json``).
    """

    def __init__(self, step, cancel: threading.Event | None = None):
        self._step = step
        self._cancel = cancel
        self.name = step.name

    def run(self, session: Session, context: RunContext):
        if self._cancel is not None and self._cancel.is_set():
            return StepResult(self.name, "skipped", detail={"reason": "cancelled"})
        printInfo(_STEP_LOG.get(self.name, f"[pipeline] {self.name}"))
        return self._step.run(session, context)


class _CancellableExporter:
    """Runs the wrapped exporter unless the run was cancelled (a cancelled run exports nothing)."""

    def __init__(self, exporter, cancel: threading.Event):
        self._exporter = exporter
        self._cancel = cancel
        self.name = exporter.name

    def export(self, session: Session, context: RunContext, results) -> None:
        if not self._cancel.is_set():
            self._exporter.export(session, context, results)


def build_pipeline(
    *,
    skip_video_rotation: bool = False,
    apply_nlms: bool = True,
    emotion_model: str | None = None,
    cancel: threading.Event | None = None,
) -> Pipeline:
    """re-ak-core's default pipeline with the Nucleus-Kit local steps, in Nucleus-Kit's order."""
    pipeline = Pipeline.default(emotion_model=emotion_model, apply_nlms=apply_nlms)
    if not skip_video_rotation:
        pipeline.with_step(_local_step("video_rotation", ensure_session_video_rotated_180), after="prepare")
    pipeline.with_step(_local_step("gps", processGPS))
    pipeline.steps = [_LoggedStep(step, cancel) for step in pipeline.steps]
    if cancel is not None:
        pipeline.exporters = [_CancellableExporter(exporter, cancel) for exporter in pipeline.exporters]
    return pipeline


class NucleusKitProcessingToolkit(BasePipeline):
    """
    Offline analytics pipeline for processing session data from a local folder.
    """

    def __init__(
        self,
        configuration=None,
        *,
        skip_video_rotation: bool = False,
        apply_nlms: bool = True,
        emotion_model: str | None = None,
        cancel: threading.Event | None = None,
    ):
        super().__init__(configuration)
        self.plot_data = False
        self.pipeline: Pipeline | None = None
        self._skip_video_rotation = skip_video_rotation
        self._apply_nlms = apply_nlms
        self._emotion_model = emotion_model
        self._cancel = cancel

        if configuration is None:
            self.configureAsDefault()
            self.configured = True
        else:
            printError("[NucleusKitProcessingToolkit] ERROR: Invalid configuration")

    @property
    def processingSteps(self) -> list[str]:
        """Names of the steps run, in order."""
        return [step.name for step in self.pipeline.steps] if self.pipeline else []

    def processSession(self, jobDetails):
        if not self.configured:
            printError("[NucleusKitProcessingToolkit] ERROR: Pipeline not configured")
            return False

        cachepath = jobDetails.path
        session = Session.open(cachepath)

        pov = resolve_pov_settings(getattr(jobDetails, "pov_config_json", None))
        screen_id = jobDetails.screenID or (pov.screen_id if pov else None)

        printInfo(f"[pipeline] Starting {self.pipeName}")
        printInfo("[pipeline] Converting POV movie")
        try:
            if screen_id and pov and pov.data_root and pov.ffmpeg_bin_dir:
                convertPOVMovieClip(screen_id, pov.data_root, pov.ffmpeg_bin_dir)
            elif screen_id:
                printWarning(
                    "[NucleusKitProcessingToolkit] POV screen id set but data_root/ffmpeg not configured; "
                    "skipping conversion"
                )
        except Exception as e:
            printWarning(f"[NucleusKitProcessingToolkit] WARNING: Converting movie failed: {e}")

        result = self.pipeline.run(session)
        if self._cancel is not None and self._cancel.is_set():
            printWarning("[pipeline] Run cancelled: the remaining steps were skipped")
        for step in result.steps:
            if step.status == "error":
                printError(f"[NucleusKitProcessingToolkit] ERROR in {step.name}: {step.error}")
            else:
                printInfo(f"[pipeline] {step.name}: {step.status} ({step.seconds:.1f} s)")

        printInfo("[pipeline] Session processing finished")
        return True

    def configureAsDefault(self):
        self.pipeName = "HermesDevPipe"
        self.pipeline = build_pipeline(
            skip_video_rotation=self._skip_video_rotation,
            apply_nlms=self._apply_nlms,
            emotion_model=self._emotion_model,
            cancel=self._cancel,
        )
