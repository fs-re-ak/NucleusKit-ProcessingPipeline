"""
Nucleus-Kit analytics toolkit (offline session processing).

Upstream reference: analyticsEngine/pipelines/HermesAnalysisPipeline.py
"""

from __future__ import annotations

import traceback
from abc import ABC, abstractmethod

from nucleuskit_toolkit.config import resolve_pov_settings
from nucleuskit_toolkit.hermes.processor.cognition_processor import computeCognitiveIndexes
from nucleuskit_toolkit.hermes.processor.emotions_processor import computeEmotions
from nucleuskit_toolkit.events.processor import eventProcessor
from nucleuskit_toolkit.camera import convertPOVMovieClip, ensure_session_video_rotated_180
from nucleuskit_toolkit.session.layout import ensure_session_rawdata_layout, prepareDirectory
from nucleuskit_toolkit.session.meta.info import extractMetaInfo
from nucleuskit_toolkit.events import seedPlaybackAnnotations
from nucleuskit_toolkit.logging_utils import printError, printInfo, printWarning
from nucleuskit_toolkit.position import processGPS, processUWB
from nucleuskit_toolkit.shimmer import computeArousal, computeHeartDynamics


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
    "prepareDirectory": "[pipeline] Preparing session folders",
    "ensure_session_video_rotated_180": "[pipeline] Checking session video orientation",
    "extractMetaInfo": "[pipeline] Reading recording length",
    "computeHeartDynamics": "[pipeline] Computing heart dynamics",
    "computeArousal": "[pipeline] Computing arousal",
    "computeCognitiveIndexes": "[pipeline] Computing cognitive indexes",
    "computeEmotions": "[pipeline] Computing emotions",
    "eventProcessor": "[pipeline] Processing events",
    "seedPlaybackAnnotations": "[pipeline] Seeding playback annotations",
    "processUWB": "[pipeline] Processing UWB position",
    "processGPS": "[pipeline] Processing GPS",
}


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
    ):
        super().__init__(configuration)
        self.plot_data = False
        self.processingSteps = []
        self._skip_video_rotation = skip_video_rotation
        self._apply_nlms = apply_nlms
        self._emotion_model = emotion_model

        if configuration is None:
            self.configureAsDefault()
            self.configured = True
        else:
            printError("[NucleusKitProcessingToolkit] ERROR: Invalid configuration")

    def processSession(self, jobDetails):
        if not self.configured:
            printError("[NucleusKitProcessingToolkit] ERROR: Pipeline not configured")
            return False

        cachepath = jobDetails.path
        ensure_session_rawdata_layout(cachepath)

        pov = resolve_pov_settings(getattr(jobDetails, "pov_config_json", None))
        screen_id = jobDetails.screenID or (pov.screen_id if pov else None)

        printInfo(f"[pipeline] Starting {self.pipeName}")
        printInfo("[pipeline] Converting POV movie")
        try:
            if screen_id and pov and pov.data_root and pov.ffmpeg_bin_dir:
                convertPOVMovieClip(screen_id, pov.data_root, pov.ffmpeg_bin_dir)
            elif screen_id:
                printWarning(
                    "[NucleusKitProcessingToolkit] POV screen id set but data_root/ffmpeg not configured; skipping conversion"
                )
        except Exception as e:
            printWarning(f"[NucleusKitProcessingToolkit] WARNING: Converting movie failed: {e}")

        for processor in self.processingSteps:
            printInfo(_STEP_LOG.get(processor.__name__, f"[pipeline] {processor.__name__}"))
            try:
                processor(cachepath)
            except BaseException as e:
                printError(
                    f"[NucleusKitProcessingToolkit] ERROR in {processor.__name__}: {type(e).__name__}: {e}"
                )
                printError(f"[NucleusKitProcessingToolkit] Traceback:\n{traceback.format_exc()}")

        printInfo("[pipeline] Session processing finished")
        return True

    @staticmethod
    def _bind(fn, **kwargs):
        """Wrap a processor function with bound keyword arguments, preserving __name__."""
        def step(path):
            return fn(path, **kwargs)
        step.__name__ = fn.__name__
        return step

    def configureAsDefault(self):
        self.pipeName = "HermesDevPipe"
        self.processingSteps = []
        self.processingSteps.append(prepareDirectory)
        if not self._skip_video_rotation:
            self.processingSteps.append(ensure_session_video_rotated_180)
        self.processingSteps.append(extractMetaInfo)
        self.processingSteps.append(computeHeartDynamics)
        self.processingSteps.append(computeArousal)
        self.processingSteps.append(self._bind(computeCognitiveIndexes, apply_nlms=self._apply_nlms))
        self.processingSteps.append(
            self._bind(
                computeEmotions,
                apply_nlms=self._apply_nlms,
                model_name=self._emotion_model,
            )
        )
        self.processingSteps.append(eventProcessor)
        self.processingSteps.append(seedPlaybackAnnotations)
        self.processingSteps.append(processUWB)
        self.processingSteps.append(processGPS)

