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


class NucleusKitProcessingToolkit(BasePipeline):
    """
    Offline analytics pipeline for processing session data from a local folder.
    """

    def __init__(self, configuration=None, *, skip_video_rotation: bool = False, apply_nlms: bool = True):
        super().__init__(configuration)
        self.plot_data = False
        self.processingSteps = []
        self._skip_video_rotation = skip_video_rotation
        self._apply_nlms = apply_nlms

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

        printInfo("[NucleusKitProcessingToolkit] Converting movie")
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
            try:
                processor(cachepath)
            except BaseException as e:
                printError(
                    f"[NucleusKitProcessingToolkit] ERROR in {processor.__name__}: {type(e).__name__}: {e}"
                )
                printError(f"[NucleusKitProcessingToolkit] Traceback:\n{traceback.format_exc()}")

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
        printInfo(f"[NucleusKitProcessingToolkit] Pipeline: {self.pipeName}")
        printInfo("[NucleusKitProcessingToolkit] Assembling toolkit")

        printInfo("- Adding preparation of directory")
        self.processingSteps.append(prepareDirectory)
        if self._skip_video_rotation:
            printInfo("- Skipping session video 180° rotation (disabled by caller)")
        else:
            printInfo("- Adding one-time session video 180° rotation (rawData/video.mp4)")
            self.processingSteps.append(ensure_session_video_rotated_180)
        printInfo("- Adding extraction of meta information")
        self.processingSteps.append(extractMetaInfo)

        printInfo("- Adding physiological metrics computation")
        self.processingSteps.append(computeHeartDynamics)
        self.processingSteps.append(computeArousal)

        printInfo("- Adding cognitive metrics computation")
        self.processingSteps.append(self._bind(computeCognitiveIndexes, apply_nlms=self._apply_nlms))

        printInfo("- Adding emotional metrics computation")
        self.processingSteps.append(self._bind(computeEmotions, apply_nlms=self._apply_nlms))

        printInfo("- Adding event processing")
        self.processingSteps.append(eventProcessor)

        printInfo("- Adding playback annotation seeding from events.csv")
        self.processingSteps.append(seedPlaybackAnnotations)

        printInfo("- Adding position processing")
        self.processingSteps.append(processUWB)
        self.processingSteps.append(processGPS)

